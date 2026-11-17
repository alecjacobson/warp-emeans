"""Expose Euclidean e-means with a GPU-resident training loop."""

import math
from dataclasses import replace

import warp as wp
from warp import utils

from . import kernels
from .recipe import BatchLaw, resolve_recipe


def _validate_matrix(x, name, device=None):
    if not isinstance(x, wp.array) or x.dtype != wp.float32 or x.ndim != 2:
        raise TypeError(f"{name} must be a two-dimensional float32 Warp array")
    if not x.is_contiguous or min(x.shape) < 1:
        raise ValueError(f"{name} must be nonempty and contiguous")
    if x.requires_grad:
        raise ValueError("Discrete clustering does not support automatic differentiation")
    if device is not None and x.device != device:
        raise ValueError(f"{name} must be on {device}")
    invalid = wp.zeros(1, dtype=wp.int32, device=x.device)
    wp.launch(kernels.validate_kernel, x.shape, [x, invalid], device=x.device)
    if invalid.numpy()[0]:
        raise ValueError(f"{name} must contain only finite values")


class Emeans:
    """Cluster finite FP32 Euclidean vectors using Warp kernels.

    Args:
        k: Number of centroids, between one and the training row count.
        nepochs: Positive training passes; a fractional pass precedes the final
            full pass when the budget is greater than one.
        batch_size: Optional positive batch size no larger than the dataset.
        alpha0: Optional initial EMA step in (0, 1].
        batch_law: Upstream auto, quality, or throughput batch policy.
        unused_threshold: Nonnegative revival threshold; None disables revival
            at d<=2 and uses 0.1 above it, following the pinned upstream code.
        seed: Nonnegative signed 32-bit seed for Warp shuffling and revival.

    All clustering arithmetic, shuffling, and revival run in Warp. Outputs stay
    on the input device. Labels are moving last-pass assignments; call assign
    for nearest-final-center labels. Autodiff is unsupported. The Warp random
    stream differs from NumPy/PyTorch, so equal seeds do not imply equal runs.
    """

    def __init__(
        self,
        k,
        nepochs=3.0,
        batch_size=None,
        alpha0=None,
        batch_law=BatchLaw.AUTO,
        unused_threshold=None,
        seed=1234,
    ):
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError("seed must be an integer")
        if not 0 <= seed < 2**31:
            raise ValueError("seed must be in [0, 2**31)")
        self.k, self.nepochs, self.batch_size = k, nepochs, batch_size
        self.alpha0, self.batch_law, self.unused_threshold = alpha0, batch_law, unused_threshold
        self.seed = seed
        self._ready = False
        self._trained = False

    def prepare(self, x, init_centroids=None, orders=None, capture=False):
        """Allocate and optionally capture a reusable training workload.

        Args:
            x: Contiguous finite (n,d) FP32 Warp array, retained by this model.
            init_centroids: Optional finite (k,d) centers on the same device.
            orders: Optional (number_of_passes,n) int32 permutations for
                controlled comparisons. Validated once on the host.
            capture: Capture all training work in a CUDA graph. Preparation
                executes a warmup and captures the graph; run reinitializes
                state and executes it. Preparation cost must be reported.

        Returns:
            This prepared model; call run to obtain a fresh training result.
        """
        self._ready = self._trained = False
        _validate_matrix(x, "x")
        self.x, self.device = x, x.device
        n, d = x.shape
        self.recipe = resolve_recipe(
            n, d, self.k, self.nepochs, self.batch_size, self.alpha0, self.batch_law, self.unused_threshold
        )
        if init_centroids is not None:
            _validate_matrix(init_centroids, "init_centroids", x.device)
            if init_centroids.shape != (self.k, d):
                raise ValueError("init_centroids must have shape (k,d)")
        if capture and not self.device.is_cuda:
            raise ValueError("CUDA graph capture requires a CUDA input array")
        if orders is not None:
            if (
                not isinstance(orders, wp.array)
                or orders.dtype != wp.int32
                or orders.shape != (len(self.recipe.epoch_rows), n)
                or not orders.is_contiguous
                or orders.device != self.device
            ):
                raise ValueError(
                    "orders must be contiguous int32 permutations with shape (passes,n) on the input device"
                )
            # Only this optional comparison facility uses NumPy. The default
            # clustering path does all permutation generation on the device.
            import numpy as np

            if not np.all(np.sort(orders.numpy(), axis=1) == np.arange(n)):
                raise ValueError("each orders row must be a permutation of [0,n)")
        self.initial, self.orders = init_centroids, orders
        self.centroids = wp.empty((self.k, d), dtype=float, device=self.device)
        self.usage = wp.empty(self.k, dtype=float, device=self.device)
        self.sums = wp.zeros((self.k, d), dtype=float, device=self.device)
        self.counts = wp.zeros(self.k, dtype=int, device=self.device)
        self.closing_sums = wp.zeros_like(self.sums)
        self.closing_counts = wp.zeros_like(self.counts)
        self.labels = wp.empty(n, dtype=int, device=self.device)
        self.distances = wp.empty(n, dtype=float, device=self.device)
        self.order = wp.empty(2 * n, dtype=int, device=self.device)
        self.shuffle_keys = wp.empty(2 * n, dtype=int, device=self.device)
        self.split_keys = wp.empty(2 * self.k, dtype=float, device=self.device)
        self.donors = wp.empty(2 * self.k, dtype=int, device=self.device)
        self.dead = wp.empty(self.k, dtype=int, device=self.device)
        self.prefix = wp.empty_like(self.dead)
        self.total_usage = wp.empty(1, dtype=float, device=self.device)
        self.split_count = wp.zeros(1, dtype=int, device=self.device)
        self._peak = wp.full(1, self.recipe.alpha0, dtype=float, device=self.device)
        self._assign_kernel = (
            kernels.cooperative_assignment_kernel(d) if self.device.is_cuda else kernels.assignment_kernel(d)
        )
        self._assignment_width = 32 if self.device.is_cuda else 1
        self.graph = None
        if capture:
            self._execute()
            wp.synchronize_device(self.device)
            with wp.ScopedCapture(device=self.device) as captured:
                self._execute()
            self.graph = captured.graph
        self._ready = True
        return self

    def _shuffle(self, seed):
        n = self.x.shape[0]
        wp.launch(kernels.shuffle_kernel, n, [self.shuffle_keys, self.order, seed], device=self.device)
        utils.radix_sort_pairs(self.shuffle_keys, self.order, n)

    def _execute(self):
        n, d = self.x.shape
        self.sums.zero_()
        self.counts.zero_()
        self.closing_sums.zero_()
        self.closing_counts.zero_()
        self.split_count.zero_()
        self.labels.fill_(-1)
        if self.initial is None:
            self._shuffle(self.seed)
            wp.launch(
                kernels.initialize_kernel,
                (self.k, d),
                [self.x, self.order, self.centroids, self.usage],
                device=self.device,
            )
        else:
            wp.copy(self.centroids, self.initial)
            self.usage.fill_(2.0**-16)
        step = 0
        for epoch, rows in enumerate(self.recipe.epoch_rows):
            if self.orders is None:
                self._shuffle((self.seed * 1000003 + epoch) % (2**31))
            else:
                wp.copy(self.order, self.orders.flatten(), src_offset=epoch * n, count=n)
            closing = int(epoch == len(self.recipe.epoch_rows) - 1)
            for offset in range(0, rows, self.recipe.batch_size):
                size = min(self.recipe.batch_size, rows - offset)
                wp.launch(
                    self._assign_kernel,
                    (size, 32) if self.device.is_cuda else size,
                    [
                        self.x,
                        self.centroids,
                        self.order,
                        offset,
                        self.k,
                        self.sums,
                        self.counts,
                        self.labels,
                        self.distances,
                        1,
                        1,
                    ],
                    device=self.device,
                    block_dim=self._assignment_width,
                )
                schedule = 0.1 + 0.45 * (1 + math.cos(math.pi * step / self.recipe.total_steps))
                wp.launch(
                    kernels.update_kernel,
                    self.k,
                    [
                        self.centroids,
                        self.usage,
                        self.sums,
                        self.counts,
                        self.closing_sums,
                        self.closing_counts,
                        schedule,
                        self._peak,
                        closing,
                    ],
                    device=self.device,
                )
                step += 1
                if self.recipe.unused_threshold > 0 and step % 5 == 0:
                    utils.array_sum(self.usage, out=self.total_usage)
                    wp.launch(
                        kernels.candidates_kernel,
                        self.k,
                        [
                            self.usage,
                            self.total_usage,
                            self.recipe.unused_threshold,
                            self.split_keys,
                            self.donors,
                            self.dead,
                            (self.seed + step) % (2**31),
                        ],
                        device=self.device,
                    )
                    utils.array_scan(self.dead, self.prefix)
                    utils.radix_sort_pairs(self.split_keys, self.donors, self.k)
                    wp.launch(
                        kernels.split_kernel,
                        self.k,
                        [
                            self.centroids,
                            self.usage,
                            self.donors,
                            self.split_keys,
                            self.dead,
                            self.prefix,
                            self.split_count,
                        ],
                        device=self.device,
                    )
        wp.launch(
            kernels.finalize_kernel,
            (self.k, d),
            [self.centroids, self.closing_sums, self.closing_counts],
            device=self.device,
        )

    def run(self, alpha0=None):
        """Reinitialize and run prepared training, returning centers and moving labels.

        Args:
            alpha0: Optional new peak step in (0,1] for this and later runs.
                Reuses a captured graph while changing the cosine amplitude.

        Returns:
            Device arrays containing (k,d) centroids and n last-pass labels.
            Unvisited rows have label -1 for budgets below one pass. Repeated
            runs reuse these arrays and overwrite their previous contents.
        """
        if not self._ready:
            raise RuntimeError("call prepare before run")
        if alpha0 is not None:
            if not math.isfinite(alpha0) or not 0 < alpha0 <= 1:
                raise ValueError("alpha0 must be finite and in (0,1]")
            self.recipe = replace(self.recipe, alpha0=float(alpha0))
            self._peak.fill_(float(alpha0))
        if self.graph is None:
            self._execute()
        else:
            wp.capture_launch(self.graph)
        self._trained = True
        return self.centroids, self.labels

    def train(self, x, init_centroids=None):
        """Prepare and run once, returning centers and moving last-pass labels."""
        return self.prepare(x, init_centroids=init_centroids).run()

    def assign(self, x):
        """Return squared distances and exact nearest-final-center labels on device."""
        if not self._trained:
            raise RuntimeError("train or run before assign")
        _validate_matrix(x, "x", self.device)
        if x.shape[1] != self.centroids.shape[1]:
            raise ValueError("x must have the training dimension")
        labels = wp.empty(x.shape[0], dtype=int, device=self.device)
        distances = wp.empty(x.shape[0], dtype=float, device=self.device)
        wp.launch(
            self._assign_kernel,
            (x.shape[0], 32) if self.device.is_cuda else x.shape[0],
            [x, self.centroids, self.order, 0, self.k, self.sums, self.counts, labels, distances, 0, 0],
            device=self.device,
            block_dim=self._assignment_width,
        )
        return distances, labels
