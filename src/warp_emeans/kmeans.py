"""Expose full-data Lloyd k-means implemented in NVIDIA Warp."""

import math

import warp as wp
from warp import utils

from . import kernels, lloyd_kernels
from .model import _validate_matrix
from .recipe import _positive_integer


def _validate_input(x, name, device=None):
    """Reject unsupported arrays and magnitudes before FP32 arithmetic."""
    _validate_matrix(x, name, device)
    invalid = wp.zeros(1, dtype=int, device=x.device)
    limit = 2.0**60 / math.sqrt(x.shape[1])
    wp.launch(lloyd_kernels.validate_magnitude_kernel, x.shape, [x, limit, invalid], device=x.device)
    if invalid.numpy()[0]:
        raise ValueError(f"{name} coordinates must have magnitude <= 2**60/sqrt(d); rescale the input")


def _assignment_policy(n, d, k, niter, device):
    """Select measured assignment strategies once during preparation."""
    if device.is_cuda and n >= 500000 and 3 <= d <= 10 and k >= 256 and (d >= 5 or k >= 512) and niter >= 10:
        return "compact"
    return "scalar"


class Kmeans:
    """Cluster finite FP32 Euclidean vectors with full-data Lloyd iterations.

    Args:
        k: Number of centroids, between one and the training row count.
        niter: Positive number of assignment-and-mean iterations.
        seed: Nonnegative signed 32-bit seed for random initialization.

    Empty clusters retain their previous centers. Exact distance ties choose
    the lowest center index. Training returns labels used by the last mean
    update; call assign for nearest-final-center labels. Autodiff is unsupported.
    """

    def __init__(self, k, niter=25, seed=1234):
        _positive_integer("k", k)
        _positive_integer("niter", niter)
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError("seed must be an integer")
        if not 0 <= seed < 2**31:
            raise ValueError("seed must be in [0, 2**31)")
        self.k, self.niter, self.seed = k, niter, seed
        self._ready = self._trained = False

    def prepare(self, x, init_centroids=None, capture=False):
        """Allocate and optionally capture a reusable Lloyd workload.

        Args:
            x: Contiguous finite (n,d) FP32 Warp matrix, retained by the model.
                Coordinate magnitudes must be at most 2**60/sqrt(d).
            init_centroids: Optional contiguous finite (k,d) matrix on the same device.
            capture: Warm up and capture training in a CUDA graph.

        Returns:
            This prepared model; run resets the centers and all accumulation state.
        """
        self._ready = self._trained = False
        _validate_input(x, "x")
        n, d = x.shape
        if self.k > n:
            raise ValueError("k must not exceed the training row count")
        if n >= 2**31 or self.k >= 2**30:
            raise ValueError("n and k must fit the signed 32-bit indexing workspace")
        if init_centroids is not None:
            _validate_input(init_centroids, "init_centroids", x.device)
            if init_centroids.shape != (self.k, d):
                raise ValueError("init_centroids must have shape (k,d)")
        if capture and not x.device.is_cuda:
            raise ValueError("CUDA graph capture requires a CUDA input array")
        self.x, self.initial, self.device = x, init_centroids, x.device
        self.centroids = wp.empty((self.k, d), dtype=float, device=x.device)
        shards = min(64, (n + 255) // 256) if x.device.is_cuda else 1
        self._sums = wp.empty((shards, self.k, d), dtype=float, device=x.device)
        self._counts = wp.empty((shards, self.k), dtype=int, device=x.device)
        self.labels = wp.empty(n, dtype=int, device=x.device)
        self._distances = wp.empty(n, dtype=float, device=x.device)
        self._order = wp.empty(2 * n, dtype=int, device=x.device) if init_centroids is None else None
        self._keys = wp.empty(2 * n, dtype=int, device=x.device) if init_centroids is None else None
        self._assign_kernel = lloyd_kernels.scalar_assignment_kernel(d)
        self._assignment = _assignment_policy(n, d, self.k, self.niter, x.device)
        if self._assignment == "compact":
            self._lower_bounds = wp.empty(n, dtype=wp.float64, device=x.device)
            self._maximum_movement = wp.empty(1, dtype=wp.float64, device=x.device)
            self._searches = wp.empty(n, dtype=int, device=x.device)
            self._previous_centers = wp.empty_like(self.centroids)
            self._flags = wp.empty(n, dtype=int, device=x.device)
            self._prefix = wp.empty_like(self._flags)
            self._active = wp.empty_like(self._flags)
            self._bounded_kernel = lloyd_kernels.bounded_assignment_kernel(d)
            self._screen_kernel = lloyd_kernels.screen_kernel(d)
            self._compact_kernel = lloyd_kernels.compact_assignment_kernel(d)
        self.graph = None
        if capture:
            self._execute()
            wp.synchronize_device(x.device)
            with wp.ScopedCapture(device=x.device) as captured:
                self._execute()
            self.graph = captured.graph
        self._ready = True
        return self

    def _assign(self, x, labels, distances, accumulate):
        wp.launch(
            self._assign_kernel,
            x.shape[0],
            [x, self.centroids, self._sums, self._counts, labels, distances, accumulate],
            device=self.device,
            block_dim=128,
        )

    def _execute(self):
        if self.initial is None:
            wp.launch(
                kernels.shuffle_kernel,
                self.x.shape[0],
                [self._keys, self._order, self.seed],
                device=self.device,
            )
            utils.radix_sort_pairs(self._keys, self._order, self.x.shape[0])
            wp.launch(
                lloyd_kernels.initialize_kernel,
                self.centroids.shape,
                [self.x, self._order, self.centroids],
                device=self.device,
            )
        else:
            wp.copy(self.centroids, self.initial)
        if self._assignment == "compact":
            self._lower_bounds.zero_()
            self._maximum_movement.zero_()
            self._searches.zero_()
            self.labels.zero_()
        for iteration in range(self.niter):
            self._sums.zero_()
            self._counts.zero_()
            if self._assignment == "compact":
                if iteration == 0:
                    wp.launch(
                        self._bounded_kernel,
                        self.x.shape[0],
                        [
                            self.x,
                            self.centroids,
                            self._sums,
                            self._counts,
                            self.labels,
                            self._distances,
                            self._lower_bounds,
                            self._maximum_movement,
                            self._searches,
                        ],
                        device=self.device,
                        block_dim=128,
                    )
                else:
                    wp.launch(
                        self._screen_kernel,
                        self.x.shape[0],
                        [
                            self.x,
                            self.centroids,
                            self.labels,
                            self._distances,
                            self._lower_bounds,
                            self._maximum_movement,
                            self._flags,
                        ],
                        device=self.device,
                        block_dim=128,
                    )
                    utils.array_scan(self._flags, self._prefix)
                    wp.launch(
                        lloyd_kernels.compact_kernel,
                        self.x.shape[0],
                        [self._flags, self._prefix, self._active],
                        device=self.device,
                    )
                    wp.launch(
                        self._compact_kernel,
                        self.x.shape[0],
                        [
                            self.x,
                            self.centroids,
                            self.labels,
                            self._distances,
                            self._lower_bounds,
                            self._searches,
                            self._active,
                            self._prefix,
                        ],
                        device=self.device,
                        block_dim=128,
                    )
                    wp.launch(
                        lloyd_kernels.accumulate_kernel,
                        self.x.shape,
                        [self.x, self.labels, self._sums, self._counts],
                        device=self.device,
                    )
                wp.copy(self._previous_centers, self.centroids)
            else:
                self._assign(self.x, self.labels, self._distances, 1)
            wp.launch(
                lloyd_kernels.update_kernel,
                self.centroids.shape,
                [self.centroids, self._sums, self._counts],
                device=self.device,
            )
            if self._assignment == "compact":
                self._maximum_movement.zero_()
                wp.launch(
                    lloyd_kernels.movement_kernel,
                    self.k,
                    [self._previous_centers, self.centroids, self._maximum_movement],
                    device=self.device,
                )

    def run(self):
        """Reset and train, returning centers and labels from the final mean update.

        Returns:
            Device arrays containing (k,d) centers and n labels. Repeated calls
            overwrite these arrays. Retained inputs must keep their shape,
            device, storage, and validated finite magnitude range.
        """
        if not self._ready:
            raise RuntimeError("call prepare before run")
        if self.graph is None:
            self._execute()
        else:
            wp.capture_launch(self.graph)
        self._trained = True
        return self.centroids, self.labels

    def train(self, x, init_centroids=None):
        """Prepare and train once, returning centers and last-update labels."""
        return self.prepare(x, init_centroids=init_centroids).run()

    def assign(self, x):
        """Return squared distances and nearest-final-center labels on device."""
        if not self._trained:
            raise RuntimeError("train or run before assign")
        _validate_input(x, "x", self.device)
        if x.shape[1] != self.centroids.shape[1]:
            raise ValueError("x must have the training dimension")
        labels = wp.empty(x.shape[0], dtype=int, device=self.device)
        distances = wp.empty(x.shape[0], dtype=float, device=self.device)
        self._assign(x, labels, distances, 0)
        return distances, labels
