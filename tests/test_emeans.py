"""Check the complete recipe against the independently executed upstream code."""

import sys
from pathlib import Path

import numpy as np
import pytest
import warp as wp
from warp import utils

from warp_emeans import Emeans, kernels
from warp_emeans.recipe import resolve_recipe

REFERENCE = Path(__file__).resolve().parents[1] / ".cache/reference"
sys.path.insert(0, str(REFERENCE))
wp.init()
DEVICES = ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("d", [2, 3, 5, 10, 100])
def test_reference_parity(device, d):
    """Match upstream on shared orders including tails and fractional passes."""
    upstream = pytest.importorskip("e_means")
    rng = np.random.default_rng(91)
    x = rng.normal(size=(71, d)).astype(np.float32)
    initial = x[[0, 13, 27, 41, 55]]
    recipe = resolve_recipe(71, d, 5, 2.5, batch_size=13, alpha0=0.2, unused_threshold=0)
    orders = np.stack([rng.permutation(len(x)) for _ in recipe.epoch_rows]).astype(np.int32)
    reference = upstream.NumpyEmeans(d, 5, nepochs=2.5, batch_size=13, alpha0=0.2, unused_threshold=0)
    reference.begin(len(x), initial.copy())
    expected_labels = np.full(len(x), -1)
    for epoch, rows in enumerate(recipe.epoch_rows):
        for begin in range(0, rows, recipe.batch_size):
            ids = orders[epoch, begin : min(begin + recipe.batch_size, rows)]
            stats = reference.accumulate(x[ids])
            if epoch == len(recipe.epoch_rows) - 1:
                expected_labels[ids] = stats["labels"]
            reference.update(stats)
        reference.end_epoch()
    expected_centers = reference.finalize()
    model = Emeans(5, nepochs=2.5, batch_size=13, alpha0=0.2, unused_threshold=0)
    model.prepare(
        wp.array(x, device=device), wp.array(initial, device=device), wp.array(orders, device=device)
    )
    centers, labels = model.run()
    np.testing.assert_array_equal(labels.numpy(), expected_labels)
    np.testing.assert_allclose(centers.numpy(), expected_centers, atol=2e-6, rtol=2e-6)
    distances, assigned = model.assign(wp.array(x, device=device))
    squared = ((x.astype(float)[:, None, :] - centers.numpy().astype(float)[None, :, :]) ** 2).sum(axis=2)
    np.testing.assert_array_equal(assigned.numpy(), squared.argmin(axis=1))
    np.testing.assert_allclose(distances.numpy(), squared.min(axis=1), rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_revival_conserves_mass(device):
    """Split distinct live donors and conserve their mass while moving dead slots."""
    centers_np = np.array([[2, 4], [6, 8], [90, 90], [95, 95]], dtype=np.float32)
    centers = wp.array(centers_np, device=device)
    usage = wp.array(np.array([10, 4, 0, 0], dtype=np.float32), device=device)
    total = wp.array(np.array([14], dtype=np.float32), device=device)
    keys, donors = wp.empty(8, dtype=float, device=device), wp.empty(8, dtype=int, device=device)
    dead, prefix = wp.empty(4, dtype=int, device=device), wp.empty(4, dtype=int, device=device)
    splits = wp.zeros(1, dtype=int, device=device)
    wp.launch(kernels.candidates_kernel, 4, [usage, total, 0.1, keys, donors, dead, 7], device=device)
    utils.array_scan(dead, prefix)
    utils.radix_sort_pairs(keys, donors, 4)
    wp.launch(kernels.split_kernel, 4, [centers, usage, donors, keys, dead, prefix, splits], device=device)
    np.testing.assert_array_equal(splits.numpy(), [2])
    np.testing.assert_allclose(usage.numpy().sum(), 14)
    np.testing.assert_array_equal(np.sort(usage.numpy()), [2, 2, 5, 5])
    np.testing.assert_allclose(centers.numpy()[:2], centers_np[:2] * [0.9999, 1.0001], rtol=1e-6)
    np.testing.assert_allclose(
        centers.numpy()[2:], centers_np[donors.numpy()[:2]] * [1.0001, 0.9999], rtol=1e-6
    )


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA graph required")
def test_graph_reinitializes_state():
    """Exercise revival and ensure graph replay resets all training state."""
    x = np.random.default_rng(8).normal(size=(1003, 3)).astype(np.float32)
    initial = np.zeros((32, 3), dtype=np.float32)
    model = Emeans(32, nepochs=2.5, batch_size=97, unused_threshold=0.2)
    model.prepare(wp.array(x, device="cuda"), wp.array(initial, device="cuda"), capture=True)
    if model.graph is None or model._assignment_width != 32:
        pytest.fail("The cooperative CUDA graph path was not selected")
    first = model.run()[0].numpy().copy()
    splits = model.split_count.numpy()[0]
    if splits <= 0:
        pytest.fail("The revival path was not exercised")
    second = model.run()[0].numpy()
    np.testing.assert_allclose(first, second, atol=3e-5, rtol=3e-5)
    np.testing.assert_array_equal(model.split_count.numpy(), [splits])
    if not np.isfinite(second).all():
        pytest.fail("Nonfinite centers")


@pytest.mark.parametrize("device", DEVICES)
def test_fractional_labels_and_last_means(device):
    """Keep unvisited labels at minus one and return means of captured labels."""
    x = np.random.default_rng(4).normal(size=(101, 2)).astype(np.float32)
    model = Emeans(4, nepochs=0.5, batch_size=17)
    centers, labels = model.train(wp.array(x, device=device))
    labels, centers = labels.numpy(), centers.numpy()
    np.testing.assert_equal(np.count_nonzero(labels >= 0), round(101 * 0.5))
    for cluster in range(4):
        if np.any(labels == cluster):
            np.testing.assert_allclose(centers[cluster], x[labels == cluster].mean(axis=0), atol=2e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_ties_and_unused_centers(device):
    """Choose the first center on exact ties and preserve unused centers."""
    x = np.zeros((16, 3), dtype=np.float32)
    initial = np.array([[0, 0, 0], [0, 0, 0], [100, 100, 100]], dtype=np.float32)
    model = Emeans(3, nepochs=1, batch_size=16, alpha0=1, unused_threshold=0)
    centers, labels = model.train(wp.array(x, device=device), wp.array(initial, device=device))
    np.testing.assert_array_equal(labels.numpy(), np.zeros(16))
    np.testing.assert_array_equal(centers.numpy(), initial)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"k": 0},
        {"k": 17},
        {"k": 2, "alpha0": 0},
        {"k": 2, "nepochs": float("nan")},
        {"k": 2, "batch_size": 0},
        {"k": 2, "unused_threshold": -1},
    ],
)
def test_invalid_recipe(kwargs):
    """Reject invalid requests before launching the training loop."""
    with pytest.raises(ValueError):
        Emeans(**kwargs).train(wp.zeros((16, 2), device="cpu"))


def test_nonfinite_input():
    """Reject NaN input before assignment can return plausible labels."""
    x = np.ones((16, 2), dtype=np.float32)
    x[3, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        Emeans(2).train(wp.array(x, device="cpu"))
