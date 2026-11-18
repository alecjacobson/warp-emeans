"""Validate Lloyd training independently of the e-means implementation."""

import numpy as np
import pytest
import warp as wp

from warp_emeans import Kmeans

wp.init()
DEVICES = ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])


def reference_lloyd(x, initial, niter):
    """Compute full-data Lloyd updates using independent FP64 NumPy arithmetic."""
    centers = initial.astype(np.float64).copy()
    points = x.astype(np.float64)
    for _ in range(niter):
        squared = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = squared.argmin(axis=1)
        for cluster in range(len(centers)):
            if np.any(labels == cluster):
                centers[cluster] = points[labels == cluster].mean(axis=0)
    return centers, labels


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("d", [2, 3, 5, 10, 32, 100])
def test_lloyd_reference(device, d):
    """Match independent Lloyd iterations, tail dimensions and final assignment."""
    x = np.random.default_rng(715 + d).normal(size=(517, d)).astype(np.float32)
    initial = x[::29][:17].copy()
    expected, labels = reference_lloyd(x, initial, 4)
    model = Kmeans(17, niter=4)
    centers, actual_labels = model.train(wp.array(x, device=device), wp.array(initial, device=device))
    np.testing.assert_array_equal(actual_labels.numpy(), labels)
    np.testing.assert_allclose(centers.numpy(), expected, atol=2e-6, rtol=2e-6)
    squared = ((x.astype(float)[:, None, :] - centers.numpy().astype(float)[None, :, :]) ** 2).sum(axis=2)
    distances, nearest = model.assign(wp.array(x, device=device))
    np.testing.assert_array_equal(nearest.numpy(), squared.argmin(axis=1))
    np.testing.assert_allclose(distances.numpy(), squared.min(axis=1), atol=1e-5, rtol=2e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_lloyd_ties_empty(device):
    """Choose the smallest tied index and preserve empty centers exactly."""
    x = np.zeros((259, 3), dtype=np.float32)
    initial = np.array([[0, 0, 0], [0, 0, 0], [1, 2, 3]], dtype=np.float32)
    centers, labels = Kmeans(3, niter=3).train(wp.array(x, device=device), wp.array(initial, device=device))
    np.testing.assert_array_equal(centers.numpy(), initial)
    np.testing.assert_array_equal(labels.numpy(), np.zeros(259, dtype=np.int32))


@pytest.mark.parametrize("device", DEVICES)
def test_lloyd_initialization(device):
    """Draw initial centers on device and reproduce the same seeded run."""
    x = wp.array(np.random.default_rng(3).normal(size=(301, 5)).astype(np.float32), device=device)
    model = Kmeans(11, niter=2, seed=81).prepare(x)
    first = model.run()[0].numpy().copy()
    second = model.run()[0].numpy()
    np.testing.assert_allclose(first, second, atol=2e-6, rtol=2e-6)
    np.testing.assert_equal(np.isfinite(first).all(), True)


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA graph required")
def test_lloyd_graph_reset():
    """Exercise captured Lloyd iterations and reset modified buffers between replays."""
    x = wp.array(np.random.default_rng(1).normal(size=(1003, 5)).astype(np.float32), device="cuda")
    model = Kmeans(35, niter=4).prepare(x, capture=True)
    if model.graph is None:
        pytest.fail("CUDA graph path not selected")
    first = model.run()[0].numpy().copy()
    model.centroids.fill_(1e4)
    model._sums.fill_(1e4)
    model._counts.fill_(1000)
    second = model.run()[0].numpy()
    np.testing.assert_allclose(first, second, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("kwargs,error", [({"k": 0}, ValueError), ({"k": True}, TypeError), ({"k": 2,"niter": 0}, ValueError), ({"k": 2,"niter": 1.5}, TypeError), ({"k": 2,"seed": -1}, ValueError)])
def test_lloyd_invalid_options(kwargs, error):
    """Reject invalid clustering parameters before allocating a workload."""
    with pytest.raises(error):
        Kmeans(**kwargs)


def test_lloyd_invalid_data():
    """Reject invalid input arrays, unsupported gradients and unprepared calls."""
    model = Kmeans(2)
    with pytest.raises(RuntimeError, match="prepare"):
        model.run()
    with pytest.raises(RuntimeError, match="train"):
        model.assign(wp.zeros((10, 3), device="cpu"))
    with pytest.raises(ValueError, match="row count"):
        Kmeans(11).train(wp.zeros((10, 3), device="cpu"))
    with pytest.raises(ValueError, match="finite"):
        model.train(wp.full((10, 3), float("nan"), device="cpu"))
    with pytest.raises(ValueError, match="differentiation"):
        model.train(wp.zeros((10, 3), device="cpu", requires_grad=True))
