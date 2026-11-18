"""Verify the optional external Lloyd baseline on representative CUDA cases."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from flash_reference import compiled_assignment_evidence, provenance, train

torch = pytest.importorskip("torch")
pytest.importorskip("flash_kmeans")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Flash-KMeans requires CUDA")


@pytest.mark.parametrize("d", [2, 3, 5, 6, 10, 100])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_flash_lloyd(d, dtype):
    """Match independent updates with tail rows, tail dimensions and shared centers."""
    rng = np.random.default_rng(710 + d)
    anchors = rng.normal(size=(17, d)) * 4
    points = anchors[np.arange(517) % 17] + rng.normal(size=(517, d)) * 0.01
    data = torch.tensor(points, dtype=dtype, device="cuda")
    initial = data[:17].clone()
    quantized = data.float().cpu().numpy().astype(np.float64)
    expected = initial.float().cpu().numpy().astype(np.float64)
    original = expected.copy()
    for _ in range(4):
        labels = ((quantized[:, None] - expected[None]) ** 2).sum(axis=2).argmin(axis=1)
        for cluster in range(17):
            members = quantized[labels == cluster]
            if len(members):
                expected[cluster] = members.mean(axis=0)
        if dtype == torch.float16:
            expected = expected.astype(np.float16).astype(np.float64)
    actual, assigned = train(data, initial, 4)
    np.testing.assert_array_equal(assigned.cpu().numpy(), labels)
    np.testing.assert_allclose(actual.float().cpu().numpy(), expected, atol=2e-5, rtol=2e-6)
    np.testing.assert_array_equal(initial.float().cpu().numpy(), original)
    np.testing.assert_equal(provenance()["flash_kmeans"], "0.3.1")
    if not compiled_assignment_evidence():
        pytest.fail("No actual Triton assignment kernel was exercised")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_flash_empty_ties_and_full_iterations(dtype):
    """Retain empty centers, break exact ties and force iterations after convergence."""
    data = torch.zeros((259, 3), dtype=dtype, device="cuda")
    initial = torch.tensor([[0, 0, 0], [0, 0, 0], [1, 2, 3]], dtype=dtype, device="cuda")
    # The adapter checks the returned count; already-converged data must run all 3.
    centers, labels = train(data, initial, 3)
    np.testing.assert_array_equal(centers.float().cpu().numpy(), initial.float().cpu().numpy())
    np.testing.assert_array_equal(labels.cpu().numpy(), np.zeros(259, dtype=np.int32))
