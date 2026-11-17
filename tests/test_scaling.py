"""Check censoring in the scientific analysis rather than fabricate point targets."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from fit_intervals import valley_interval


def test_flat_valley_unidentified():
    """Return unbounded uncertainty when the entire step curve is flat."""
    lo, hi = valley_interval([0.01, 0.1, 1], [1, 1, 1])
    np.testing.assert_equal([lo, hi], [-np.inf, np.inf])


def test_boundary_valley_one_sided():
    """Keep a minimum at the search boundary one-sided and interpolate its crossing."""
    lo, hi = valley_interval([0.01, 0.1, 1], [1, 1.1, 2])
    np.testing.assert_equal(lo, -np.inf)
    np.testing.assert_allclose(hi, np.log(0.01) + 0.02 * np.log(10), rtol=1e-12)
    lo, hi = valley_interval([0.01, 0.1, 1], [2, 1.1, 1])
    np.testing.assert_equal(hi, np.inf)
    np.testing.assert_allclose(lo, -0.02 * np.log(10), rtol=1e-12)
