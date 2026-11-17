"""Check split independence and stable synthetic prefixes used by the law fit."""

import numpy as np
import pytest

from warp_emeans.datasets import synthetic


@pytest.mark.parametrize("family", ["uniform", "gaussian", "mixture", "manifold"])
def test_nested_independent_splits(family):
    """Keep smaller n as a prefix while generating disjoint validation samples."""
    small = synthetic(family, 17, 6)
    large = synthetic(family, 29, 6)
    validation = synthetic(family, 17, 6, "validation")
    np.testing.assert_array_equal(small, large[:17])
    if np.any(np.all(small[:, None, :] == validation[None, :, :], axis=-1)):
        pytest.fail("Synthetic train/validation collision")
    np.testing.assert_equal(np.isfinite(large).all(), True)
