"""Generate controlled distributions with independent parameter and sample seeds."""

from functools import cache
from pathlib import Path

import numpy as np


@cache
def _parameters(family, d):
    rng = np.random.default_rng(1901 + d)
    if family == "mixture":
        centers = rng.normal(size=(12, d)) * 1.7
        scales = np.exp(rng.uniform(-2.5, -0.1, size=(12, d)))
        weights = np.exp(rng.uniform(-2, 0, 12))
        return centers, scales, weights / weights.sum()
    if family == "manifold":
        return np.linalg.qr(rng.normal(size=(max(d, 4), max(d, 4))))[0][:4, :d]
    return None


def synthetic(family, n, d, split="train"):
    """Sample a distribution with fixed geometry and disjoint split RNG streams.

    Args:
        family: uniform, gaussian, mixture, or manifold.
        n: Number of independent samples.
        d: Ambient Euclidean dimension.
        split: train, validation, or test.

    Returns:
        A contiguous FP32 matrix. Sizes are nested prefixes within each split;
        family parameters are shared across splits but samples are independent.
    """
    split_id = {"train": 0, "validation": 1, "test": 2}[split]
    family_id = {"uniform": 0, "gaussian": 1, "mixture": 2, "manifold": 3}[family]
    rng = np.random.default_rng(91823 + 1009 * d + 31 * family_id + split_id)
    if family == "uniform":
        values = rng.uniform(-1, 1, size=(n, d))
    elif family == "gaussian":
        values = rng.normal(size=(n, d))
    elif family == "mixture":
        centers, scales, weights = _parameters(family, d)
        # One random row per sample keeps all n settings nested prefixes.
        raw = rng.random(size=(n, d + 1))
        ids = np.searchsorted(np.cumsum(weights), raw[:, 0])
        from scipy.special import ndtri

        normal = ndtri(np.clip(raw[:, 1:], 1e-12, 1 - 1e-12))
        values = centers[ids] + normal * scales[ids]
    else:
        raw = rng.normal(size=(n, d + 2))
        angles = raw[:, :2]
        features = np.stack(
            (np.sin(angles[:, 0]), np.cos(angles[:, 0]), np.sin(angles[:, 1]), np.cos(angles[:, 1])), axis=1
        )
        values = features @ _parameters(family, d) + 0.01 * raw[:, 2:]
    return np.ascontiguousarray(values, dtype=np.float32)


def load_dataset(name, n, d=None, split="train", root="data/processed"):
    """Load a deterministic prefix or generate synthetic vectors without upsampling."""
    if name in ("uniform", "gaussian", "mixture", "manifold"):
        if d is None:
            raise ValueError("Synthetic data require d")
        return synthetic(name, n, d, split)
    values = np.load(Path(root) / f"{name}_{split}.npy", mmap_mode="r")
    if len(values) < n:
        raise ValueError(f"{name} {split} has {len(values)} rows; requested {n}")
    if d is not None and values.shape[1] != d:
        raise ValueError(f"{name} dimension is {values.shape[1]}, not {d}")
    return np.ascontiguousarray(values[:n])
