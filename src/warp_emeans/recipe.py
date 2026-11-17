"""Resolve the pinned upstream Euclidean training recipe."""

import math
from dataclasses import dataclass
from enum import Enum


class BatchLaw(str, Enum):
    """Select a published upstream batch policy."""

    AUTO = "auto"
    QUALITY = "quality"
    THROUGHPUT = "throughput"


@dataclass(frozen=True)
class Recipe:
    """Describe the resolved work and optimization parameters."""

    batch_size: int
    alpha0: float
    epoch_rows: tuple[int, ...]
    total_steps: int
    unused_threshold: float
    batch_law: str


def _positive_integer(name, value):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")


def resolve_recipe(
    n, d, k, nepochs=3.0, batch_size=None, alpha0=None, batch_law=BatchLaw.AUTO, unused_threshold=None
):
    """Resolve batches, cosine peak and revival policy for finite Euclidean input."""
    for name, value in (("n", n), ("d", d), ("k", k)):
        _positive_integer(name, value)
    if k > n:
        raise ValueError("k must not exceed n")
    if n >= 2**31 or k >= 2**30:
        raise ValueError("n and k must fit the signed 32-bit indexing workspace")
    if not math.isfinite(nepochs) or nepochs <= 0:
        raise ValueError("nepochs must be finite and positive")
    policy = BatchLaw(batch_law)
    if policy == BatchLaw.AUTO:
        policy = BatchLaw.QUALITY if d <= 16 else BatchLaw.THROUGHPUT
    if batch_size is None:
        coefficient = 0.00439175 if policy == BatchLaw.QUALITY else 0.035134
        raw = coefficient * k**0.134749 * n**0.876473 * nepochs**0.813344
        batch_size = 2 ** max(0, math.floor(math.log2(raw) + 0.5))
        cap = n // 2 if policy == BatchLaw.QUALITY else 2 ** max(0, math.floor(math.log2(n / 2) + 0.5))
        batch_size = max(1, min(n, batch_size, cap))
    else:
        _positive_integer("batch_size", batch_size)
        if batch_size > n:
            raise ValueError("batch_size must not exceed n")
    if alpha0 is None:
        alpha0 = min(1.0, max(1e-4, 0.8244 * k**-0.4617 * batch_size**0.9003 * n**-0.4498 * nepochs**-0.1057))
    if not math.isfinite(alpha0) or not 0 < alpha0 <= 1:
        raise ValueError("alpha0 must be finite and in (0, 1]")
    if unused_threshold is None:
        unused_threshold = 0.0 if d <= 2 else 0.1
    if not math.isfinite(unused_threshold) or unused_threshold < 0:
        raise ValueError("unused_threshold must be finite and nonnegative")
    whole = math.floor(nepochs)
    remainder = round((nepochs - whole) * n)
    rows = ((n,) * (whole - 1) + ((remainder,) if remainder else ()) + (n,)) if whole else (remainder,)
    if min(rows) < 1:
        raise ValueError("nepochs rounds to zero training rows")
    return Recipe(
        batch_size,
        float(alpha0),
        rows,
        sum(math.ceil(r / batch_size) for r in rows),
        float(unused_threshold),
        policy.value,
    )
