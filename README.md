# warp-emeans

[e-means](https://e-means.github.io/) and full-data Lloyd k-means implemented in NVIDIA Warp. Clustering, initialization and updates run in Warp; PyTorch is only needed for reference benchmarks.

## Usage

```sh
uv sync
```

```python
import numpy as np
import warp as wp
from warp_emeans import Emeans, Kmeans

x = wp.array(np.random.default_rng(0).normal(size=(100_000, 3)).astype(np.float32), device="cuda")
model = Emeans(k=256, nepochs=3)
centers, moving_labels = model.train(x)
distances, nearest_labels = model.assign(x)

# Full-data Lloyd baseline: 25 assignment-and-mean iterations.
baseline = Kmeans(k=256, niter=25)
lloyd_centers, last_update_labels = baseline.train(x)

# Reuse a fixed workload, with preparation/capture cost paid separately:
baseline.prepare(x, capture=True)
lloyd_centers, last_update_labels = baseline.run()
```

Inputs are finite, contiguous FP32 Warp matrices. Outputs stay on the input device. `moving_labels` come from the last training pass; `assign` computes nearest-final-center labels. Autodiff is unsupported. See [API and numerical scope](docs/api.md).

```sh
uv sync --extra research --extra dev
uv run python scripts/run_reference.py
uv run python scripts/benchmark.py
uv run pytest -q
```

## Current flagship result

**Pure Warp Lloyd k-means trains RGB clusters in 3.46 ms**, versus **25.28 ms for Warp e-means** and **36.16 ms for upstream Triton e-means**: 86% and 90% less time, respectively. This uses 100,000 RGB vectors, k=64, 100 Lloyd iterations versus three e-means passes, and ordinary training on an RTX PRO 6000 Blackwell MIG slice.

The Lloyd budget was selected on validation data. Across ten new paired initialization seeds, its mean test loss is 1.85% lower than Triton's (paired 95% interval: 0.34–3.26% lower). **Only this one of seven real feature datasets establishes the specified 0.2% loss margin against Triton.** E-means retains advantages on other workloads, including some large and high-dimensional cases. [All cases, fixed-budget matrix, graph timings and limitations](docs/experiments/kmeans-baseline.md).

The [scaling-study evaluator](docs/experiments/kmeans-baseline.md#scaling-study-integration) includes `warp_lloyd25` and `warp_lloyd100` alongside e-means on the same held-out configurations through d=100.
