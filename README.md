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

**Pure Warp Lloyd trains RGB clusters in 3.43 ms**, versus **18.87 ms for Flash-KMeans with IEEE FP32 products** and **20.78 ms for its paper-style FP16 mode**: 82% and 83% less time. All three run 100 Lloyd iterations on 100,000 RGB vectors with k=64 and shared starting centers, using ordinary training on an RTX PRO 6000 Blackwell MIG slice.

Across ten new initialization seeds, Warp meets the specified 0.2% loss margin against both modes. Its mean test loss is 0.03% lower than Flash IEEE (paired 95% interval: 0.16% lower to 0.15% higher). **Flash's default TF32 mode takes 19.27 ms and has lower mean loss; Warp's quality margin against it is not established.** Flash also wins on some larger and higher-dimensional workloads. The new seeds do not establish the earlier RGB quality advantage over e-means.

[Full Flash comparison, precision controls, all seven real feature datasets and scaling configurations through d=100](docs/experiments/flash-kmeans.md). [Earlier Warp/e-means study and graph timings](docs/experiments/kmeans-baseline.md).
