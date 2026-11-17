# warp-emeans

[e-means](https://e-means.github.io/) clustering implemented in NVIDIA Warp. Clustering, shuffling, updates and revival run in Warp; PyTorch is only needed for reference benchmarks.

## Usage

```sh
uv sync
```

```python
import numpy as np
import warp as wp
from warp_emeans import Emeans

x = wp.array(np.random.default_rng(0).normal(size=(100_000, 3)).astype(np.float32), device="cuda")
model = Emeans(k=256, nepochs=3)
centers, moving_labels = model.train(x)
distances, nearest_labels = model.assign(x)

# Reuse a fixed workload, with preparation/capture cost paid separately:
model.prepare(x, capture=True)
centers, moving_labels = model.run()
```

Inputs are finite, contiguous FP32 Warp matrices. Outputs stay on the input device. `moving_labels` come from the last training pass; `assign` computes nearest-final-center labels. Autodiff is unsupported. See [API and numerical scope](docs/api.md).

```sh
uv sync --extra research --extra dev
uv run python scripts/run_reference.py
uv run python scripts/benchmark.py
uv run pytest -q
```

## Current flagship result

On an RTX PRO 6000 Blackwell MIG slice, **ordinary Warp training is 1.4–2.4× faster than pinned upstream Triton** in the measured d=2–10 cases (n=10,000–1,000,000; k=32–1,024; three passes, three seeds, FP32). Validation MSE is close but varies with the different random streams; this is a matched-recipe speed comparison, not an equal-loss guarantee. Reusable CUDA graphs have separately reported setup costs. Dimensions 32 and 100 remain slower. [Measurements and raw results](docs/experiments/warp-baseline.md).
