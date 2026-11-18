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

**Pure Warp trains 1.44× faster than upstream Triton on Bunny XYZ** (100,000 points, k=256, three passes), with test loss within a predeclared 0.2% margin: Warp/Triton mean MSE 0.99869, paired 95% interval [0.99474, 1.00195], ten fresh seeds. Median ordinary training is 25.05ms versus 36.06ms on an RTX PRO 6000 Blackwell MIG slice. The quality margin passes narrowly; four other workloads did not establish the same match. [Full protocol and all five results](docs/experiments/equal-loss.md).

In a separate benchmark on synthetic data, Warp and Triton used the same batch sizes, learning-rate schedules and three passes through the data. Warp took **27–57% less time than Triton** for the tested inputs with 2–10 features per point. With **32 features**, Warp took **4.11 ms versus Triton's 7.63 ms** (46% less time). With **100 features**, Warp took **13.03 ms versus Triton's 8.90 ms** (46% more time). Here, dimension d means the number of features per point; XYZ positions have d=3. These runtime comparisons do not establish equal clustering quality. [Performance matrix](docs/experiments/warp-baseline.md).

Experimental scaling laws, fitted on synthetic data, reduce mean test loss on six of seven real feature datasets in a 30-seed confirmation; defaults remain unchanged. [Scaling results, coefficients and limitations](docs/experiments/scaling.md).
