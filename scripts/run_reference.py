"""Run the upstream NumPy and GPU Triton implementations reproducibly."""

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
from fetch_reference import REVISION, fetch
from scipy.spatial.distance import cdist

sys.path.insert(0, str(fetch()))
import faiss
import torch
from e_means import Emeans, NumpyEmeans

faiss.omp_set_num_threads(4)
torch.set_num_threads(4)
rng = np.random.default_rng(481)
records = []
for d in (2, 3, 5, 10, 100):
    x = rng.normal(size=(100_000, d)).astype(np.float32)
    validation = rng.normal(size=(4096, d)).astype(np.float32)
    initial = x[np.random.default_rng(19).choice(len(x), 256, replace=False)]
    for backend in ("numpy", "triton"):
        for repeat in range(2):
            if backend == "numpy":
                model = NumpyEmeans(d, 256, nepochs=3, seed=19)
                data, init = x, initial
            else:
                model = Emeans(d, 256, nepochs=3, seed=19, assignment_backend="triton")
                data, init = torch.tensor(x, device="cuda"), torch.tensor(initial, device="cuda")
                torch.cuda.synchronize()
            start = time.perf_counter()
            centroids, labels = model.train(data, init_centroids=init)
            if backend == "triton":
                torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            if backend == "triton":
                centroids, labels = centroids.cpu().numpy(), labels.cpu().numpy()
            if not np.isfinite(centroids).all() or not ((labels >= 0) & (labels < 256)).all():
                raise ValueError("Invalid reference output")
            mse = cdist(validation, centroids, metric="sqeuclidean").min(axis=1).mean()
            record = {
                "d": d,
                "n": len(x),
                "k": 256,
                "epochs": 3,
                "seed": 19,
                "backend": backend,
                "repeat": repeat,
                "seconds": seconds,
                "validation_mse_fp64": mse,
                "recipe": model.recipe,
                "n_splits": model.n_splits,
            }
            records.append(record)
            print(
                json.dumps(
                    {key: record[key] for key in ("d", "backend", "repeat", "seconds", "validation_mse_fp64")}
                ),
                flush=True,
            )
output = {
    "reference_revision": REVISION,
    "python": platform.python_version(),
    "numpy": np.__version__,
    "torch": torch.__version__,
    "faiss": faiss.__version__,
    "gpu": torch.cuda.get_device_name(),
    "records": records,
}
Path("results/reference.json").write_text(json.dumps(output, indent=2) + "\n")
