"""Compare ordinary training and graph replay against pinned upstream Triton."""

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import warp as wp
from fetch_reference import REVISION, fetch
from scipy.spatial.distance import cdist

from warp_emeans import Emeans

sys.path.insert(0, str(fetch()))
from e_means import Emeans as ReferenceEmeans

parser = argparse.ArgumentParser()
parser.add_argument("--dimensions", type=int, nargs="+", default=[2, 3, 4, 5, 6, 8, 10, 32, 100])
parser.add_argument("--n", type=int, default=100000)
parser.add_argument("--k", type=int, default=256)
parser.add_argument("--repeats", type=int, default=3)
parser.add_argument("--output", default="results/benchmark.json")
args = parser.parse_args()
wp.init()
torch.set_num_threads(4)
records = []
for d in args.dimensions:
    rng = np.random.default_rng(481 + d)
    x = rng.normal(size=(args.n, d)).astype(np.float32)
    validation = rng.normal(size=(8192, d)).astype(np.float32)
    x_wp, x_torch = wp.array(x, device="cuda"), torch.tensor(x, device="cuda")
    for seed in range(args.repeats):
        initial = x[np.random.default_rng(seed).choice(len(x), args.k, replace=False)]
        init_wp, init_torch = wp.array(initial, device="cuda"), torch.tensor(initial, device="cuda")
        for mode in ("triton", "warp", "warp_graph"):
            for repeat in range(2):
                torch.cuda.synchronize()
                start = time.perf_counter()
                setup = 0.0
                if mode == "triton":
                    model = ReferenceEmeans(d, args.k, nepochs=3, seed=seed, assignment_backend="triton")
                    centers, _ = model.train(x_torch, init_centroids=init_torch)
                else:
                    model = Emeans(args.k, nepochs=3, seed=seed)
                    if mode == "warp_graph":
                        model.prepare(x_wp, init_wp, capture=True)
                        wp.synchronize()
                        setup = time.perf_counter() - start
                        start = time.perf_counter()
                        centers, _ = model.run()
                    else:
                        centers, _ = model.train(x_wp, init_wp)
                torch.cuda.synchronize()
                seconds = time.perf_counter() - start
                centers = centers.cpu().numpy() if mode == "triton" else centers.numpy()
                mse = cdist(validation, centers, "sqeuclidean").min(axis=1).mean()
                record = {
                    "mode": mode,
                    "n": len(x),
                    "d": d,
                    "k": args.k,
                    "epochs": 3,
                    "seed": seed,
                    "repeat": repeat,
                    "seconds": seconds,
                    "setup_seconds": setup,
                    "validation_mse_fp64": mse,
                    "recipe": model.recipe if mode == "triton" else asdict(model.recipe),
                }
                records.append(record)
                print(json.dumps({k: v for k, v in record.items() if k != "recipe"}), flush=True)
    Path(args.output).write_text(
        json.dumps(
            {
                "upstream_revision": REVISION,
                "warp": wp.__version__,
                "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(),
                "records": records,
            },
            indent=2,
        )
        + "\n"
    )
