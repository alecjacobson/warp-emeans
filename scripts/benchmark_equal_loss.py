"""Select Warp budgets on validation, then benchmark fresh seeds on test data."""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import warp as wp
from fetch_reference import REVISION, fetch
from scipy.spatial.distance import cdist

from warp_emeans import Emeans
from warp_emeans.datasets import load_dataset

sys.path.insert(0, str(fetch()))
from e_means import Emeans as ReferenceEmeans

PROBLEMS = [
    ("intel_xy", 2, 65536, 64),
    ("bsds_rgb", 3, 100000, 64),
    ("bsds_labxy", 5, 100000, 64),
    ("bunny_xyz", 3, 100000, 256),
    ("bunny_xyzn", 6, 100000, 256),
]


def run(backend, x_wp, x_torch, initial, seed, k, epochs):
    """Time ordinary resident-input training after inputs are synchronized."""
    init_wp = wp.array(initial, device="cuda")
    init_torch = torch.tensor(initial, device="cuda")
    gc.collect()
    torch.cuda.synchronize()
    start = time.perf_counter()
    if backend == "warp":
        model = Emeans(k, nepochs=epochs, seed=seed)
        centers, _ = model.train(x_wp, init_wp)
    else:
        model = ReferenceEmeans(x_torch.shape[1], k, nepochs=epochs, seed=seed, assignment_backend="triton")
        centers, _ = model.train(x_torch, init_centroids=init_torch)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    centers = centers.numpy() if backend == "warp" else centers.cpu().numpy()
    del model
    gc.collect()
    return centers, seconds


def main():
    """Keep validation budget selection separate from confirmatory test timing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["select", "test"], required=True)
    args = parser.parse_args()
    wp.init()
    wp.config.quiet = True
    torch.set_num_threads(4)
    records = []
    selected = {}
    if args.phase == "test":
        selected = json.loads(Path("results/equal_loss_selection.json").read_text())["selected_epochs"]
    for name, d, n, k in PROBLEMS:
        x = load_dataset(name, n, d)
        evaluation = load_dataset(name, 16384, d, "validation" if args.phase == "select" else "test").astype(
            np.float64
        )
        x_wp, x_torch = wp.array(x, device="cuda"), torch.tensor(x, device="cuda")
        if args.phase == "select":
            arms = [("triton", 3)] + [("warp", epochs) for epochs in (1, 2, 3, 5, 10, 25)]
            seeds = range(3)
        else:
            if selected.get(name) is None:
                continue
            arms = [("triton", 3), ("warp", selected[name])]
            seeds = range(100, 110)
            initial = x[np.random.default_rng(999).choice(n, k, replace=False)]
            for backend, epochs in arms:
                run(backend, x_wp, x_torch, initial, 999, k, epochs)
        # Alternate backend order across fresh seeds to reduce order effects.
        for seed in seeds:
            initial = x[np.random.default_rng(seed).choice(n, k, replace=False)]
            for backend, epochs in arms if seed % 2 == 0 else list(reversed(arms)):
                centers, seconds = run(backend, x_wp, x_torch, initial, seed, k, epochs)
                mse = cdist(evaluation, centers, "sqeuclidean").min(axis=1).mean()
                records.append(
                    {
                        "dataset": name,
                        "d": d,
                        "n": n,
                        "k": k,
                        "seed": seed,
                        "backend": backend,
                        "epochs": epochs,
                        "seconds": seconds,
                        "mse": float(mse),
                    }
                )
        if args.phase == "select":
            target = np.mean([r["mse"] for r in records if r["dataset"] == name and r["backend"] == "triton"])
            selected[name] = None
            for epochs in (1, 2, 3, 5, 10, 25):
                measured = np.mean(
                    [
                        r["mse"]
                        for r in records
                        if r["dataset"] == name and r["backend"] == "warp" and r["epochs"] == epochs
                    ]
                )
                if measured <= target * 1.002:
                    selected[name] = epochs
                    break
        print(name, "selected epochs", selected[name], flush=True)
        output = {
            "phase": args.phase,
            "upstream_revision": REVISION,
            "warp": wp.__version__,
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(),
            "margin": 0.002,
            "selected_epochs": selected,
            "records": records,
            "timing": "Select-phase timing is not performance evidence. Test-phase training includes allocation/validation; excludes transfers, JIT warmup and evaluation.",
        }
        destination = (
            "results/equal_loss_selection.json" if args.phase == "select" else "results/equal_loss_test.json"
        )
        Path(destination).write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
