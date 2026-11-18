"""Compare pure Warp Lloyd and e-means with shared data and initialization."""

import argparse
import gc
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import warp as wp
from fetch_reference import REVISION, fetch
from scipy.spatial.distance import cdist

from warp_emeans import Emeans, Kmeans
from warp_emeans.datasets import load_dataset

sys.path.insert(0, str(fetch()))
from e_means import Emeans as ReferenceEmeans

REAL_PROBLEMS = [
    ("intel_xy", 2, 65536, 64),
    ("bsds_rgb", 3, 100000, 64),
    ("bsds_labxy", 5, 100000, 64),
    ("bunny_xyz", 3, 100000, 256),
    ("bunny_xyzn", 6, 100000, 256),
    ("drill_xyz", 3, 100000, 128),
    ("drill_xyzn", 6, 100000, 128),
]
BUDGETS = [1, 3, 5, 10, 25, 50, 100, 200]


def measure(backend, budget, points_wp, points_torch, initial, seed, repeats, graph=False):
    """Measure resident-input training, excluding transfers and model cleanup."""
    initial_wp = wp.array(initial, device="cuda")
    initial_torch = torch.tensor(initial, device="cuda")
    samples = []
    setup = None
    model = None
    for repeat in range(repeats + 1):
        # Destroy the previous model and graph before starting another timer.
        del model
        gc.collect()
        torch.cuda.synchronize()
        start = time.perf_counter()
        if backend == "triton_emeans":
            model = ReferenceEmeans(
                initial.shape[1], len(initial), nepochs=budget, seed=seed, assignment_backend="triton"
            )
            centers, _ = model.train(points_torch, init_centroids=initial_torch)
        else:
            model = (
                Kmeans(len(initial), niter=budget, seed=seed)
                if backend == "warp_lloyd"
                else Emeans(len(initial), nepochs=budget, seed=seed)
            )
            if graph:
                model.prepare(points_wp, initial_wp, capture=True)
                wp.synchronize_device("cuda")
                setup = time.perf_counter() - start
                # One warm replay, then repeated executions of this same graph.
                model.run()
                wp.synchronize_device("cuda")
                graph_samples = []
                for _ in range(repeats):
                    start = time.perf_counter()
                    centers, _ = model.run()
                    wp.synchronize_device("cuda")
                    graph_samples.append(time.perf_counter() - start)
                samples = graph_samples
                break
            centers, _ = model.train(points_wp, initial_wp)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        if repeat:
            samples.append(seconds)
    strategy = model._assignment if backend == "warp_lloyd" else None
    centers = centers.cpu().numpy() if backend == "triton_emeans" else centers.numpy()
    # Host copies, quality evaluation and all cleanup are outside the timers.
    del model
    gc.collect()
    return centers, {
        "seconds": statistics.median(samples),
        "samples": samples,
        "setup_seconds": setup,
        "assignment": strategy,
    }


def main():
    """Keep the fixed-budget matrix, validation selection and confirmation separate."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["matrix", "select", "test"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--selection", default="results/kmeans_selection.json")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError("Output already exists; choose a fresh path to preserve evidence")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    wp.init()
    wp.config.quiet = True
    torch.set_num_threads(4)
    selected = {}
    if args.phase == "test":
        selected = json.loads(Path(args.selection).read_text())["selected_iterations"]
    problems = REAL_PROBLEMS
    if args.phase == "matrix":
        problems = [
            ("normal", d, n, k)
            for n, k, dimensions in [
                (10000, 32, [2, 5, 10]),
                (100000, 256, [2, 3, 4, 5, 6, 8, 10, 32, 100]),
                (1000000, 1024, [2, 5, 10]),
            ]
            for d in dimensions
        ] + REAL_PROBLEMS
    result = {
        "phase": args.phase,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "code_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [Path(__file__), *sorted(Path("src/warp_emeans").glob("*.py"))]
        },
        "upstream_revision": REVISION,
        "warp": wp.__version__,
        "torch": torch.__version__,
        "gpu": wp.get_device("cuda").name,
        "margin": 0.002,
        "budgets": BUDGETS,
        "selected_iterations": selected,
        "selection_sha256": hashlib.sha256(Path(args.selection).read_bytes()).hexdigest()
        if args.phase == "test"
        else None,
        "timing": "Resident FP32 inputs and shared initial centers. Ordinary calls include validation/allocation; exclude transfers, warmup, cleanup and FP64 loss. Graph replay includes reset and all iterations; setup/capture is separate. One untimed warmup; median of recorded repeats. Select-phase timing is not performance evidence.",
        "quality": "Select: seeds 0..2 and first 16384 validation rows. Test: seeds 200..209 and first 16384 test rows; paired bootstrap describes initialization variation only. Matrix: seeds 0..2 and 8192 test rows (normal uses independent RNG draws).",
        "records": [],
    }
    for name, d, n, k in problems:
        if name == "normal":
            rng = np.random.default_rng(481 + d)
            x = rng.normal(size=(n, d)).astype(np.float32)
            evaluation = rng.normal(size=(8192, d))
        else:
            x = load_dataset(name, n, d)
            evaluation = load_dataset(
                name,
                8192 if args.phase == "matrix" else 16384,
                d,
                "validation" if args.phase == "select" else "test",
            ).astype(np.float64)
        points_wp = wp.array(x, device="cuda")
        points_torch = torch.tensor(x, device="cuda")
        if args.phase == "select":
            arms = [("triton_emeans", 3, False)] + [("warp_lloyd", b, False) for b in BUDGETS]
        elif args.phase == "test":
            arms = [("triton_emeans", 3, False), ("warp_emeans", 3, False)]
            # Report failures too, at the maximum attempted budget, without a match claim.
            arms.append(("warp_lloyd", selected[name] or max(BUDGETS), False))
        else:
            arms = [
                ("triton_emeans", 3, False),
                ("warp_emeans", 3, False),
                ("warp_lloyd", 25, False),
                ("warp_emeans", 3, True),
                ("warp_lloyd", 25, True),
            ]
        seeds = range(200, 210) if args.phase == "test" else range(3)
        for seed in seeds:
            initial = x[np.random.default_rng(seed).choice(n, k, replace=False)]
            for backend, budget, graph in arms if seed % 2 == 0 else reversed(arms):
                centers, timing = measure(
                    backend,
                    budget,
                    points_wp,
                    points_torch,
                    initial,
                    seed,
                    1 if args.phase == "select" else args.repeats,
                    graph,
                )
                if not np.isfinite(centers).all():
                    raise ValueError(f"Nonfinite centers: {name} {backend} {seed}")
                loss = float(cdist(evaluation, centers, "sqeuclidean").min(axis=1).mean())
                result["records"].append(
                    {
                        "dataset": name,
                        "n": n,
                        "d": d,
                        "k": k,
                        "seed": seed,
                        "backend": backend,
                        "budget": budget,
                        "graph": graph,
                        "mse": loss,
                        **timing,
                    }
                )
        if args.phase == "select":
            rows = [r for r in result["records"] if r["dataset"] == name]
            target = np.mean([r["mse"] for r in rows if r["backend"] == "triton_emeans"])
            selected[name] = next(
                (
                    b
                    for b in BUDGETS
                    if np.mean([r["mse"] for r in rows if r["backend"] == "warp_lloyd" and r["budget"] == b])
                    <= 1.002 * target
                ),
                None,
            )
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(name, n, d, k, "selected", selected.get(name), "records", len(result["records"]), flush=True)


if __name__ == "__main__":
    main()
