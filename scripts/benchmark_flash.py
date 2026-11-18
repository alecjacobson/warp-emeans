"""Compare Flash-KMeans with Warp on the existing workload and quality protocols."""

import argparse
import gc
import hashlib
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def configurations(phase):
    """Reuse archived workload identities without selecting on new test outcomes."""
    if phase == "scaling":
        rows = [json.loads(line) for line in Path("results/kmeans_scaling.jsonl").read_text().splitlines()]
        keys = ("dataset", "n", "d", "k", "epochs", "partition")
    else:
        source = "results/kmeans_matrix.json" if phase == "matrix" else "results/kmeans_test.json"
        rows = json.loads(Path(source).read_text())["records"]
        keys = ("dataset", "n", "d", "k")
    return [dict(zip(keys, values)) for values in dict.fromkeys(tuple(row[k] for k in keys) for row in rows)]


def arms_for(phase, precision, budget, epochs):
    """Keep both Lloyd implementations at identical iteration counts."""
    budgets = [25, 100] if phase == "scaling" else [budget]
    if precision == "ieee":
        return [("flash_ieee", b) for b in budgets]
    arms = [(backend, b) for b in budgets for backend in ("warp_lloyd", "flash_tf32", "flash_fp16")]
    arms.append(("warp_emeans", epochs))
    if phase != "scaling":
        arms.append(("triton_emeans", epochs))
    return arms


def worker(args):
    """Run one configuration in an isolated process with immutable Triton precision."""
    import numpy as np
    import torch
    import warp as wp
    from benchmark_kmeans import measure
    from fetch_reference import REVISION
    from flash_reference import compiled_assignment_evidence, provenance, train
    from scipy.spatial.distance import cdist

    from warp_emeans.datasets import load_dataset

    config = json.loads(args.worker)
    name, n, d, k = [config[key] for key in ("dataset", "n", "d", "k")]
    wp.init()
    wp.config.quiet = True
    torch.set_num_threads(4)
    reference = provenance()
    if reference["dot_precision"] != args.precision:
        raise ValueError("Triton precision must be fixed before process startup")
    if name == "normal":
        rng = np.random.default_rng(481 + d)
        points = rng.normal(size=(n, d)).astype(np.float32)
        evaluation = rng.normal(size=(8192, d))
    else:
        points = load_dataset(name, n, d)
        count = 24576 if args.phase == "scaling" else (16384 if args.phase == "confirmation" else 8192)
        evaluation = load_dataset(name, count, d, "test").astype(np.float64)
        if args.phase == "scaling":
            evaluation = evaluation[16384:]
    points_wp = wp.array(points, device="cuda")
    points_torch = torch.tensor(points, device="cuda")
    points_half = points_torch.half()
    if not torch.isfinite(points_half).all().item():
        raise ValueError("Input overflows FP16; no silent clipping is permitted")
    if args.phase == "confirmation":
        selected = json.loads(Path("results/kmeans_selection.json").read_text())["selected_iterations"]
        budget = selected[name] or 200
        seeds = range(300, 310)
    else:
        budget = 25
        seeds = range(20, 23) if args.phase == "scaling" else range(3)
    arms = arms_for(args.phase, args.precision, budget, config.get("epochs", 3))
    result = {
        "config": config,
        "reference": reference,
        "upstream_revision": REVISION,
        "warp": wp.__version__,
        "gpu": wp.get_device("cuda").name,
        "data_sha256": hashlib.sha256(points.tobytes()).hexdigest(),
        "evaluation_sha256": hashlib.sha256(evaluation.tobytes()).hexdigest(),
        "records": [],
    }
    for seed in seeds:
        initial = points[np.random.default_rng(seed).choice(n, k, replace=False)]
        initial_torch = torch.tensor(initial, device="cuda")
        initial_half = initial_torch.half()
        for backend, iterations in arms if seed % 2 == 0 else reversed(arms):
            if backend.startswith("flash_"):
                data, start = (
                    (points_half, initial_half) if backend == "flash_fp16" else (points_torch, initial_torch)
                )
                samples = []
                centers = labels = None
                for repeat in range(args.repeats + 1):
                    del centers, labels
                    gc.collect()
                    torch.cuda.synchronize()
                    before = time.perf_counter()
                    centers, labels = train(data, start, iterations)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - before
                    if repeat:
                        samples.append(elapsed)
                centers = centers.float().cpu().numpy()
                timing = {"seconds": statistics.median(samples), "samples": samples}
            else:
                centers, timing = measure(
                    backend, iterations, points_wp, points_torch, initial, seed, args.repeats
                )
            if not np.isfinite(centers).all():
                raise ValueError(f"Nonfinite centers for {config} {backend} {seed}")
            result["records"].append(
                {
                    **config,
                    "seed": seed,
                    "backend": backend,
                    "budget": iterations,
                    **timing,
                    "initial_sha256": hashlib.sha256(initial.tobytes()).hexdigest(),
                    "mse": float(cdist(evaluation, centers, "sqeuclidean").min(axis=1).mean()),
                }
            )
    result["compiled_assignment"] = compiled_assignment_evidence()
    # FP16 dots are marked IEEE regardless of the FP32 default. Require at least
    # one compiled FP32 dot with the requested precision in each worker.
    operations = [op for kernel in result["compiled_assignment"] for op in kernel["dot_operations"]]
    fp32_operations = [op for op in operations if "xf32> * tensor" in op]
    # MLIR elides the default IEEE attribute; TF32 is explicit.
    verified = fp32_operations and all(
        ("inputPrecision = tf32" in op)
        if args.precision == "tf32"
        else ("inputPrecision" not in op or "inputPrecision = ieee" in op)
        for op in fp32_operations
    )
    if not verified:
        raise RuntimeError(f"Compiled FP32 dot does not verify requested precision {args.precision}")
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")


def main():
    """Archive paired measurements, preserving earlier evidence and process isolation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["matrix", "confirmation", "scaling"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--precision", choices=["tf32", "ieee"], default="tf32", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    output = Path(args.output)
    if output.exists():
        raise ValueError("Output exists; use a fresh path to preserve prior evidence")
    if args.worker:
        worker(args)
        return
    problems = configurations(args.phase)
    if args.datasets:
        missing = set(args.datasets) - {p["dataset"] for p in problems}
        if missing:
            parser.error(f"Unknown datasets: {sorted(missing)}")
        problems = [p for p in problems if p["dataset"] in args.datasets]
    sources = [
        Path(__file__),
        Path("scripts/flash_reference.py"),
        Path("scripts/benchmark_kmeans.py"),
        Path("scripts/fetch_reference.py"),
        *sorted(Path("src/warp_emeans").glob("*.py")),
    ]
    inputs = [
        Path("results/kmeans_selection.json"),
        Path("results/kmeans_matrix.json"),
        Path("results/kmeans_test.json"),
        Path("results/kmeans_scaling.jsonl"),
        Path("uv.lock"),
    ]
    result = {
        "phase": args.phase,
        "repeats": args.repeats,
        "datasets": args.datasets,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "code_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        "timing": "Ordinary public train calls on resident inputs, synchronized wall time; one warmup then median of repeats. Includes allocations, initial-center clone, Flash norm computation, Python loop and convergence synchronization even with tol=-1. Excludes transfers, dtype conversion, compilation, garbage collection and FP64 loss. Fresh process per configuration and FP32-dot mode; alternating arm order by seed. No CUDA graphs.",
        "quality": "FP64 held-out squared Euclidean loss on original unrounded data. Shared FP32 initial centers; FP16 casts both data and initial centers. Confirmation uses frozen prior validation budgets and new seeds 300..309 on the same known test vectors. Scaling reuses seeds 20..22 and rows 16384:24576. No new scaling-law selection or independent domain claim.",
        "workers": [],
        "records": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="flash-", dir=".cache") as scratch:
        for config in problems:
            for precision in ("tf32", "ieee"):
                worker_output = Path(scratch) / "worker.json"
                environment = dict(os.environ, TRITON_F32_DEFAULT=precision, TRITON_DEFAULT_FP_FUSION="1")
                subprocess.run(
                    [
                        sys.executable,
                        __file__,
                        "--phase",
                        args.phase,
                        "--output",
                        str(worker_output),
                        "--worker",
                        json.dumps(config),
                        "--precision",
                        precision,
                        "--repeats",
                        str(args.repeats),
                    ],
                    env=environment,
                    check=True,
                    timeout=1800,
                )
                completed = json.loads(worker_output.read_text())
                result["records"].extend(completed.pop("records"))
                result["workers"].append(completed)
                worker_output.unlink()
                output.write_text(json.dumps(result, indent=2) + "\n")
                print(config, precision, "records", len(result["records"]), flush=True)


if __name__ == "__main__":
    main()
