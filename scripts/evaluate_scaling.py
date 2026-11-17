"""Evaluate frozen candidate laws on fresh seeds and untouched test vectors."""

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import faiss
import numpy as np
import warp as wp
from fit_scaling import aggregate, predict
from scipy.spatial.distance import cdist

from warp_emeans import Emeans
from warp_emeans.datasets import load_dataset


def main():
    """Evaluate baselines, validation-selected oracle and candidate laws without test tuning."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", default="results/scaling_canonical.jsonl")
    parser.add_argument("--fits", nargs="+", default=["results/scaling_intervals.json"])
    parser.add_argument("--output", default="results/scaling_evaluation.jsonl")
    parser.add_argument("--only-real", action="store_true")
    parser.add_argument("--seed-start", type=int, default=20)
    parser.add_argument("--test-offset", type=int, default=16384)
    parser.add_argument("--test-rows", type=int, default=8192)
    args = parser.parse_args()
    models = {}
    for filename in args.fits:
        models.update(json.loads(Path(filename).read_text())["models"])
    problems = [p for p in aggregate(args.sweep) if p["config"]["partition"] != "fit"]
    if args.only_real:
        problems = [p for p in problems if p["config"]["partition"] == "real_holdout"]
    wp.init()
    wp.config.quiet = True
    faiss.omp_set_num_threads(4)
    output = Path(args.output)
    done = set()
    if output.exists():
        done = {json.loads(line)["run_id"] for line in output.read_text().splitlines()}
    meta = {
        "seeds": list(range(args.seed_start, args.seed_start + 3)),
        "test_rows": args.test_rows,
        "test_offset": args.test_offset,
        "validation_rows": 4096,
        "fit_files": args.fits,
        "sweep": args.sweep,
        "fit_sha256": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in args.fits},
        "timing": "Quality evaluation only. Possible concurrent sweep; these times are not performance evidence.",
        "oracle": "Best median validation-grid setting on seeds 0,1,2, rerun on fresh evaluation seeds; not a test-selected oracle.",
    }
    output.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    with output.open("a", buffering=1) as file:
        for index, problem in enumerate(problems):
            config = problem["config"]
            name, d, n, k, epochs = [config[key] for key in ("dataset", "d", "n", "k", "epochs")]
            x = load_dataset(name, n, d)
            validation = load_dataset(name, 4096, d, "validation").astype(np.float64)
            test = load_dataset(name, args.test_offset + args.test_rows, d, "test")[
                args.test_offset :
            ].astype(np.float64)
            x_wp = wp.array(x, device="cuda")
            bi, ai = np.unravel_index(problem["loss"].argmin(), problem["loss"].shape)
            settings = {
                "upstream_auto": {"batch_law": "auto"},
                "upstream_quality": {"batch_law": "quality"},
                "upstream_throughput": {"batch_law": "throughput"},
                "validation_oracle": {
                    "batch_size": int(problem["batches"][bi]),
                    "alpha0": float(problem["alphas"][ai]),
                },
                "lloyd25": {},
            }
            for label, model in models.items():
                if model["alpha"] is not None and model["batch"] is not None:
                    batch, alpha = predict(model, config)
                    settings[label] = {"batch_size": batch, "alpha0": alpha}
            for seed in range(args.seed_start, args.seed_start + 3):
                initial = x[np.random.default_rng(seed).choice(n, k, replace=False)]
                initial_wp = wp.array(initial, device="cuda")
                for label, parameters in settings.items():
                    identity = {
                        **config,
                        "model": label,
                        "seed": seed,
                        "parameters": parameters,
                        "test_offset": args.test_offset,
                        "test_rows": args.test_rows,
                    }
                    run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
                    if run_id in done:
                        continue
                    wp.synchronize()
                    start = time.perf_counter()
                    if label == "lloyd25":
                        model = faiss.Kmeans(
                            d, k, niter=25, nredo=1, seed=seed, max_points_per_centroid=(n + k - 1) // k + 1
                        )
                        model.train(x, init_centroids=initial)
                        centers = model.centroids
                        recipe = {"backend": "faiss_cpu", "iterations": 25, "training_rows": n}
                    else:
                        model = Emeans(k, nepochs=epochs, seed=seed, **parameters)
                        model.prepare(x_wp, initial_wp)
                        if model.recipe.total_steps > 32768:
                            file.write(
                                json.dumps(
                                    {
                                        **identity,
                                        "run_id": run_id,
                                        "status": "budget_exceeded",
                                        "recipe": asdict(model.recipe),
                                    }
                                )
                                + "\n"
                            )
                            done.add(run_id)
                            continue
                        centers, _ = model.run()
                        centers = centers.numpy()
                        recipe = asdict(model.recipe)
                    seconds = time.perf_counter() - start
                    if not np.isfinite(centers).all():
                        raise ValueError(f"Nonfinite centers: {identity}")
                    row = {
                        **identity,
                        "run_id": run_id,
                        "status": "ok",
                        "recipe": recipe,
                        "seconds": seconds,
                        "validation_mse": float(cdist(validation, centers, "sqeuclidean").min(axis=1).mean()),
                        "test_mse": float(cdist(test, centers, "sqeuclidean").min(axis=1).mean()),
                    }
                    file.write(json.dumps(row, separators=(",", ":")) + "\n")
                    done.add(run_id)
            print(f"{index + 1}/{len(problems)} {name} d={d} n={n} records={len(done)}", flush=True)


if __name__ == "__main__":
    main()
