"""Run resumable Appendix-B-style step/batch sweeps on declared splits."""

import argparse
import hashlib
import itertools
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import warp as wp
from scipy.spatial.distance import cdist

from warp_emeans import Emeans
from warp_emeans.datasets import load_dataset


def configurations(profile="pilot"):
    """Enumerate a factorial fit set and independent transfer configurations."""
    configs = []
    for family, d, n, k, epochs in itertools.product(
        ("uniform", "mixture"), (2, 3, 5, 10), (32768, 131072), (32, 128), (1, 3)
    ):
        configs.append(dict(dataset=family, d=d, n=n, k=k, epochs=epochs, partition="fit"))
    for family, d in itertools.product(("uniform", "mixture"), (4, 6, 7, 8, 9, 16, 32, 64, 100)):
        configs.append(dict(dataset=family, d=d, n=65536, k=64, epochs=2, partition="dimension_holdout"))
    for family, d in itertools.product(("gaussian", "manifold"), (2, 3, 5, 10)):
        configs.append(dict(dataset=family, d=d, n=65536, k=64, epochs=2, partition="family_holdout"))
    for family, d in (
        ("intel_xy", 2),
        ("bsds_rgb", 3),
        ("bsds_labxy", 5),
        ("bunny_xyz", 3),
        ("drill_xyz", 3),
        ("bunny_xyzn", 6),
        ("drill_xyzn", 6),
    ):
        for n in (32768, 65536):
            configs.append(dict(dataset=family, d=d, n=n, k=64, epochs=2, partition="real_holdout"))
    if profile == "canonical":
        configs = []
        for family, d, n, k, epochs in itertools.product(
            ("uniform", "mixture"), (2, 3, 5, 10), (100000, 1000000), (64, 256), (1, 3)
        ):
            configs.append(dict(dataset=family, d=d, n=n, k=k, epochs=epochs, partition="fit"))
        for family, d in itertools.product(("uniform", "mixture"), (4, 6, 7, 8, 9, 16, 32, 64, 100)):
            configs.append(
                dict(dataset=family, d=d, n=100000, k=128, epochs=2, partition="dimension_holdout")
            )
        for family, d in itertools.product(("gaussian", "manifold"), (2, 3, 5, 10)):
            configs.append(dict(dataset=family, d=d, n=100000, k=128, epochs=2, partition="family_holdout"))
        for family, d in (
            ("intel_xy", 2),
            ("bsds_rgb", 3),
            ("bsds_labxy", 5),
            ("bunny_xyz", 3),
            ("drill_xyz", 3),
            ("bunny_xyzn", 6),
            ("drill_xyzn", 6),
        ):
            configs.append(
                dict(
                    dataset=family,
                    d=d,
                    n=65536 if family == "intel_xy" else 100000,
                    k=128,
                    epochs=2,
                    partition="real_holdout",
                )
            )
    return configs


def main():
    """Append each completed measurement immediately, skipping existing run IDs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/scaling_sweep.jsonl")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--profile", choices=["pilot", "canonical"], default="pilot")
    args = parser.parse_args()
    wp.init()
    wp.config.quiet = True
    configs = configurations(args.profile)
    if args.limit is not None:
        configs = configs[: args.limit]
    output = Path(args.output)
    done = set()
    if output.exists():
        for line in output.read_text().splitlines():
            done.add(json.loads(line)["run_id"])
    alphas = (
        [1e-5, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
        if args.profile == "canonical"
        else [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
    )
    metadata = {
        "version": 2,
        "profile": args.profile,
        "maximum_steps_per_run": 32768,
        "warp": wp.__version__,
        "gpu": wp.get_device("cuda").name,
        "validation_rows": 4096,
        "alphas": alphas,
        "seeds": [0, 1, 2],
        "configurations": configs,
        "note": "Pilot: sizes below paper noise threshold; full factorial fit and held-out transfers; FP64 MSE.",
        "code_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), *Path("src/warp_emeans").glob("*.py")]
        },
    }
    output.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
    start = time.perf_counter()
    with output.open("a", buffering=1) as file:
        for index, config in enumerate(configs):
            name, d, n, k, epochs = [config[key] for key in ("dataset", "d", "n", "k", "epochs")]
            x = load_dataset(name, n, d)
            validation = load_dataset(name, 4096, d, "validation").astype(np.float64)
            x_wp = wp.array(x, device="cuda")
            batches = sorted(
                set(
                    [32, 128, 512, 2048, 8192, 32768, n]
                    if args.profile == "canonical"
                    else [128, 512, 2048, 8192, n]
                )
            )
            batches = [b for b in batches if b <= n and epochs * ((n + b - 1) // b) <= 32768]
            for batch, seed in itertools.product(batches, range(3)):
                pending = []
                for alpha in alphas:
                    identity = {**config, "batch": batch, "alpha": alpha, "seed": seed}
                    run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
                    if run_id not in done:
                        pending.append((alpha, identity, run_id))
                if not pending:
                    continue
                initial = x[np.random.default_rng(seed).choice(n, k, replace=False)]
                model = Emeans(k, nepochs=epochs, batch_size=batch, alpha0=alphas[0], seed=seed)
                prep = time.perf_counter()
                model.prepare(x_wp, wp.array(initial, device="cuda"), capture=True)
                wp.synchronize()
                setup = time.perf_counter() - prep
                for alpha, identity, run_id in pending:
                    before = time.perf_counter()
                    centers, _ = model.run(alpha0=alpha)
                    wp.synchronize()
                    seconds = time.perf_counter() - before
                    centers_np = centers.numpy()
                    if not np.isfinite(centers_np).all():
                        raise ValueError(f"Nonfinite centers in {identity}")
                    mse = cdist(validation, centers_np.astype(np.float64), "sqeuclidean").min(axis=1).mean()
                    row = {
                        **identity,
                        "run_id": run_id,
                        "validation_mse": float(mse),
                        "seconds": seconds,
                        "graph_setup_seconds": setup,
                        "recipe": asdict(model.recipe),
                    }
                    file.write(json.dumps(row, separators=(",", ":")) + "\n")
                    done.add(run_id)
            print(
                f"{index + 1}/{len(configs)} {name} d={d} n={n} k={k} T={epochs} records={len(done)} elapsed={time.perf_counter() - start:.1f}s",
                flush=True,
            )


if __name__ == "__main__":
    main()
