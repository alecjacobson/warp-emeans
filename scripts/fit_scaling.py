"""Fit censored pilot targets and report held-out grid regret and uncertainty."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

KEYS = ("dataset", "d", "n", "k", "epochs", "partition")


def features(config, batch=None, dimension=True):
    """Build the declared log-space predictor without data-dependent choices."""
    result = [1.0, np.log(config["k"])]
    if batch is not None:
        result.append(np.log(batch))
    result += [np.log(config["n"]), np.log(config["epochs"])]
    if dimension:
        result.append(np.log(config["d"]))
    return result


def aggregate(path):
    """Aggregate seed losses into a complete rectangular problem grid."""
    groups = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        groups[tuple(row[key] for key in KEYS)].append(row)
    problems = []
    for key, rows in groups.items():
        config = dict(zip(KEYS, key, strict=True))
        if "unused_threshold" in rows[0]:
            config["unused_threshold"] = rows[0]["unused_threshold"]
        batches = sorted({r["batch"] for r in rows})
        alphas = sorted({r["alpha"] for r in rows})
        seeds = sorted({r["seed"] for r in rows})
        values = np.full((len(batches), len(alphas), len(seeds)), np.nan)
        for row in rows:
            values[batches.index(row["batch"]), alphas.index(row["alpha"]), seeds.index(row["seed"])] = row[
                "validation_mse"
            ]
        if not np.isfinite(values).all():
            raise ValueError(f"Incomplete grid: {config}")
        problems.append(
            {
                "config": config,
                "batches": np.array(batches),
                "alphas": np.array(alphas),
                "values": values,
                "loss": np.median(values, axis=2),
            }
        )
    return problems


def targets(problems, dimension, resample_seeds=None):
    """Extract interior step minima and bracketed batch knees from validation."""
    alpha_x, alpha_y, batch_x, batch_y = [], [], [], []
    diagnostics = defaultdict(int)
    for problem in problems:
        config, batches, alphas = [problem[key] for key in ("config", "batches", "alphas")]
        loss = problem["loss"]
        if resample_seeds is not None:
            ids = resample_seeds.integers(0, problem["values"].shape[2], problem["values"].shape[2])
            loss = np.median(problem["values"][:, :, ids], axis=2)
        best_alpha = loss.argmin(axis=1)
        for batch, index in zip(batches, best_alpha, strict=True):
            if index in (0, len(alphas) - 1):
                diagnostics["alpha_boundary"] += 1
                continue
            alpha_x.append(features(config, int(batch), dimension))
            alpha_y.append(np.log(alphas[index]))
        valley = loss.min(axis=1)
        minimum = int(valley.argmin())
        if minimum == 0:
            diagnostics["batch_minimum_lower_boundary"] += 1
            continue
        good = np.flatnonzero(valley <= 1.002 * valley.min())
        last = int(good[-1])
        if last == len(batches) - 1:
            diagnostics["batch_knee_right_censored"] += 1
            continue
        # Linear loss interpolation in log batch, declared explicitly.
        fraction = (1.002 * valley.min() - valley[last]) / (valley[last + 1] - valley[last])
        knee = np.exp(np.log(batches[last]) + fraction * (np.log(batches[last + 1]) - np.log(batches[last])))
        batch_x.append(features(config, dimension=dimension))
        batch_y.append(np.log(knee))
    return (
        (np.asarray(alpha_x), np.asarray(alpha_y)),
        (np.asarray(batch_x), np.asarray(batch_y)),
        dict(diagnostics),
    )


def regression(x, y):
    """Reject unidentified fits instead of silently returning coefficients."""
    if len(x) == 0 or len(x) < x.shape[1] or np.linalg.matrix_rank(x) < x.shape[1]:
        return None
    coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y - x @ coefficients
    r2 = 1 - np.sum(residual**2) / np.sum((y - y.mean()) ** 2)
    return {
        "coefficients": coefficients.tolist(),
        "r2": float(r2),
        "targets": len(y),
        "condition_number": float(np.linalg.cond(x)),
    }


def fit(problems, dimension):
    """Fit both laws on the fit partition only."""
    step, batch, diagnostics = targets(problems, dimension)
    return {
        "dimension": dimension,
        "alpha_floor": float(min(p["alphas"][0] for p in problems)),
        "alpha": regression(*step),
        "batch": regression(*batch),
        "censoring": diagnostics,
    }


def predict(model, config):
    """Round predicted batches and clip steps to the declared search domain."""
    if model["batch"] is None or model["alpha"] is None:
        raise ValueError("Cannot predict with an unidentified law")
    log_batch = np.dot(features(config, dimension=model["dimension"]), model["batch"]["coefficients"])
    batch = min(config["n"], max(1, 2 ** int(np.floor(log_batch / np.log(2) + 0.5))))
    alpha = float(
        np.clip(
            np.exp(np.dot(features(config, batch, model["dimension"]), model["alpha"]["coefficients"])),
            model.get("alpha_floor", 1e-4),
            1,
        )
    )
    return batch, alpha


def main():
    """Write candidate coefficients, bootstrap intervals and transfer diagnostics."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/scaling_sweep.jsonl")
    parser.add_argument("--output", default="results/scaling_fit.json")
    args = parser.parse_args()
    problems = aggregate(args.input)
    train = [p for p in problems if p["config"]["partition"] == "fit"]
    models = {}
    for dimension in (False, True):
        name = "with_dimension" if dimension else "without_dimension"
        model = fit(train, dimension)
        rng = np.random.default_rng(517)
        boot = defaultdict(list)
        for _ in range(200):
            sample = [train[i] for i in rng.integers(0, len(train), len(train))]
            step, batch, _ = targets(sample, dimension, resample_seeds=rng)
            for kind, xy in [("alpha", step), ("batch", batch)]:
                result = regression(*xy)
                if result is not None:
                    boot[kind].append(result["coefficients"])
        for kind in ("alpha", "batch"):
            if model[kind] is not None and boot[kind]:
                model[kind]["bootstrap_95"] = np.quantile(boot[kind], [0.025, 0.975], axis=0).tolist()
                model[kind]["bootstrap_successes"] = len(boot[kind])
        regrets = defaultdict(list)
        if model["alpha"] is not None and model["batch"] is not None:
            for problem in problems:
                batch, alpha = predict(model, problem["config"])
                bi = int(np.abs(np.log(problem["batches"] / batch)).argmin())
                ai = int(np.abs(np.log(problem["alphas"] / alpha)).argmin())
                regret = float(problem["loss"][bi, ai] / problem["loss"].min() - 1)
                regrets[problem["config"]["partition"]].append(regret)
        model["nearest_grid_validation_regret"] = {
            group: {"median": float(np.median(v)), "p90": float(np.quantile(v, 0.9)), "count": len(v)}
            for group, v in regrets.items()
        }
        models[name] = model
    output = {
        "models": models,
        "problem_count": len(problems),
        "fit_problem_count": len(train),
        "alpha_feature_order": ["intercept", "log(k)", "log(B)", "log(n)", "log(T)", "optional log(d)"],
        "batch_feature_order": ["intercept", "log(k)", "log(n)", "log(T)", "optional log(d)"],
        "bootstrap": "200 problem-block bootstrap draws with seed resampling inside each problem",
        "warning": "Nearest-grid validation regret is a diagnostic. Continuous law predictions require independent reruns and test evaluation.",
    }
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
