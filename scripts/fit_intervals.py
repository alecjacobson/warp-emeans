"""Fit quality-valley intervals while retaining censored sweep observations."""

import argparse
import json
from pathlib import Path

import numpy as np
from fit_scaling import aggregate, features, predict
from scipy.optimize import least_squares


def valley_interval(axis, loss, tolerance=0.002):
    """Bracket the contiguous near-optimal valley around a measured minimum.

    Return log-coordinate bounds; an unbracketed end is infinite. Interpolate
    loss linearly in log coordinate. An entirely flat curve is uninformative.
    """
    axis = np.asarray(axis, dtype=float)
    loss = np.asarray(loss, dtype=float)
    target = loss.min() * (1 + tolerance)
    lower = upper = int(loss.argmin())
    while lower > 0 and loss[lower - 1] <= target:
        lower -= 1
    while upper < len(axis) - 1 and loss[upper + 1] <= target:
        upper += 1
    low = -np.inf
    high = np.inf
    if lower > 0:
        fraction = (target - loss[lower]) / (loss[lower - 1] - loss[lower])
        low = np.log(axis[lower]) + fraction * np.log(axis[lower - 1] / axis[lower])
    if upper < len(axis) - 1:
        fraction = (target - loss[upper]) / (loss[upper + 1] - loss[upper])
        high = np.log(axis[upper]) + fraction * np.log(axis[upper + 1] / axis[upper])
    return low, high


def observations(problems, dimension, rng=None):
    """Retain one-sided information from boundary minima and flat curves."""
    alpha_x, alpha_bounds, batch_x, batch_bounds = [], [], [], []
    for problem in problems:
        loss = problem["loss"]
        if rng is not None:
            ids = rng.integers(0, problem["values"].shape[2], problem["values"].shape[2])
            loss = np.median(problem["values"][:, :, ids], axis=2)
        config = problem["config"]
        for batch, curve in zip(problem["batches"], loss, strict=True):
            lo, hi = valley_interval(problem["alphas"], curve)
            if np.isfinite(lo) or np.isfinite(hi):
                alpha_x.append(features(config, int(batch), dimension))
                alpha_bounds.append((lo, hi))
        curve = loss.min(axis=1)
        _, knee = valley_interval(problem["batches"], curve)
        minimum = int(curve.argmin())
        if np.isfinite(knee):
            # A lower-boundary minimum only bounds the true knee from above,
            # conditional on a single right-hand valley. Do not make it a point.
            bounds = (-np.inf, knee) if minimum == 0 else (knee, knee)
        elif minimum == 0:
            continue
        else:
            bounds = (np.log(problem["batches"][-1]), np.inf)
        batch_x.append(features(config, dimension=dimension))
        batch_bounds.append(bounds)
    return (np.asarray(alpha_x), np.asarray(alpha_bounds)), (np.asarray(batch_x), np.asarray(batch_bounds))


def solve(x, bounds, prior):
    """Minimize log-interval violations with a declared weak reference prior."""

    def residual(coefficients):
        prediction = x @ coefficients
        violation = np.maximum(bounds[:, 0] - prediction, 0) - np.maximum(prediction - bounds[:, 1], 0)
        return np.concatenate((violation, np.sqrt(1e-4) * (coefficients - prior)))

    fit = least_squares(residual, prior, max_nfev=1000)
    if not fit.success:
        raise RuntimeError(f"Interval fit failed: {fit.message}")
    return {
        "coefficients": fit.x.tolist(),
        "targets": len(x),
        "rms_log_violation": float(np.sqrt(np.mean(residual(fit.x)[: len(x)] ** 2))),
    }


def main():
    """Fit candidate laws on fit configurations and preserve bootstrap uncertainty."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/scaling_canonical.jsonl")
    parser.add_argument("--output", default="results/scaling_intervals.json")
    args = parser.parse_args()
    problems = aggregate(args.input)
    train = [p for p in problems if p["config"]["partition"] == "fit"]
    models = {}
    for dimension in (False, True):
        suffix = [0.0] if dimension else []
        priors = {
            "alpha": np.array([np.log(0.8244), -0.4617, 0.9003, -0.4498, -0.1057] + suffix),
            "batch": np.array([np.log(0.00439175), 0.134749, 0.876473, 0.813344] + suffix),
        }
        data = observations(train, dimension)
        model = {"dimension": dimension, "alpha_floor": float(min(p["alphas"][0] for p in train))}
        for kind, xy in zip(("alpha", "batch"), data, strict=True):
            model[kind] = solve(*xy, priors[kind])
        rng = np.random.default_rng(194)
        boot = {kind: [] for kind in ("alpha", "batch")}
        for _ in range(100):
            sample = [train[i] for i in rng.integers(0, len(train), len(train))]
            data = observations(sample, dimension, rng)
            for kind, xy in zip(("alpha", "batch"), data, strict=True):
                boot[kind].append(solve(*xy, priors[kind])["coefficients"])
        for kind in boot:
            model[kind]["bootstrap_95"] = np.quantile(boot[kind], [0.025, 0.975], axis=0).tolist()
        regrets = {}
        for partition in sorted({p["config"]["partition"] for p in problems}):
            values = []
            for problem in problems:
                if problem["config"]["partition"] != partition:
                    continue
                batch, alpha = predict(model, problem["config"])
                bi = np.abs(np.log(problem["batches"] / batch)).argmin()
                ai = np.abs(np.log(problem["alphas"] / alpha)).argmin()
                values.append(float(problem["loss"][bi, ai] / problem["loss"].min() - 1))
            regrets[partition] = {
                "median": float(np.median(values)),
                "p90": float(np.quantile(values, 0.9)),
                "count": len(values),
            }
        model["nearest_grid_validation_regret"] = regrets
        models["interval_dimension" if dimension else "interval_no_dimension"] = model
    output = {
        "models": models,
        "method": "0.2% contiguous validation valleys; log-coordinate interpolation; one-sided censoring; weak ridge 1e-4 toward upstream quality law; 100 problem/seed bootstrap draws",
        "warning": "Candidate extension to Appendix B, not the paper fitting method. Batch upper bounds assume a single right-hand valley; evaluate actual predictions independently.",
    }
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
