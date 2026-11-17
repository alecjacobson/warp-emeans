"""Summarize fresh-seed runtime and paired quality uncertainty for every case."""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

result = json.loads(Path("results/equal_loss_test.json").read_text())
rows = result["records"]
summaries = []
rng = np.random.default_rng(903)
for name, epochs in result["selected_epochs"].items():
    if epochs is None:
        summaries.append({"dataset": name, "status": "validation_target_not_reached"})
        continue
    pairs = {
        backend: sorted(
            (r for r in rows if r["dataset"] == name and r["backend"] == backend), key=lambda r: r["seed"]
        )
        for backend in ("warp", "triton")
    }
    if len(pairs["warp"]) != 10 or len(pairs["triton"]) != 10:
        raise ValueError("Expected ten fresh paired seeds per backend")
    mse = {backend: np.array([r["mse"] for r in pairs[backend]]) for backend in pairs}
    duration = {backend: np.array([r["seconds"] for r in pairs[backend]]) for backend in pairs}
    indices = rng.integers(0, 10, size=(10000, 10))
    ratios = mse["warp"][indices].mean(axis=1) / mse["triton"][indices].mean(axis=1)
    speedups = np.median(duration["triton"][indices], axis=1) / np.median(duration["warp"][indices], axis=1)
    ci = np.quantile(ratios, [0.025, 0.975])
    summary = {
        "dataset": name,
        "n": pairs["warp"][0]["n"],
        "d": pairs["warp"][0]["d"],
        "k": pairs["warp"][0]["k"],
        "warp_epochs": epochs,
        "warp_ms": float(np.median(duration["warp"]) * 1000),
        "triton_ms": float(np.median(duration["triton"]) * 1000),
        "speedup": float(np.median(duration["triton"]) / np.median(duration["warp"])),
        "speedup_95": np.quantile(speedups, [0.025, 0.975]).tolist(),
        "test_mse_ratio": float(mse["warp"].mean() / mse["triton"].mean()),
        "test_mse_ratio_95": ci.tolist(),
        "quality_margin_pass": bool(ci[1] <= 1.002),
    }
    summaries.append(summary)
Path("results/equal_loss_summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
lines = [
    "# Equal-loss runtime confirmation on real data",
    "",
    "Warp epoch budgets were chosen on the validation split using seeds 0,1,2, before this test evaluation. The target was upstream Triton e-means at three epochs, with a predeclared 0.2% loss margin. The first passing budget among 1,2,3,5,10,25 was frozen. Test uses ten fresh paired initialization seeds (100–109), 16,384 test vectors, and FP64 squared-distance MSE. Backend order alternates. Each backend is warmed before measurement.",
    "",
    "All timings are ordinary train calls with resident GPU inputs, including allocation and validation; input transfers, JIT warmup and loss evaluation are excluded. CUDA graph preparation/replay is not used. Both implementations compute in FP32 on the RTX PRO 6000 Blackwell MIG slice. Warp and Triton use different shuffle/revival random streams. GPU experiments launched by this session were stopped/completed before test timing, but the machine can be shared by other users.",
    "",
    "The quality criterion is the upper endpoint of a paired 95% bootstrap interval for mean Warp test MSE / mean Triton test MSE being at most 1.002. Intervals resample ten seeds 10,000 times and describe initialization variation on these fixed datasets, not uncertainty across all applications. Every selected case is reported.",
    "",
    "| Dataset | d | n | k | Warp epochs | Triton ms | Warp ms | Speedup | Test MSE ratio [95% interval] | Within 0.2% margin? |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
]
for s in summaries:
    if "status" in s:
        lines.append(f"| {s['dataset']} | — | — | — | — | — | — | — | Validation target not reached | No |")
        continue
    lo, hi = s["test_mse_ratio_95"]
    lines.append(
        f"| {s['dataset']} | {s['d']} | {s['n']:,} | {s['k']} | {s['warp_epochs']} | {s['triton_ms']:.2f} | {s['warp_ms']:.2f} | {s['speedup']:.2f}× | {s['test_mse_ratio']:.5f} [{lo:.5f}, {hi:.5f}] | {'Yes' if s['quality_margin_pass'] else 'No' if lo > 1.002 else 'Inconclusive'} |"
    )
lines += [
    "",
    "Bunny XYZ meets the specified margin, narrowly: its interval upper endpoint is 1.00195. This is evidence for that fixed workload, not general equivalence. LabXY exceeds the margin; Intel XY, RGB and Bunny XYZ+normals remain inconclusive. Model/graph cleanup and garbage collection occur outside the timers; earlier exploratory timings affected by cleanup are excluded.",
    "",
    "![Runtime and loss confirmation](equal-loss.png)",
    "",
    "Raw evidence: [frozen validation choices](../../results/equal_loss_selection.json), [test runs](../../results/equal_loss_test.json), [summary](../../results/equal_loss_summary.json). [Dataset meanings and limitations](../datasets.md). The wider-dimensional performance gap remains [issue #6](https://github.com/alecjacobson/warp-emeans/issues/6).",
    "",
    "Reproduce with `uv run python scripts/benchmark_equal_loss.py --phase select`, then `--phase test`, and `uv run python scripts/report_equal_loss.py`. Avoid concurrent GPU experiments for the test phase.",
]
Path("docs/experiments/equal-loss.md").write_text("\n".join(lines) + "\n")
valid = [s for s in summaries if "status" not in s]
figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
y = np.arange(len(valid))
axes[0].barh(y - 0.18, [s["triton_ms"] for s in valid], height=0.35, label="Upstream Triton (3 epochs)")
axes[0].barh(y + 0.18, [s["warp_ms"] for s in valid], height=0.35, label="Warp (validation-selected epochs)")
axes[0].set(yticks=y, yticklabels=[s["dataset"] for s in valid], xlabel="Ordinary training (median ms)")
axes[0].legend(fontsize=8)
means = np.array([s["test_mse_ratio"] for s in valid])
interval = np.array([s["test_mse_ratio_95"] for s in valid])
axes[1].errorbar(
    100 * (means - 1),
    y,
    xerr=100 * np.stack((means - interval[:, 0], interval[:, 1] - means)),
    fmt="o",
    capsize=3,
)
axes[1].axvline(0.2, color="tab:red", ls="--", label="0.2% margin")
axes[1].axvline(0, color="black", lw=0.7)
axes[1].set(yticks=y, yticklabels=[s["dataset"] for s in valid], xlabel="Warp / Triton test loss − 1 (%)")
axes[1].legend(fontsize=8)
figure.savefig("docs/experiments/equal-loss.png", dpi=180)
figure.savefig("docs/experiments/equal-loss.svg")
plt.close(figure)
print(json.dumps(summaries, indent=2))
