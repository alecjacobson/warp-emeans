"""Generate a reproducible scientific report and standalone figures."""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fit_scaling import aggregate

ROOT = Path("docs/experiments")
ROOT.mkdir(parents=True, exist_ok=True)
problems = aggregate("results/scaling_canonical.jsonl")
fits = json.loads(Path("results/scaling_intervals.json").read_text())["models"]
point = json.loads(Path("results/scaling_canonical_fit.json").read_text())["models"]
rows = [json.loads(line) for line in Path("results/scaling_evaluation.jsonl").read_text().splitlines()]
groups = defaultdict(lambda: defaultdict(list))
for row in rows:
    if row["status"] == "ok":
        key = (row["dataset"], row["d"], row["n"], row["k"], row["epochs"], row["partition"])
        groups[key][row["model"]].append(row["test_mse"])
methods = [
    "upstream_auto",
    "upstream_quality",
    "upstream_throughput",
    "interval_no_dimension",
    "interval_dimension",
    "lloyd25",
]
regrets = defaultdict(lambda: defaultdict(list))
for key, values in groups.items():
    oracle = np.median(values["validation_oracle"])
    for method in methods:
        if method in values:
            regrets[key[-1]][method].append(100 * (np.median(values[method]) / oracle - 1))
lines = [
    "# Low-dimensional scaling study",
    "",
    "This study produces candidate empirical laws, not a universal low-dimensional scaling law. The implementation retains upstream defaults. Candidates are fitted on synthetic distributions and tested on unseen dimensions, distribution families and real feature datasets. All results below are measured; boundary censoring, three-seed uncertainty, and corpus coverage limit the conclusions.",
    "",
    "## Coverage and protocol",
    "",
    f"The larger sweep contains {sum(p['values'].size for p in problems):,} training runs across {len(problems)} configurations. The preceding pilot contains 14,040 runs across 104 configurations. Both use three initialization seeds per cell. The larger fit set uses n=100,000/1,000,000, k=64/256, T=1/3, d=2,3,5,10, with uniform and anisotropic mixture distributions. Held-out dimensions are 4,6,7,8,9,16,32,64,100. Gaussian and noisy two-angle manifold controls test distribution transfer. Seven real feature datasets cover d=2,3,5,6; see the [dataset manifest and application sources](../datasets.md).",
    "",
    "The batch ladder is 32,128,512,2048,8192,32768,n; the step ladder is 1e-5,1e-4,1e-3,.01,.03,.1,.3,1. Configurations exceeding 32,768 updates are explicitly omitted, so some large-n small-batch optima remain censored. Training uses FP32 Warp; validation uses 4,096 independent vectors and FP64 direct squared distances. The final test comparison uses fresh seeds 20,21,22 and 8,192 test vectors beginning at offset 16,384, disjoint from the pilot evaluation sample. No test loss selected the candidate coefficients.",
    "",
    "The reference fitting target follows [Appendix B](https://arxiv.org/html/2611.09501v1#A2): conditional step minima and a batch knee at +0.2% validation loss. The finite grid cannot establish that precise regret universally. An additional interval fit retains one-sided boundary information and treats flat step valleys as unidentified; it uses a declared weak ridge toward the upstream quality law. Both candidates below are reported without choosing a winner on test loss.",
    "",
    "## Candidate coefficients",
    "",
    "The forms are `alpha0 = clip(Ca k^ak B^aB n^an T^aT d^ad, 1e-5, 1)` and `B = round_power_of_two(Cb k^bk n^bn T^bT d^bd)`, capped to [1,n]. Omitting d fixes its exponent at zero. These are experimental settings, not replacement defaults.",
    "",
    "| Candidate | Ca | ak | aB | an | aT | ad |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
for name, model in fits.items():
    c = model["alpha"]["coefficients"] + [0.0] * (not model["dimension"])
    lines.append(f"| {name} | {np.exp(c[0]):.5g} | " + " | ".join(f"{v:.4f}" for v in c[1:]) + " |")
lines += ["", "| Candidate | Cb | bk | bn | bT | bd |", "|---|---:|---:|---:|---:|---:|"]
for name, model in fits.items():
    c = model["batch"]["coefficients"] + [0.0] * (not model["dimension"])
    lines.append(f"| {name} | {np.exp(c[0]):.5g} | " + " | ".join(f"{v:.4f}" for v in c[1:]) + " |")
lines += [
    "",
    "Coefficient bootstrap intervals and residuals are in [scaling_intervals.json](../../results/scaling_intervals.json). The 100 bootstrap draws resample problem configurations and seeds within this finite design; they do not estimate population uncertainty across arbitrary applications. The two source distributions and shared nested samples limit effective independence.",
    "",
    "The ordinary point-minimum fit is retained in [scaling_canonical_fit.json](../../results/scaling_canonical_fit.json). Its censoring diagnostics are: `"
    + json.dumps(point["without_dimension"]["censoring"])
    + "`. Do not interpret a flat curve’s arbitrarily selected minimum as an identified optimum.",
    "",
    "## Independent test comparison",
    "",
    "Entries are median / 90th-percentile percent loss relative to the validation-selected grid setting, rerun on the same fresh seeds. Negative values mean a method beat that validation-selected setting on test; the comparator is not a test-selected optimum. Lloyd uses 25 iterations with all training rows; e-means transfer runs use two passes. These are loss comparisons, not cross-device timing comparisons.",
    "",
    "| Method | Unseen dimensions | Unseen families | Real feature datasets |",
    "|---|---:|---:|---:|",
]
for method in methods:
    fields = []
    for partition in ("dimension_holdout", "family_holdout", "real_holdout"):
        values = regrets[partition][method]
        fields.append(f"{np.median(values):+.3f}% / {np.quantile(values, 0.9):+.3f}%")
    lines.append("| " + method + " | " + " | ".join(fields) + " |")
lines += [
    "",
    "| Real dataset | Upstream auto | No-dimension candidate | Dimension candidate |",
    "|---|---:|---:|---:|",
]
for key, values in groups.items():
    if key[-1] != "real_holdout":
        continue
    oracle = np.median(values["validation_oracle"])
    fields = [
        f"{100 * (np.median(values[m]) / oracle - 1):+.3f}%"
        for m in ("upstream_auto", "interval_no_dimension", "interval_dimension")
    ]
    lines.append("| " + key[0] + " | " + " | ".join(fields) + " |")
lines += [
    "",
    "The real-data table has seven feature datasets, not seven independent application domains: RGB and LabXY share images; XYZ and XYZ+normal variants share two meshes; Intel is one robot trajectory with revisited places. Surface samples are independent points on fixed reconstructions, not independent acquired scans. SLIC and VCCS features are Euclidean surrogates without their original locality/connectivity constraints.",
    "",
    "## Revival ablation",
    "",
    "Both arms retune B and alpha on the same validation ladder. Entries show best median validation MSE with revival on relative to off; three seeds do not establish significance. This checks a confound: the default recipe changes revival at d=2, so the dimension exponent is conditional on that policy.",
    "",
    "| d | Uniform on/off difference | Mixture on/off difference |",
    "|---|---:|---:|",
]
revival = aggregate("results/scaling_revival.jsonl")
for d in (2, 3, 5, 10):
    fields = []
    for family in ("uniform", "mixture"):
        arms = {
            p["config"]["partition"]: p["loss"].min()
            for p in revival
            if p["config"]["d"] == d and p["config"]["dataset"] == family
        }
        fields.append(f"{100 * (arms['revival_on'] / arms['revival_off'] - 1):+.3f}%")
    lines.append(f"| {d} | " + " | ".join(fields) + " |")
lines += [
    "",
    "## Figures",
    "",
    "![Validation step and batch curves](scaling-valleys.png)",
    "",
    "![Independent test loss comparison](scaling-transfer.png)",
    "",
    "## Reproduction and remaining work",
    "",
    "To regenerate the published report and figures from the committed measurements, run `uv run python scripts/report_scaling.py` followed by `uv run python scripts/report_real_confirmation.py`.\n\nFor a new measurement run, use a fresh destination so the original evidence remains intact:\n\n```sh\nuv sync --extra research --extra dev\nuv run python scripts/build_corpus.py\nmkdir -p results/rerun\nuv run python scripts/sweep_scaling.py --profile canonical --output results/rerun/canonical.jsonl\nuv run python scripts/fit_scaling.py --input results/rerun/canonical.jsonl --output results/rerun/point_fit.json\nuv run python scripts/fit_intervals.py --input results/rerun/canonical.jsonl --output results/rerun/intervals.json\nuv run python scripts/evaluate_scaling.py --sweep results/rerun/canonical.jsonl --fits results/rerun/intervals.json --output results/rerun/evaluation.jsonl\nuv run python scripts/evaluate_scaling.py --sweep results/rerun/canonical.jsonl --fits results/rerun/intervals.json --output results/rerun/confirmation.jsonl --only-real --seed-start 30 --seed-count 30 --test-offset 24576 --test-rows 4096 --methods upstream_auto validation_oracle interval_no_dimension interval_dimension\nuv run python scripts/sweep_scaling.py --profile revival --output results/rerun/revival.jsonl\n```\n\nResume is allowed only when recorded provenance matches; changed code or settings require a fresh output. The larger sweep's generating files were verified against commit `682326e4cf238ae37587c1fd87a4d2b9fe9f6ae7`, recorded in its metadata. That version predates the subnormal revival fix and transposed CUDA center layout. The subnormal corner is outside the ordinary-scale corpus; final performance and correctness tests use the fixes. Archived evaluations have less complete source metadata than the sweep; newer evaluation runs record code and input hashes. The current report scripts intentionally read the published result filenames; compare reruns separately before replacing any evidence.",
    "",
    "Raw append-only records: [larger sweep](../../results/scaling_canonical.jsonl), [pilot](../../results/scaling_sweep.jsonl), [fresh-seed test evaluation](../../results/scaling_evaluation.jsonl), [revival](../../results/scaling_revival.jsonl). Companion meta files record recipes, grids, software/source fingerprints and evaluation provenance. Exact repeated FP32 outputs can differ because atomics and split thresholds are sensitive to accumulation order.",
    "",
    "Further work should refine the grid near identified valleys, widen censored small-batch regions, add independent scenes and robot logs, fit multiple distribution realizations, and evaluate more seeds before changing defaults. Study intrinsic dimension and covariance/cluster imbalance separately from ambient dimension. Strong equal-loss timing evidence is tracked in the [real-data benchmark](equal-loss.md); wider-dimensional runtime optimization remains [issue #6](https://github.com/alecjacobson/warp-emeans/issues/6).",
]
(ROOT / "scaling.md").write_text("\n".join(lines) + "\n")

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
example = next(
    p
    for p in problems
    if p["config"] == dict(dataset="uniform", d=3, n=100000, k=64, epochs=3, partition="fit")
)
for batch, curve in zip(example["batches"], example["loss"], strict=True):
    axes[0].plot(
        example["alphas"], 100 * (curve / example["loss"].min() - 1), marker="o", ms=3, label=f"B={batch:,}"
    )
axes[0].set(
    xscale="log",
    yscale="symlog",
    xlabel="Initial EMA step",
    ylabel="Validation loss above grid minimum (%)",
    title="Step valleys: uniform d=3, n=100k, k=64, T=3",
)
axes[0].legend(fontsize=8, ncols=2)
axes[0].set_ylim(bottom=0)
for d in (2, 3, 5, 10):
    p = next(
        p
        for p in problems
        if p["config"] == dict(dataset="uniform", d=d, n=100000, k=64, epochs=3, partition="fit")
    )
    axes[1].plot(
        p["batches"], 100 * (p["loss"].min(axis=1) / p["loss"].min() - 1), marker="o", label=f"d={d}"
    )
axes[1].axhline(0.2, color="black", ls=":", lw=1, label="0.2% knee threshold")
axes[1].set(
    xscale="log",
    yscale="symlog",
    xlabel="Batch size (step retuned at every point)",
    ylabel="Validation loss above grid minimum (%)",
    title="Batch valleys: uniform, n=100k, k=64, T=3",
)
axes[1].legend(fontsize=8)
axes[1].set_ylim(bottom=0)
figure.savefig(ROOT / "scaling-valleys.png", dpi=180)
figure.savefig(ROOT / "scaling-valleys.svg")
plt.close(figure)
figure, axes = plt.subplots(1, 3, figsize=(13, 4.3), layout="constrained")
shown = ["upstream_auto", "interval_no_dimension", "interval_dimension", "lloyd25"]
labels = ["Upstream auto", "Fit without d", "Fit with d", "Lloyd 25"]
for axis, partition, title in zip(
    axes,
    ("dimension_holdout", "family_holdout", "real_holdout"),
    ("Unseen dimensions", "Unseen families", "Real feature datasets"),
    strict=True,
):
    values = [regrets[partition][m] for m in shown]
    axis.boxplot(values, tick_labels=labels, showfliers=True)
    axis.axhline(0, color="black", lw=0.7)
    axis.tick_params(axis="x", rotation=35)
    axis.set(title=title, ylabel="Test loss relative to validation oracle (%)")
figure.savefig(ROOT / "scaling-transfer.png", dpi=180)
figure.savefig(ROOT / "scaling-transfer.svg")
plt.close(figure)
for svg in ROOT.glob("scaling-*.svg"):
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
print(ROOT / "scaling.md")
