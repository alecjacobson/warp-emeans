"""Summarize fixed-budget timing and paired Lloyd quality confirmation."""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def table(headers, rows):
    """Build a Markdown table with matching header and delimiter widths."""
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *["| " + " | ".join(map(str, row)) + " |" for row in rows],
    ]


def main():
    """Generate the report from archived evidence without running GPU experiments."""
    matrix = json.loads(Path("results/kmeans_matrix.json").read_text())
    confirmation = json.loads(Path("results/kmeans_test.json").read_text())
    replication = json.loads(Path("results/kmeans_timing_replication.json").read_text())
    replicated_times = {
        name: {
            backend: float(
                np.median(
                    [
                        r["seconds"]
                        for r in replication["records"]
                        if r["dataset"] == name and r["backend"] == backend
                    ]
                )
                * 1000
            )
            for backend in ("warp_lloyd", "warp_emeans", "triton_emeans")
        }
        for name in dict.fromkeys(r["dataset"] for r in replication["records"])
    }
    groups = defaultdict(list)
    for row in matrix["records"]:
        groups[tuple(row[key] for key in ("dataset", "n", "d", "k"))].append(row)
    matrix_summary = []
    for (name, n, d, k), rows in groups.items():
        summary = {"dataset": name, "n": n, "d": d, "k": k}
        for backend in ("warp_lloyd", "warp_emeans", "triton_emeans"):
            for graph in (False, True):
                selected = [r for r in rows if r["backend"] == backend and r["graph"] == graph]
                if selected:
                    summary[backend + ("_graph" if graph else "")] = {
                        "ms": float(np.median([r["seconds"] for r in selected]) * 1000),
                        "mse": float(np.mean([r["mse"] for r in selected])),
                        "setup_ms": float(np.median([r["setup_seconds"] for r in selected]) * 1000)
                        if graph
                        else None,
                    }
        matrix_summary.append(summary)
    rng = np.random.default_rng(903)
    tests = []
    for name, budget in confirmation["selected_iterations"].items():
        rows = [r for r in confirmation["records"] if r["dataset"] == name]
        paired = {}
        summary = {"dataset": name, "selected_iterations": budget}
        for backend in ("warp_lloyd", "warp_emeans", "triton_emeans"):
            selected = sorted((r for r in rows if r["backend"] == backend), key=lambda r: r["seed"])
            if [r["seed"] for r in selected] != list(range(200, 210)):
                raise ValueError(f"Expected ten paired seeds for {name} {backend}")
            paired[backend] = np.array([r["mse"] for r in selected])
            summary[backend + "_ms"] = float(np.median([r["seconds"] for r in selected]) * 1000)
        indices = rng.integers(0, 10, size=(10000, 10))
        for comparator in ("warp_emeans", "triton_emeans"):
            ratios = paired["warp_lloyd"][indices].mean(axis=1) / paired[comparator][indices].mean(axis=1)
            interval = np.quantile(ratios, [0.025, 0.975]).tolist()
            summary[comparator + "_comparison"] = {
                "loss_ratio": float(paired["warp_lloyd"].mean() / paired[comparator].mean()),
                "loss_ratio_95": interval,
                "margin_pass": bool(budget is not None and interval[1] <= 1.002),
                "time_reduction_percent": 100 * (1 - summary["warp_lloyd_ms"] / summary[comparator + "_ms"]),
            }
        tests.append(summary)
    scaling_rows = [
        json.loads(line) for line in Path("results/kmeans_scaling.jsonl").read_text().splitlines()
    ]
    scaling_groups = defaultdict(dict)
    for row in scaling_rows:
        identity = tuple(row[key] for key in ("dataset", "n", "d", "k", "epochs", "seed"))
        scaling_groups[identity][row["model"]] = row
    scaling = defaultdict(list)
    for group in scaling_groups.values():
        reference = group["upstream_auto"]
        for method in ("warp_lloyd25", "warp_lloyd100"):
            scaling[(reference["partition"], method)].append(
                group[method]["test_mse"] / reference["test_mse"] - 1
            )
    scaling_summary = [
        {
            "partition": partition,
            "method": method,
            "runs": len(values),
            "median_percent": float(100 * np.median(values)),
            "p90_percent": float(100 * np.quantile(values, 0.9)),
        }
        for (partition, method), values in scaling.items()
    ]
    Path("results/kmeans_summary.json").write_text(
        json.dumps(
            {
                "matrix": matrix_summary,
                "confirmation": tests,
                "scaling": scaling_summary,
                "timing_replication_ms": replicated_times,
            },
            indent=2,
        )
        + "\n"
    )
    passed = [r["dataset"] for r in tests if r["triton_emeans_comparison"]["margin_pass"]]
    lines = [
        "# Pure Warp Lloyd k-means baseline",
        "",
        "`warp_emeans.Kmeans` is a full-data Lloyd baseline. Every iteration assigns all points and updates every nonempty center. All clustering kernels, initialization, scans and sorting use Warp; no PyTorch, Triton or Faiss is used by the implementation. Those packages appear only in reference experiments. See the [API contract](../api.md#full-data-lloyd-baseline).",
        "",
        f"The validation-selected baseline satisfies the predeclared 0.2% loss margin against three-pass upstream Triton e-means on **{len(passed)} of seven** real feature datasets in this confirmation: {', '.join(passed) or 'none'}. Reported intervals quantify initialization variation on these fixed inputs. They do not establish superiority across graphics or robotics applications.",
        "",
        "## Implementation and tuning",
        "",
        "The direct path keeps one point in registers per thread, scans centers without an n-by-k distance matrix, and fuses assignment with partial sum/count atomics. CUDA uses up to 64 accumulation shards to reduce contention, followed by a mean reduction. Empty centers are retained; FP32 distance ties choose the smallest center index. Random initialization samples input row indices without replacement, not k-means++. No early stopping is hidden in the iteration budget.",
        "",
        "The large-input path maintains a lower bound on competing-center distance. It subtracts the maximum center movement, tests the current winner, scans flags on device, and groups points requiring another exhaustive search. No active-count readback is needed. The automatic heuristic selects it on CUDA for n>=500,000, 3<=d<=10, k>=256, at least ten iterations, and either d>=5 or k>=512. Other workloads use direct search. This policy was tuned on the present Blackwell GPU; hardware and data distribution can change the best choice.",
        "",
        "Bounds use FP64 state and outward relaxation; searched squared distances use direct FP32 subtraction/multiplication/accumulation. Relative guard 16*d*2^-23 and absolute distance guard d*2^-60 cover FP32 rounding and underflow in the supported low-dimensional path. The winner is skipped only with a strict guarded separation, so near ties trigger a search. Maximum movement is inflated by (1+1e-12) plus 1e-30; relaxed bounds shrink by the reciprocal-side margin. Coordinate validation rejects magnitudes above 2^60/sqrt(d), preventing squared-distance and reduction overflow within the indexing limits. Atomic mean updates still vary in the last bits, so this is not a claim of bitwise equivalence to an FP64 Lloyd trajectory.",
        "",
        "Distance-bound pruning follows the idea of [Hamerly's accelerated Lloyd method](https://cs.baylor.edu/~hamerly/papers/sdm_2010.pdf), with conservative guards and GPU compaction added here. Bounds without compaction did not improve the tested large workloads because partially active warps still perform searches. Cooperative searches, direct Warp tiles, TF32 tile products with direct-distance refinement, and forced wide-dimension unrolling were also tested. They did not justify replacing the direct path across the tested configurations. Experimental variants remain in `scripts/kmeans_variants.py`, outside the public runtime.",
        "",
        "The committed `kmeans_tuning_*.json` files retain exploratory probes, including losing variants and raw samples. Early probes include intermediate source hashes and are not the final performance evidence. The final matrix below measures the public API from the frozen source commit recorded in its metadata. No claim is made that this is the fastest possible k-means implementation on every GPU.",
        "",
        "## Fixed-budget matrix",
        "",
        "Same 15 synthetic configurations as the existing e-means runtime matrix, plus all seven real feature datasets. Synthetic inputs use the existing normal-distribution generator; real inputs use the [documented corpus](../datasets.md). All methods receive identical initial centers for each of seeds 0,1,2. Lloyd uses 25 full iterations; both e-means implementations use three passes. These are different algorithms and budgets: runtime alone does not establish matched quality.",
        "",
        "Timings are medians across three seed medians, each with three warm measurements. Ordinary calls include input validation and allocation. Transfers, JIT warmup, model/graph cleanup and FP64 loss evaluation are outside the timers. Graph replay includes reset and all iterations; its preparation is reported separately. Method order alternates across seeds. All inputs and training arithmetic are FP32 on an NVIDIA RTX PRO 6000 Blackwell MIG 2g.48gb, Warp 1.19.0. This session ran GPU experiments sequentially; the machine can be shared by other users.",
        "",
        "Loss columns show percent change in held-out FP64 MSE; negative is lower loss. Three seeds are descriptive, not an equivalence test.",
        "",
    ]
    rows = []
    for row in matrix_summary:
        lloyd, warp, triton = [row[backend] for backend in ("warp_lloyd", "warp_emeans", "triton_emeans")]
        rows.append(
            [
                row["dataset"],
                f"{row['n']:,}",
                row["d"],
                row["k"],
                f"{lloyd['ms']:.2f}",
                f"{warp['ms']:.2f}",
                f"{triton['ms']:.2f}",
                f"{100 * (lloyd['mse'] / warp['mse'] - 1):+.2f}%",
                f"{100 * (lloyd['mse'] / triton['mse'] - 1):+.2f}%",
            ]
        )
    lines += table(
        [
            "Dataset",
            "n",
            "d",
            "k",
            "Lloyd 25 ms",
            "Warp e-means ms",
            "Triton e-means ms",
            "Lloyd loss vs Warp",
            "Lloyd loss vs Triton",
        ],
        rows,
    )
    lines += [
        "",
        "### Prepared graph replay",
        "",
        "Each entry is replay ms / preparation ms (including warmup and capture). Setup is paid once per prepared workload.",
        "",
    ]
    lines += table(
        ["Dataset", "n", "d", "k", "Lloyd 25 replay / setup ms", "Warp e-means replay / setup ms"],
        [
            [
                r["dataset"],
                f"{r['n']:,}",
                r["d"],
                r["k"],
                *[
                    f"{r[b]['ms']:.2f} / {r[b]['setup_ms']:.2f}"
                    for b in ("warp_lloyd_graph", "warp_emeans_graph")
                ],
            ]
            for r in matrix_summary
        ],
    )
    lines += [
        "",
        "## Validation-selected quality confirmation",
        "",
        "The first Lloyd budget among 1,3,5,10,25,50,100,200 whose mean validation MSE was at most 1.002 times upstream Triton's three-pass MSE was frozen using seeds 0,1,2 and 16,384 validation vectors. Confirmation uses ten new paired initialization seeds 200–209 and 16,384 test vectors. The test split was also used in earlier studies; only initialization seeds are new here. It never selects the Lloyd budget. Cases missing the validation target are still reported at 200 iterations and cannot pass confirmation.",
        "",
        "Each timing is the median across seeds of three warmed ordinary train measurements. A paired bootstrap resamples the ten seeds 10,000 times; a match requires the 95% interval upper endpoint for Lloyd/Triton mean loss to be <=1.002. Intervals below are expressed as percent loss difference, so the threshold is +0.2%. The Warp e-means comparator remains at three passes and has separate paired statistics in the summary JSON.",
        "",
    ]
    rows = []
    for row in tests:
        comparison = row["triton_emeans_comparison"]
        lo, hi = [100 * (x - 1) for x in comparison["loss_ratio_95"]]
        status = (
            "Yes"
            if comparison["margin_pass"]
            else "No validation match"
            if row["selected_iterations"] is None
            else "No"
            if lo > 0.2
            else "Inconclusive"
        )
        rows.append(
            [
                row["dataset"],
                row["selected_iterations"] or "200 (no match)",
                f"{row['warp_lloyd_ms']:.2f}",
                f"{row['warp_emeans_ms']:.2f}",
                f"{row['triton_emeans_ms']:.2f}",
                f"{100 * (comparison['loss_ratio'] - 1):+.2f}% [{lo:+.2f}, {hi:+.2f}]",
                status,
            ]
        )
    lines += table(
        [
            "Dataset",
            "Lloyd iterations",
            "Lloyd ms",
            "Warp e-means ms",
            "Triton e-means ms",
            "Lloyd loss vs Triton [95% interval]",
            "Within margin?",
        ],
        rows,
    )
    lines += [
        "",
        "### Timing replication",
        "",
        "The first confirmation showed intermittent Triton timing outliers in the later cases. A fresh-process replication repeated RGB, Bunny XYZ+normals and both drill cases with five warm measurements per seed, unchanged budgets, and the same ten seeds. The elevated Triton medians did not reproduce; their cause was not isolated. Both archives are retained. Use the replicated times below for those cases; the original quality analysis above remains frozen. RGB's flagship timing is stable across both runs. Entries below are original / replicated ordinary milliseconds.",
        "",
    ]
    lines += table(
        ["Dataset", "Lloyd ms", "Warp e-means ms", "Triton e-means ms"],
        [
            [
                row["dataset"],
                *[
                    f"{row[backend + '_ms']:.2f} / {replicated_times[row['dataset']][backend]:.2f}"
                    for backend in ("warp_lloyd", "warp_emeans", "triton_emeans")
                ],
            ]
            for row in tests
            if row["dataset"] in replicated_times
        ],
    )
    lines += [
        "",
        "![Lloyd runtime and quality confirmation](kmeans-confirmation.png)",
        "The figure uses replicated timings where available and the original paired quality intervals.",
        "",
        "## Scaling-study integration",
        "",
        "The evaluation harness now includes `warp_lloyd25` and `warp_lloyd100`; historical `lloyd25` remains the Faiss CPU baseline. A fresh run evaluates both Warp budgets and upstream-auto Warp e-means on all 33 held-out configurations, using the original seed/split protocol (20–22, 8,192 test rows at offset 16,384). This is a baseline addition on known test data, not a new independent validation of the scaling laws. E-means uses the configured two passes. Timings from this evaluation are not performance evidence.",
        "",
    ]
    lines += table(
        ["Partition", "Method", "Runs", "Median loss vs auto", "90th percentile"],
        [
            [
                r["partition"],
                r["method"],
                r["runs"],
                f"{r['median_percent']:+.2f}%",
                f"{r['p90_percent']:+.2f}%",
            ]
            for r in scaling_summary
        ],
    )
    lines += [
        "",
        "## Evidence and reproduction",
        "",
        "Raw [matrix](../../results/kmeans_matrix.json), [validation selection](../../results/kmeans_selection.json), [confirmation](../../results/kmeans_test.json), [timing replication](../../results/kmeans_timing_replication.json), [scaling evaluation](../../results/kmeans_scaling.jsonl), and [summary](../../results/kmeans_summary.json) retain per-seed measurements. Timing files record source hashes and the source commit; the confirmation hashes its frozen selection file. [Validation evidence](../../results/kmeans_validation.json) records tests and sanitizer results. [Issue #14](https://github.com/alecjacobson/warp-emeans/issues/14) tracks implementation and results.",
        "",
        "Use fresh output paths; benchmark scripts reject overwriting evidence. Reporting reads the published filenames. Run GPU measurements sequentially:",
        "",
        "```sh",
        "uv sync --extra research --extra dev",
        "uv run python scripts/benchmark_kmeans.py --phase matrix --output results/rerun_matrix.json",
        "uv run python scripts/benchmark_kmeans.py --phase select --output results/rerun_selection.json",
        "uv run python scripts/benchmark_kmeans.py --phase test --selection results/rerun_selection.json --output results/rerun_test.json",
        "uv run python scripts/benchmark_kmeans.py --phase test --selection results/rerun_selection.json --datasets bsds_rgb bunny_xyzn drill_xyz drill_xyzn --repeats 5 --output results/rerun_timing_replication.json",
        "uv run python scripts/evaluate_scaling.py --methods upstream_auto warp_lloyd25 warp_lloyd100 --output results/rerun_scaling.jsonl",
        "uv run python scripts/report_kmeans.py  # regenerate the archived report",
        "```",
        "",
        "Remaining limitations: one GPU; a small corpus with shared source scenes; FP32 atomic nondeterminism; retained empty centers can limit quality; no k-means++ initialization, spherical metric or distributed/autodiff support. Fixed-budget performance differs from performance at matched loss. The high-dimensional cases and datasets that miss the validation/test margin must remain visible.",
    ]
    Path("docs/experiments/kmeans-baseline.md").write_text("\n".join(lines) + "\n")
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    y = np.arange(len(tests))
    for offset, backend, label in [
        (-0.24, "triton_emeans", "Triton e-means, 3 passes"),
        (0, "warp_emeans", "Warp e-means, 3 passes"),
        (0.24, "warp_lloyd", "Warp Lloyd, selected budget"),
    ]:
        axes[0].barh(
            y + offset,
            [
                replicated_times[r["dataset"]][backend]
                if r["dataset"] in replicated_times
                else r[backend + "_ms"]
                for r in tests
            ],
            height=0.23,
            label=label,
        )
    axes[0].set(yticks=y, yticklabels=[r["dataset"] for r in tests], xlabel="Ordinary training (median ms)")
    axes[0].legend(fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.01))
    for index, row in enumerate(tests):
        comparison = row["triton_emeans_comparison"]
        lo, hi = 100 * (np.array(comparison["loss_ratio_95"]) - 1)
        axes[1].hlines(index, lo, hi, color="tab:blue")
        axes[1].plot(100 * (comparison["loss_ratio"] - 1), index, "o", color="tab:blue")
    axes[1].axvline(0.2, ls="--", color="tab:red", label="0.2% margin")
    axes[1].axvline(0, lw=0.7, color="black")
    axes[1].set_xscale("symlog", linthresh=3)
    axes[1].set_xticks([-3, -1, 0, 1, 3, 10, 30], labels=["−3", "−1", "0", "1", "3", "10", "30"])
    axes[1].set(
        yticks=y,
        yticklabels=[r["dataset"] for r in tests],
        xlabel="Lloyd loss relative to Triton e-means (%)\nLinear within ±3%; logarithmic beyond",
    )
    axes[1].legend(fontsize=8)
    figure.savefig("docs/experiments/kmeans-confirmation.png", dpi=180)
    plt.close(figure)
    print(json.dumps(tests, indent=2))


if __name__ == "__main__":
    main()
