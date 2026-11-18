"""Generate the Flash-KMeans comparison from archived measurements."""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from report_kmeans import table

BACKENDS = ["warp_lloyd", "flash_tf32", "flash_ieee", "flash_fp16", "warp_emeans", "triton_emeans"]
LABELS = ["Warp Lloyd", "Flash TF32", "Flash IEEE", "Flash FP16", "Warp e-means", "Triton e-means"]


def summarize(rows, expected_seeds, backends):
    """Reject missing pairs and calculate timing and paired initialization intervals."""
    summary = {}
    losses = {}
    indices = np.random.default_rng(903).integers(0, len(expected_seeds), size=(10000, len(expected_seeds)))
    for backend in backends:
        selected = sorted((row for row in rows if row["backend"] == backend), key=lambda row: row["seed"])
        if [row["seed"] for row in selected] != list(expected_seeds):
            raise ValueError(f"Missing or duplicate paired seeds: {backend}")
        losses[backend] = np.array([row["mse"] for row in selected])
        summary[backend] = {
            "milliseconds": float(np.median([row["seconds"] for row in selected]) * 1000),
            "mse": float(losses[backend].mean()),
            "budget": selected[0]["budget"],
        }
        if len({row["budget"] for row in selected}) != 1:
            raise ValueError("Iteration budgets vary within a comparison")
    for backend in backends:
        if backend == "warp_lloyd":
            continue
        ratio = losses["warp_lloyd"].mean() / losses[backend].mean()
        interval = np.quantile(
            losses["warp_lloyd"][indices].mean(axis=1) / losses[backend][indices].mean(axis=1), [0.025, 0.975]
        )
        summary[backend]["warp_loss_percent"] = float(100 * (ratio - 1))
        summary[backend]["warp_loss_percent_95"] = (
            (100 * (interval - 1)).tolist() if len(expected_seeds) == 10 else None
        )
        summary[backend]["warp_loss_margin_pass"] = (
            bool(interval[1] <= 1.002) if len(expected_seeds) == 10 else None
        )
        summary[backend]["warp_time_reduction_percent"] = float(
            100 * (1 - summary["warp_lloyd"]["milliseconds"] / summary[backend]["milliseconds"])
        )
    return summary


def grouped(rows, keys):
    """Collect records by an explicit scientific workload identity."""
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    return groups


def main():
    """Write complete tables, provenance and limitations without rerunning training."""
    matrix = json.loads(Path("results/flash_matrix.json").read_text())
    confirmation = json.loads(Path("results/flash_confirmation.json").read_text())
    scaling = json.loads(Path("results/flash_scaling.json").read_text())
    selected = json.loads(Path("results/kmeans_selection.json").read_text())["selected_iterations"]
    for archive, expected_cases, expected_count in [
        (matrix, 22, 396),
        (confirmation, 7, 420),
        (scaling, 33, 891),
    ]:
        if len(archive["records"]) != expected_count or len(archive["workers"]) != 2 * expected_cases:
            raise ValueError(f"Incomplete {archive['phase']} archive")
        # Precision processes must have received identical training/evaluation arrays.
        for workers in grouped(archive["workers"], ["data_sha256", "evaluation_sha256"]).values():
            if len(workers) != 2 or {w["reference"]["dot_precision"] for w in workers} != {"tf32", "ieee"}:
                raise ValueError("Unpaired precision workers or mismatched data")
        for rows in grouped(archive["records"], ["dataset", "n", "d", "k", "seed"]).values():
            if len({r["initial_sha256"] for r in rows}) != 1:
                raise ValueError("Initial centers differ between methods")
    keys = ["dataset", "n", "d", "k"]
    fixed = [
        {**dict(zip(keys, identity)), "methods": summarize(rows, range(3), BACKENDS)}
        for identity, rows in grouped(matrix["records"], keys).items()
    ]
    confirmed = [
        {
            **dict(zip(keys, identity)),
            "validation_selected": selected[identity[0]],
            "methods": summarize(rows, range(300, 310), BACKENDS),
        }
        for identity, rows in grouped(confirmation["records"], keys).items()
    ]
    scale = []
    for identity, rows in grouped(scaling["records"], keys + ["partition"]).items():
        for budget in (25, 100):
            lloyd_rows = [r for r in rows if r["budget"] == budget and r["backend"] != "warp_emeans"]
            scale.append(
                {
                    **dict(zip(keys + ["partition"], identity)),
                    "budget": budget,
                    "methods": summarize(lloyd_rows, range(20, 23), BACKENDS[:4]),
                    "warp_emeans_mse": float(
                        np.mean([r["mse"] for r in rows if r["backend"] == "warp_emeans"])
                    ),
                }
            )
    summary = {"matrix": fixed, "confirmation": confirmed, "scaling": scale}
    Path("results/flash_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    rgb = next(r["methods"] for r in confirmed if r["dataset"] == "bsds_rgb")
    lines = [
        "# Flash-KMeans GPU Lloyd baseline",
        "",
        "This adds the missing external GPU Lloyd comparison to the [Warp baseline study](kmeans-baseline.md). It measures the unmodified [**Flash-KMeans 0.3.1**](https://pypi.org/project/flash-kmeans/0.3.1/) API used by the e-means paper, with shared initial centers and identical Lloyd iteration counts. It does not establish the fastest possible Triton implementation on every GPU.",
        "",
        "Across the seven real feature datasets at the frozen Lloyd budgets, Warp takes less time and establishes the +0.2% loss margin against Flash IEEE on all seven, against Flash FP16 on six, and against Flash TF32 on two. These are per-dataset initialization results without a multiple-comparison adjustment; the datasets are not seven independent application domains.",
        "",
        f"On RGB (100,000 vectors, k=64, 100 Lloyd iterations), Warp takes **{rgb['warp_lloyd']['milliseconds']:.2f} ms**, Flash IEEE **{rgb['flash_ieee']['milliseconds']:.2f} ms**, Flash TF32 **{rgb['flash_tf32']['milliseconds']:.2f} ms**, and Flash FP16 **{rgb['flash_fp16']['milliseconds']:.2f} ms**. The corresponding three-pass Warp/Triton e-means calls take **{rgb['warp_emeans']['milliseconds']:.2f}/{rgb['triton_emeans']['milliseconds']:.2f} ms**. Quality comparisons and every other case follow below.",
        "",
        "RGB meets the specified +0.2% loss margin against Flash IEEE and FP16. Against Flash TF32, Warp's mean loss is higher and the margin is not established. The new seeds also do not establish the earlier RGB quality advantage over either e-means implementation. The old ten-seed result remains archived; initialization variability on a fixed corpus limits how broadly either result can be interpreted.",
        "",
        "In the fixed-budget synthetic matrix, Flash TF32/FP16 beat Warp Lloyd at n=100,000 with d=32/100, and at n=1,000,000 with d=10. Warp Lloyd leads on the remaining measured low-dimensional cases. IEEE Flash is slower than its lower-precision modes, particularly at n=1,000,000. Runtime crossovers and numerical differences both matter; none of these observations establish a universal best k-means implementation.",
        "",
        "## Reference and numerical controls",
        "",
        "[Appendix C](https://arxiv.org/html/2611.09501v1#A3) identifies Flash-KMeans 0.3.1. The [pinned upstream benchmark](https://github.com/facebookresearch/faiss/blob/c8f283867285af547db7a701c1b8dfa89631b3ea/benchs/e_means/paper_xb.py) calls `flash_kmeans.kmeans_triton_impl.batch_kmeans_Euclid` once, supplies initial centers, and uses `max_iters=T, tol=-1`. We use that call and assert the returned iteration count equals T, including on already-converged data. Flash and Warp both retain empty centers. The installed Flash sources are unmodified and SHA256-fingerprinted in each archive.",
        "",
        "| Arm | Resident input / centers | Assignment arithmetic |",
        "| --- | --- | --- |",
        "| Warp Lloyd | FP32 | Direct squared differences in FP32; conservative pruning on eligible large cases |",
        "| Flash TF32 | FP32 | Unmodified default: TF32 dot products, FP32 norms and updates |",
        "| Flash IEEE | FP32 | `TRITON_F32_DEFAULT=ieee`: IEEE FP32 dot products, FP32 norms and updates |",
        "| Flash FP16 | FP16 | Paper-style input and center rounding, FP16 products with FP32 accumulation/updates; means cast to FP16 each iteration |",
        "",
        "All Flash modes form distances using norms and dot products and clamp negative distances to zero. The public FP16 call also stores point norms in FP16. Even IEEE Flash and direct-distance Warp can diverge on close ties and follow different Lloyd trajectories. IEEE means product precision, not bitwise agreement or an exact-real-arithmetic oracle. FP16 receives rounded copies of the shared FP32 data/centers, so it is a separate precision tradeoff. Quality is always evaluated on the original unrounded held-out vectors in FP64.",
        "",
        "The compiled Triton IR is checked for the requested FP32 product precision and archived with IR/PTX hashes. Fresh processes separate TF32 and IEEE to prevent reuse of an in-memory specialization with another precision setting. Explicit `TRITON_DEFAULT_FP_FUSION=1` preserves the installed Triton default. No patches for norm overflow or manual feature padding were needed for this corpus.",
        "",
        "Hardware: NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 2g.48gb, 94 SMs. Warp 1.19.0, Flash-KMeans 0.3.1, PyTorch 2.15.0+cu132, Triton 3.9.0. Flash 0.3.1 does not recognize this GPU in its named architecture tuning tables and selects its conservative split-D fallback. We retain the API's default `use_heuristic=True`, as the pinned paper call does. These results therefore measure that public implementation on this machine; they are not a reproduction of the paper's H100/low-precision e-means comparison or a kernel-only comparison.",
        "",
        "## Timing protocol",
        "",
        "Ordinary training with resident inputs: one untimed full-call warmup, then three synchronized wall-time samples per seed; tables show the median across seed medians. Each configuration and FP32-dot mode uses a fresh process. TF32/FP16/Warp arms alternate order by seed; the IEEE worker follows immediately for the same configuration. Timers include allocation, center initialization from the supplied array, norm computation, all iterations, and Flash's Python loop/convergence synchronization even with `tol=-1`. They exclude input transfer/casting, compilation, cleanup and loss evaluation. No graph replay is mixed into this comparison. The container/GPU can be shared with other agents; samples characterize this run, not an uncontended-machine guarantee. Scaling uses one warmed sample per record for quality evaluation only.",
        "",
        "## Fixed-budget matrix",
        "",
        "Same 15 synthetic configurations and seven real feature datasets as the existing study; seeds 0–2. Lloyd uses 25 full-data iterations and e-means three passes. This table measures elapsed time at those budgets; it does not claim equal quality. Full mean losses and raw samples are archived.",
        "",
    ]
    lines += table(
        ["Dataset", "n", "d", "k", *[label + " ms" for label in LABELS]],
        [
            [
                r["dataset"],
                f"{r['n']:,}",
                r["d"],
                r["k"],
                *[f"{r['methods'][b]['milliseconds']:.2f}" for b in BACKENDS],
            ]
            for r in fixed
        ],
    )
    lines += [
        "",
        "![Synthetic fixed-budget training times](flash-runtime.png)",
        "",
        "## Frozen-budget real-data comparison",
        "",
        "Reuse the previous validation-selected Lloyd budgets without tuning them for Flash. Cases with no validation match run the maximum attempted 200 iterations and remain labeled as failures. The no-match label refers to the original e-means validation target; same-budget comparisons between Lloyd implementations remain valid. All Lloyd arms receive the same budget; e-means keeps three passes. Ten new initialization seeds (300–309) use the same known 16,384 test vectors. These are new initialization replicates, not new scenes, new test data, or an independent validation of scaling laws.",
        "",
    ]
    lines += table(
        ["Dataset", "Lloyd iterations", *[label + " ms" for label in LABELS]],
        [
            [
                r["dataset"],
                r["validation_selected"] or "200 (no validation match)",
                *[f"{r['methods'][b]['milliseconds']:.2f}" for b in BACKENDS],
            ]
            for r in confirmed
        ],
    )
    lines += [
        "",
        "The next table is **Warp Lloyd loss relative to each comparator**, in percent, with a paired 95% bootstrap interval over ten initialization seeds (10,000 resamples). Negative means lower Warp loss. The predeclared noninferiority margin is +0.2%; the upper interval endpoint must be <=+0.2% to establish it. This is a one-sided quality criterion, not symmetric equivalence, and it is not adjusted for multiple comparisons. Intervals describe initialization variability on these fixed datasets only. Values displayed as 0.00% have magnitude below 0.005%; the summary retains full precision.",
        "",
    ]
    comparison_rows = []
    for r in confirmed:
        for backend, label in zip(BACKENDS[1:], LABELS[1:]):
            stats = r["methods"][backend]
            low, high = stats["warp_loss_percent_95"]
            valid = backend.startswith("flash_") or r["validation_selected"] is not None
            comparison_rows.append(
                [
                    r["dataset"],
                    label,
                    f"{stats['warp_loss_percent']:+.2f}%",
                    f"[{low:+.2f}%, {high:+.2f}%]",
                    "yes"
                    if valid and stats["warp_loss_margin_pass"]
                    else ("no validation match" if not valid else "not established"),
                ]
            )
    lines += table(
        ["Dataset", "Comparator", "Warp loss difference", "95% interval", "+0.2% margin established?"],
        comparison_rows,
    )
    lines += [
        "",
        "## Scaling-study coverage",
        "",
        "All 33 previously held-out configurations are also evaluated at 25 and 100 Lloyd iterations, with all three Flash precision modes and Warp. This reuses seeds 20–22 and test rows 16,384–24,575 from the existing evaluator. Warp e-means is rerun with each configuration's original two-pass auto recipe. This is additional baseline coverage on known data, not refitting or confirmation of a new scaling law. Three seeds support descriptive mean losses only; no quality-match claim is based on this table. Scaling timings are archived but are not used for speed claims.",
        "",
        "Entries below give Flash's mean loss difference relative to Warp Lloyd (positive means higher Flash loss). Full losses, e-means comparisons and all timings are in the summary JSON.",
        "",
    ]
    lines += table(
        ["Dataset", "d", "Lloyd iterations", "Flash TF32 loss", "Flash IEEE loss", "Flash FP16 loss"],
        [
            [
                r["dataset"],
                r["d"],
                r["budget"],
                *[
                    f"{100 * (r['methods'][b]['mse'] / r['methods']['warp_lloyd']['mse'] - 1):+.3f}%"
                    for b in BACKENDS[1:4]
                ],
            ]
            for r in scale
        ],
    )
    lines += [
        "",
        "## Reproduce and inspect",
        "",
        "```sh",
        "uv sync --locked --extra research --extra dev",
        "uv run --no-sync python scripts/fetch_reference.py",
        "# Prepare the documented datasets first if data/processed is absent.",
        "uv run --no-sync python scripts/benchmark_flash.py --phase matrix --output results/flash_matrix_new.json",
        "uv run --no-sync python scripts/benchmark_flash.py --phase confirmation --output results/flash_confirmation_new.json",
        "uv run --no-sync python scripts/benchmark_flash.py --phase scaling --repeats 1 --output results/flash_scaling_new.json",
        "# Regenerate the published report from its original archived filenames:",
        "uv run --no-sync python scripts/report_flash.py",
        "```",
        "",
        "Benchmarks refuse to overwrite an existing output. Each archive records the exact source commit, local source hashes, locked dependencies, upstream e-means revision, Flash package hashes, data/initialization hashes, compiled precision evidence and every timing sample. The source commits must remain in Git history. [Dataset preparation and limitations](../datasets.md).",
        "",
        "Evidence: [matrix](../../results/flash_matrix.json), [confirmation](../../results/flash_confirmation.json), [scaling](../../results/flash_scaling.json), [summary](../../results/flash_summary.json), and [validation](../../results/flash_validation.json). Earlier Warp/e-means results remain unchanged. The seven feature datasets reuse images/meshes and are not seven independently sampled application domains.",
        "",
    ]
    Path("docs/experiments/flash-kmeans.md").write_text("\n".join(lines))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), layout="constrained")
    for axis, n in zip(axes, [10000, 100000, 1000000]):
        cases = [r for r in fixed if r["dataset"] == "normal" and r["n"] == n]
        for backend, label in zip(BACKENDS, LABELS):
            axis.plot(
                range(len(cases)),
                [r["methods"][backend]["milliseconds"] for r in cases],
                "o-",
                label=label,
                markersize=3,
            )
        axis.set_xticks(range(len(cases)), [r["d"] for r in cases])
        axis.set(
            xlabel="dimension", ylabel="training time (ms)", yscale="log", title=f"n={n:,}, k={cases[0]['k']}"
        )
        axis.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.07), ncol=6, fontsize=8)
    fig.savefig("docs/experiments/flash-runtime.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
