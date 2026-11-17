"""Report paired uncertainty for the frozen law candidates on 30 fresh seeds."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

rows = [json.loads(line) for line in Path("results/scaling_real_confirmation.jsonl").read_text().splitlines()]
groups = defaultdict(lambda: defaultdict(dict))
for row in rows:
    if row["status"] != "ok":
        raise ValueError(f"Incomplete confirmatory record: {row}")
    groups[row["dataset"]][row["model"]][row["seed"]] = row["test_mse"]
rng = np.random.default_rng(198)
records = []
lines = [
    "## Thirty-seed real-data confirmation",
    "",
    "Both larger-study laws were frozen before this confirmation. Thirty fresh seeds (30–59) and 4,096 test points beginning at offset 24,576 are disjoint from the earlier reported evaluation. The tables compare mean test MSE with upstream auto using paired seed bootstrap intervals (10,000 resamples). They quantify initialization uncertainty for these fixed seven feature datasets. A law is not accepted merely because its aggregate median looks good.",
    "",
    "| Dataset | No-dimension / auto [95% interval] | Dimension / auto [95% interval] |",
    "|---|---:|---:|",
]
for dataset, methods in groups.items():
    seeds = sorted(methods["upstream_auto"])
    if len(seeds) != 30:
        raise ValueError("Expected 30 seeds")
    baseline = np.array([methods["upstream_auto"][seed] for seed in seeds])
    ids = rng.integers(0, 30, size=(10000, 30))
    fields = []
    for method in ("interval_no_dimension", "interval_dimension"):
        candidate = np.array([methods[method][seed] for seed in seeds])
        ratio = candidate.mean() / baseline.mean()
        interval = np.quantile(candidate[ids].mean(axis=1) / baseline[ids].mean(axis=1), [0.025, 0.975])
        records.append(
            {
                "dataset": dataset,
                "model": method,
                "ratio_to_upstream_auto": float(ratio),
                "ratio_95": interval.tolist(),
            }
        )
        fields.append(f"{ratio:.4f} [{interval[0]:.4f}, {interval[1]:.4f}]")
    lines.append("| " + dataset + " | " + " | ".join(fields) + " |")
lines += [
    "",
    "These results do not justify replacing upstream defaults with either candidate across all low-dimensional applications. Dataset geometry and distribution shift remain material. The next fit should include representative surface/image/robotics families with additional independent source scenes and distribution descriptors, while preserving genuinely new evaluation data. [Raw confirmation](../../results/scaling_real_confirmation.jsonl) and [summary](../../results/scaling_real_confirmation_summary.json).",
    "",
]
Path("results/scaling_real_confirmation_summary.json").write_text(json.dumps(records, indent=2) + "\n")
p = Path("docs/experiments/scaling.md")
text = p.read_text()
marker = "## Thirty-seed real-data confirmation"
if marker in text:
    text = text[: text.index(marker)].rstrip() + "\n"
p.write_text(text + "\n" + "\n".join(lines))
print(json.dumps(records, indent=2))
