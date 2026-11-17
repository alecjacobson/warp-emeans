"""Regenerate the numerical table from measured benchmark records."""

import json
import statistics
from pathlib import Path

path = Path("docs/experiments/warp-baseline.md")
text = path.read_text()
start = text.index("| n |")
end = text.index("\nThe original", start)
lines = [
    "| n | k | d | Triton ms | Warp train ms | Warp graph replay ms | Graph setup ms | Train speedup | Warp/Triton validation MSE |",
    "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
ratios = []
for name in ("benchmark_small", "benchmark_cooperative", "benchmark_large"):
    records = json.loads(Path(f"results/{name}.json").read_text())["records"]
    for d in sorted({r["d"] for r in records}):
        med = {}
        for mode in ("triton", "warp", "warp_graph"):
            selected = [r for r in records if r["mode"] == mode and r["d"] == d and r["repeat"] == 1]
            med[mode] = {
                key: statistics.median(r[key] for r in selected)
                for key in ("seconds", "setup_seconds", "validation_mse_fp64")
            }
        ratio = med["warp"]["validation_mse_fp64"] / med["triton"]["validation_mse_fp64"]
        ratios.append(ratio)
        lines.append(
            f"| {selected[0]['n']:,} | {selected[0]['k']} | {d} | {med['triton']['seconds'] * 1000:.2f} | {med['warp']['seconds'] * 1000:.2f} | {med['warp_graph']['seconds'] * 1000:.2f} | {med['warp_graph']['setup_seconds'] * 1000:.2f} | {med['triton']['seconds'] / med['warp']['seconds']:.2f}× | {ratio:.5f} |"
        )
lines += [
    "",
    f"Maximum median loss ratio: {max(ratios):.5f}. Three seeds are too few for a strong equivalence claim.",
    "",
]
text = text[:start] + "\n".join(lines) + text[end:]
text = text.replace(
    "No other experiment was intentionally using the GPU during the timed runs. This remains single-machine evidence.",
    "The GPU is shared with other possible users; this remains single-machine evidence. The cooperative sweep was repeated after initial kernel testing.",
)
path.write_text(text)
