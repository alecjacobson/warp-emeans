"""Measure complete Lloyd iterations for candidate assignment layouts."""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import numpy as np
import warp as wp
from scipy.spatial.distance import cdist

from warp_emeans import lloyd_kernels

parser = argparse.ArgumentParser()
parser.add_argument("--dimensions", nargs="+", type=int, default=[2, 3, 5, 10, 32, 100])
parser.add_argument("--n", type=int, default=100000)
parser.add_argument("--k", type=int, default=256)
parser.add_argument("--output", default="results/kmeans_tuning_initial.json")
parser.add_argument("--variants", nargs="+", default=["scalar", "cooperative", "tile32x32", "tile8x64"])
args = parser.parse_args()
wp.init()
wp.config.quiet = True
records = []
for d in args.dimensions:
    rng = np.random.default_rng(481 + d)
    x = rng.normal(size=(args.n, d)).astype(np.float32)
    initial = x[rng.choice(args.n, args.k, replace=False)]
    points = wp.array(x, device="cuda")
    for variant in args.variants:
        centers = (
            wp.empty((args.k, d), dtype=float, device="cuda")
            if variant == "scalar"
            else wp.empty((d, args.k), dtype=float, device="cuda").transpose()
        )
        initial_wp = wp.array(initial, device="cuda")
        wp.copy(centers, initial_wp)
        sums = wp.zeros((64, args.k, d), dtype=float, device="cuda")
        counts = wp.zeros((64, args.k), dtype=int, device="cuda")
        labels = wp.empty(args.n, dtype=int, device="cuda")
        distances = wp.empty(args.n, dtype=float, device="cuda")
        if variant == "scalar":
            kernel = lloyd_kernels.scalar_assignment_kernel(d)
        elif variant == "cooperative":
            kernel = lloyd_kernels.cooperative_assignment_kernel(d)
        else:
            point_tile, center_tile = map(int, variant.removeprefix("tile").split("x"))
            kernel = lloyd_kernels.tiled_assignment_kernel(d, point_tile, center_tile)

        def assign():
            if variant.startswith("tile"):
                wp.launch_tiled(kernel, dim=(args.n + point_tile - 1) // point_tile, inputs=[points, centers, labels, distances], block_dim=128)
                wp.launch(lloyd_kernels.accumulate_kernel, (args.n, d), [points, labels, sums, counts])
            else:
                wp.launch(kernel, (args.n, 32) if variant == "cooperative" else args.n,
                          [points, centers, sums, counts, labels, distances, 1],
                          block_dim=32 if variant == "cooperative" else 128)

        assign()
        expected = cdist(x[:511], initial, "sqeuclidean")
        np.testing.assert_array_equal(labels.numpy()[:511], expected.argmin(axis=1))
        np.testing.assert_allclose(distances.numpy()[:511], expected.min(axis=1), atol=1e-5, rtol=2e-6)

        def execute():
            wp.copy(centers, initial_wp)
            for _ in range(25):
                sums.zero_()
                counts.zero_()
                assign()
                wp.launch(lloyd_kernels.update_kernel, centers.shape, [centers, sums, counts])

        execute()
        wp.synchronize()
        with wp.ScopedCapture() as captured:
            execute()
        samples = []
        for _ in range(5):
            wp.synchronize()
            start = time.perf_counter()
            wp.capture_launch(captured.graph)
            wp.synchronize()
            samples.append(time.perf_counter() - start)
        record = dict(n=args.n, k=args.k, d=d, variant=variant, seconds=statistics.median(samples), samples=samples)
        records.append(record)
        print(json.dumps(record), flush=True)
        del captured
    Path(args.output).write_text(json.dumps({
        "records": records, "iterations": 25, "timing": "Prepared graph replay; includes reset and full assignment/accumulation/mean updates; setup excluded",
        "warp": wp.__version__, "gpu": wp.get_device("cuda").name,
        "code_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), Path("src/warp_emeans/lloyd_kernels.py")]},
    }, indent=2) + "\n")
