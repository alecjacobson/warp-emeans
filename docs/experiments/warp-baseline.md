# Warp baseline and cooperative assignment

Measured 2026-11-17; RTX PRO 6000 Blackwell MIG 2g.48gb, Warp 1.19.0, PyTorch 2.15.0+cu132. Upstream Faiss c8f283867285af547db7a701c1b8dfa89631b3ea with explicit Triton. FP32 standard-normal inputs and same supplied initial centroids. Three seeds, one warmup per configuration then one timed run per seed; medians below. Transfers and FP64 validation evaluation excluded from timing. Ordinary Warp includes allocation and finite-input checks; Triton uses its normal train API. All final timing sweeps ran sequentially with no other GPU experiment from this session. The GPU can be shared with other users; this remains single-machine evidence. Model/graph destruction and garbage collection are outside the timers.

Both implementations use the same batch sizes, learning-rate schedules and three training passes. Their different random sequences can produce different final clusterings, so these runtime comparisons do not establish equal clustering quality. Dimension d is the number of features per point. The largest median relative MSE difference in this matrix is reported below. Raw files retain every seed, cold call, training settings and graph setup time.

| n | k | d | Triton ms | Warp train ms | Warp graph replay ms | Graph setup ms | Train speedup | Warp/Triton validation MSE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 32 | 2 | 25.97 | 15.53 | 3.70 | 29.97 | 1.67× | 1.00794 |
| 10,000 | 32 | 5 | 28.32 | 20.63 | 8.68 | 39.94 | 1.37× | 1.00380 |
| 10,000 | 32 | 10 | 29.64 | 20.41 | 9.60 | 39.86 | 1.45× | 1.00001 |
| 100,000 | 256 | 2 | 35.43 | 18.89 | 4.74 | 36.58 | 1.88× | 0.99406 |
| 100,000 | 256 | 3 | 36.12 | 24.92 | 10.43 | 48.28 | 1.45× | 1.00878 |
| 100,000 | 256 | 4 | 36.51 | 24.99 | 11.49 | 48.66 | 1.46× | 0.99429 |
| 100,000 | 256 | 5 | 36.86 | 25.00 | 12.31 | 48.81 | 1.47× | 1.00140 |
| 100,000 | 256 | 6 | 37.36 | 25.18 | 12.69 | 48.87 | 1.48× | 1.00500 |
| 100,000 | 256 | 8 | 36.79 | 25.42 | 13.90 | 48.86 | 1.45× | 0.99844 |
| 100,000 | 256 | 10 | 36.92 | 25.09 | 14.40 | 48.68 | 1.47× | 1.00003 |
| 100,000 | 256 | 32 | 7.63 | 4.11 | 3.86 | 7.33 | 1.85× | 1.00004 |
| 100,000 | 256 | 100 | 8.90 | 13.03 | 12.99 | 16.23 | 0.68× | 1.00017 |
| 1,000,000 | 1024 | 2 | 54.46 | 23.30 | 10.57 | 44.86 | 2.34× | 1.00199 |
| 1,000,000 | 1024 | 5 | 54.76 | 30.85 | 23.45 | 60.19 | 1.78× | 1.00336 |
| 1,000,000 | 1024 | 10 | 54.42 | 30.97 | 35.00 | 60.89 | 1.76× | 0.99935 |

Maximum median loss ratio: 1.00878. Three seeds are too few for a strong equivalence claim.

The original one-thread-per-point kernel underutilized the GPU on small batches. The current kernel shares each search among 32 threads and stores center coordinates so neighboring lanes read neighboring centers. The exposed center shape remains (k,d), with noncontiguous CUDA strides. At d=32 this layout brings ordinary training to 4.11ms against Triton’s 7.63ms. At d=100 the scalar distance loop remains slower (13.03ms versus 8.90ms); [issue #6](https://github.com/alecjacobson/warp-emeans/issues/6) tracks this gap and broader equal-loss coverage. Graph replay is not always faster than ordinary training; preparation is reported separately.

Final raw results: [small](../../results/benchmark_small_final.json), [main dimension sweep](../../results/benchmark_transposed.json), [large](../../results/benchmark_large_final.json). Reproduce with:

```sh
uv run python scripts/benchmark.py --dimensions 2 5 10 --n 10000 --k 32 --output results/benchmark_small_final.json
uv run python scripts/benchmark.py --output results/benchmark_transposed.json
uv run python scripts/benchmark.py --dimensions 2 5 10 --n 1000000 --k 1024 --output results/benchmark_large_final.json
uv run python scripts/summarize_benchmark.py
```

Earlier scalar/cooperative/small/large and `benchmark_transposed_before_cleanup.json` files are retained as exploratory records. Their ordinary-training timers could include destruction of a previous prepared model. They are excluded from the final table and speed claims; see [issue #9](https://github.com/alecjacobson/warp-emeans/issues/9).

Validation: 37 CPU/GPU tests passed, including upstream shared-order parity in d=2,3,5,10,100, tails/fractional epochs, direct assignment, ties/empty centers, revival mass/donor invariants, subnormal split separation, graph reset/runtime step changes, dataset splits and interval censoring. Compute Sanitizer memcheck passed ten focused graph/revival/subnormal cases with zero errors. [Independent real-data loss-margin confirmation](equal-loss.md) reports all five cases separately.
