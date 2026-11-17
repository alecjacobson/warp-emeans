# Warp baseline and cooperative assignment

Measured 2026-11-17; RTX PRO 6000 Blackwell MIG 2g.48gb, Warp 1.19.0, PyTorch 2.15.0+cu132. Upstream Faiss c8f283867285af547db7a701c1b8dfa89631b3ea with explicit Triton. FP32 standard-normal inputs and same supplied initial centroids. Three seeds, one warmup per configuration then one timed run per seed; medians below. Transfers and FP64 validation evaluation excluded from timing. Ordinary Warp includes allocation and finite-input checks; Triton uses its normal train API. The GPU is shared with other possible users; this remains single-machine evidence. The cooperative sweep was repeated after initial kernel testing.

Different RNG streams mean this is a recipe-matched comparison, not strict equal-loss timing. The largest median relative MSE difference in this matrix is reported below. Raw files retain every seed, cold call, recipe and graph setup time.

| n | k | d | Triton ms | Warp train ms | Warp graph replay ms | Graph setup ms | Train speedup | Warp/Triton validation MSE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 32 | 2 | 24.57 | 14.28 | 2.91 | 27.77 | 1.72× | 1.00794 |
| 10,000 | 32 | 5 | 27.03 | 19.28 | 6.69 | 37.56 | 1.40× | 1.00380 |
| 10,000 | 32 | 10 | 28.17 | 19.28 | 7.58 | 37.90 | 1.46× | 1.00001 |
| 100,000 | 256 | 2 | 35.04 | 17.64 | 4.19 | 34.69 | 1.99× | 0.99406 |
| 100,000 | 256 | 3 | 35.00 | 23.58 | 10.16 | 47.53 | 1.48× | 1.00878 |
| 100,000 | 256 | 4 | 34.75 | 23.78 | 10.00 | 47.91 | 1.46× | 0.99429 |
| 100,000 | 256 | 5 | 35.57 | 23.81 | 10.57 | 47.87 | 1.49× | 1.00140 |
| 100,000 | 256 | 6 | 35.78 | 23.85 | 10.96 | 47.73 | 1.50× | 1.00500 |
| 100,000 | 256 | 8 | 35.27 | 24.00 | 11.72 | 48.30 | 1.47× | 0.99844 |
| 100,000 | 256 | 10 | 35.70 | 23.74 | 12.25 | 47.91 | 1.50× | 1.00003 |
| 100,000 | 256 | 32 | 6.55 | 14.40 | 14.57 | 17.80 | 0.46× | 1.00004 |
| 100,000 | 256 | 100 | 8.09 | 18.97 | 18.93 | 22.28 | 0.43× | 1.00027 |
| 1,000,000 | 1024 | 2 | 54.03 | 22.58 | 9.69 | 44.22 | 2.39× | 1.00199 |
| 1,000,000 | 1024 | 5 | 54.36 | 29.66 | 20.58 | 59.21 | 1.83× | 1.00260 |
| 1,000,000 | 1024 | 10 | 54.27 | 30.31 | 34.47 | 60.70 | 1.79× | 1.00086 |

Maximum median loss ratio: 1.00878. Three seeds are too few for a strong equivalence claim.

The original one-thread-per-point kernel underutilized the GPU on small batches. Sharing each search among 32 threads cuts d=10 graph execution from about 41ms to 13ms at n=100k,k=256. Above d=16 the upstream batch policy changes; Triton’s matrix-based assignment wins in the larger-batch, wider-dimensional regime. The pure Warp kernel retains that regression openly; [issue #6](https://github.com/alecjacobson/warp-emeans/issues/6) tracks it.

Raw results: [scalar baseline](../../results/benchmark_scalar.json), [cooperative](../../results/benchmark_cooperative.json), [small](../../results/benchmark_small.json), [large](../../results/benchmark_large.json). Reproduce using scripts/benchmark.py and its --dimensions, --n, --k, --output arguments.

Validation: 24 CPU/GPU tests passed, including upstream shared-order parity in d=2,3,5,10,100, tails/fractional epochs, direct assignment, ties/empty centers, revival mass/donor invariants, graph reset, and invalid-input rejection.
