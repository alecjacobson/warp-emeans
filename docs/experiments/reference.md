# Pinned reference reproduction

Faiss revision `c8f283867285af547db7a701c1b8dfa89631b3ea`. Run 2026-11-17 on NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 2g.48gb. NumPy 2.5.5, PyTorch 2.15.0+cu132, Faiss 1.16.0. Four CPU threads. Inputs: 100,000 independent standard normal vectors, k=256, T=3, FP32, supplied initial centroids; validation uses 4,096 independent vectors and SciPy FP64 squared distances. Each row reports the second call after one warmup. Transfers and evaluation are outside the timer. CPU/GPU use different random permutations, so loss differences are not a parity test.

| d | NumPy seconds | Triton seconds | NumPy validation MSE | Triton validation MSE |
|---|---:|---:|---:|---:|
| 2 | 0.137418 | 0.034777 | 0.016818143 | 0.017129447 |
| 3 | 0.128723 | 0.035158 | 0.13324486 | 0.13232774 |
| 5 | 0.132385 | 0.035584 | 0.79732065 | 0.79900342 |
| 10 | 0.134750 | 0.038986 | 4.1508539 | 4.1184768 |
| 100 | 0.171914 | 0.008414 | 91.868597 | 91.868979 |

All returned centroids were finite and training labels in range. This confirms the reference runs here; two calls are insufficient for a performance claim. The raw file records cold calls, recipes and split counts. CuTe targets SM90; Triton is upstream’s supported backend for this SM120 GPU.

Reproduce: `uv run python scripts/run_reference.py`. [Raw results](../../results/reference.json).

Current code uses initial mass 2^-16, split epsilon 1e-4, quality batches at d<=16 and no revival at d<=2. These differ from Algorithm 1.
