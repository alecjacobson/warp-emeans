# Research protocol

Progress is tracked in GitHub issues and pull requests. README contains usage and the current measured flagship result only.

## Implementation

Pin and run the authors' Faiss e-means sources. Implement Euclidean FP32 e-means using Warp kernels, including EMA counts, cosine schedule, proportional donor splitting, shuffled passes, and last-pass means. Establish numerical agreement with shared initial centroids and batch order. Benchmark warmed end-to-end training and report compilation/setup separately, with identical inputs, budgets, precision, and final evaluation.

## Scaling study

Follow [Appendices A/B of the paper](https://arxiv.org/html/2611.09501v1#A2): log-space fits to measured step minima and the largest batch within 0.2% of the minimum validation MSE. Use at least three seeds; retain train/validation/test separation; hold out dataset families, dimensions, sizes, and budgets. Test upstream throughput, quality, and current auto policies explicitly. Treat minima at search boundaries as censored, not reliable fit targets. Extend or flag them. Report regret and confidence intervals, not just R². Small pilot sweeps establish machinery and hypotheses, not universal laws.

Cover d=2..10, then 16,32,64,100; vary n, k, T independently (n/k >= 8). Include uniform, anisotropic mixture, and low-intrinsic-dimensional controls alongside documented graphics/robotics data. Distinguish ambient from intrinsic dimension and preserve task-specific metrics and feature weights. Never treat an arbitrary Euclidean encoding of rotations as a validated pose metric.

## Reference provenance

Faiss commit `c8f283867285af547db7a701c1b8dfa89631b3ea`, retrieved 2026-11-17. Current source defaults already choose quality batches at d<=16 and disable revival at d<=2. Those policies and several numerical constants differ from the paper; report code and paper comparisons separately.
