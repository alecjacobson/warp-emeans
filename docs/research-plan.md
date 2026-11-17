# Research protocol

Progress is tracked in GitHub issues and pull requests. README contains usage and the current measured flagship result only.

## Implementation

Pin and run the authors' Faiss e-means sources. Implement Euclidean FP32 e-means using Warp kernels, including EMA counts, cosine schedule, proportional donor splitting, shuffled passes, and last-pass means. Establish numerical agreement with shared initial centroids and batch order. Benchmark warmed end-to-end training and report compilation/setup separately, with identical inputs, budgets, precision, and final evaluation.

## Scaling study

Follow [Appendices A/B of the paper](https://arxiv.org/html/2611.09501v1#A2): log-space fits to measured step minima and the largest batch within 0.2% of the minimum validation MSE. Use at least three seeds; retain train/validation/test separation; hold out dataset families, dimensions, sizes, and budgets. Test upstream throughput, quality, and current auto policies explicitly. Treat minima at search boundaries as censored, not reliable fit targets. Extend or flag them. Report regret and confidence intervals, not just R². Small pilot sweeps establish machinery and hypotheses, not universal laws.

Cover d=2..10, then 16,32,64,100; vary n, k, T independently (n/k >= 8). Include uniform, anisotropic mixture, and low-intrinsic-dimensional controls alongside documented graphics/robotics data. Distinguish ambient from intrinsic dimension and preserve task-specific metrics and feature weights. Never treat an arbitrary Euclidean encoding of rotations as a validated pose metric.

## Reference provenance

Faiss commit `c8f283867285af547db7a701c1b8dfa89631b3ea`, retrieved 2026-11-17. Current source defaults already choose quality batches at d<=16 and disable revival at d<=2. Those policies and several numerical constants differ from the paper; report code and paper comparisons separately.

## Executed pilot design

The first factorial fit uses uniform and unequal anisotropic mixtures at d=2,3,5,10; n=32,768/131,072; k=32/128; T=1/3. Transfer grids hold out d=4,6,7,8,9,16,32,64,100, plus Gaussian/manifold families and all real application datasets. Each problem sweeps B=128,512,2048,8192,n and a geometric alpha ladder from 1e-4 to 1, with seeds 0,1,2. Validation is a separate 4,096-row sample, evaluated by FP64 direct squared distances. The smaller n and coarser ladder intentionally make this a pilot; they do not reproduce the paper's 100k–100M scale or 0.2%-regret precision.

Models are fitted only on the designated fit configurations. Repeated seeds are aggregated before selecting minima. Alpha minima at either boundary are counted and excluded from ordinary log-regression; batch knees reaching the largest rung are right-censored. Both no-dimension and log-dimension laws are compared, with problem-level bootstrap uncertainty. Validation-oracle settings and candidate laws are subsequently rerun on fresh seeds and the untouched test split. Validation-derived oracle means the best measured validation setting, not an oracle chosen on test loss.

## Larger study and censored fitting

The extended matrix uses n=100,000/1,000,000, k=64/256, T=1/3, d=2,3,5,10 on uniform and mixture distributions. Unseen dimensions, Gaussian/manifold families and real datasets use n=100,000,k=128,T=2 (Intel uses n=65,536 because only 95,172 training points exist). B candidates are 32,128,512,2048,8192,32768,n; alpha is 1e-5,1e-4,1e-3,.01,.03,.1,.3,1. Runs above 32,768 updates are explicitly excluded from the grid, censoring its small-B side at the largest n/T. Three seeds per cell remain a limitation. This changes the original pilot size range and extends its boundaries; both datasets are retained separately.

The point-minimum fit is retained as an Appendix-B-style baseline. An additional candidate fits contiguous 0.2%-loss valleys as log-alpha intervals, so flat curves and boundary minima are not invented point observations. Batch knees use the same tolerance and interpolation in log B. If the minimum is at the smallest batch, the observed knee supplies only an upper bound under a single-valley assumption. A weak ridge of 1e-4 toward the upstream quality law regularizes unidentified directions. This is an explicitly different fitting method, not a claim to reproduce the paper's exact procedure. Bootstrap intervals describe sensitivity within this finite grid; they are not population uncertainty across arbitrary graphics or robotics datasets.

Final comparisons report both dimension and no-dimension candidates without selecting a winner on test MSE. Held-out loss is relative to a validation-selected grid setting rerun on fresh seeds; that setting can lose on test. An independent equal-loss speed protocol selects the smallest Warp epoch budget from 1,2,3,5,10,25 whose three-seed validation MSE is within 0.2% of upstream Triton at three epochs, then freezes the choice. Confirmatory timing uses ten fresh seeds, GPU-resident inputs, warm kernels, ordinary train calls, independent FP64 test loss, and paired bootstrap intervals. All five real-data cases are reported, including any failed quality match. Graph replay is excluded from this end-to-end speed claim.

## Thirty-seed confirmation and timing correction

Freeze both candidate coefficient sets before evaluating seeds 30–59 on seven real feature datasets. Use 4,096 test points beginning at offset 24,576, disjoint from earlier law evaluation. Report paired bootstrap intervals for each dataset and candidate; no aggregate success claim should hide inconclusive LabXY results. Confidence intervals describe initialization variation, not coverage across independent domains.

Final timing runs explicitly collect garbage and destroy the preceding model outside the timer. The earlier ordinary-training records could include CUDA graph cleanup and are retained only as exploratory evidence. Final matrix filenames end in `_final`, except the main dimension matrix `benchmark_transposed.json`. The final equal-loss confirmation uses the fixed timing harness, ten fresh initialization seeds and frozen validation-selected budgets.
