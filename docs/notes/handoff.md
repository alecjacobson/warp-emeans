# Warp e-means research handoff

Workspace: /home/horde/projects/emeans (persistent). Engine: Codex. Ledger run: 21cbfd40-dc1c-49f1-959a-bbc8c9bf5312. Last checkpoint: 2026-11-17, final research PR preparation.

User authorized upstream reproduction, pure Warp implementation, low-dimensional scaling research, public issues/PRs and self-merges in https://github.com/alecjacobson/warp-emeans. Keep README limited to usage and the current flagship result. No subagents requested or used.

## Delivered state

PR #5 reproduced pinned NumPy/Triton e-means; PR #7 introduced the pure Warp implementation. The corpus-and-scaling branch adds the full corpus, larger study, confirmation, corrected final timing, transposed CUDA center storage and subnormal revival fix. Consult GitHub for its final merge status. Preserve commit 682326e4cf238ae37587c1fd87a4d2b9fe9f6ae7 with a merge commit: every source fingerprint for the canonical sweep was verified against it. Do not squash this branch and lose that provenance.

The local uv research/dev environment works with Warp 1.19.0, PyTorch 2.15.0+cu132, and the RTX PRO 6000 Blackwell MIG 2g.48gb. Use the installed wheel; no native Warp build is needed. Reference sources are checksum-fetched into ignored .cache/reference from Faiss c8f283867285af547db7a701c1b8dfa89631b3ea. Keep the shared default Warp cache intact.

Pure Warp implements shuffling, cooperative assignment, EMA, cosine scheduling, weighted donor revival and last-pass means, plus reusable CUDA graphs with runtime alpha changes. CUDA centers have shape (k,d) with transposed strides; consumers must honor strides. Different RNG streams prevent exact seed-wise agreement with upstream. Shared-order no-revival parity is tested. Explicit float/int/bool casts in kernels are required by Warp; do not apply Ruff unsafe cast-removal fixes.

Final validation: 37 tests passed, Ruff passed; Compute Sanitizer memcheck passed ten graph/revival/subnormal cases with zero errors. Evidence and source hashes: results/final_validation.json. The subnormal regression failed before the fix (four cases); using array += was also wrong because CUDA atomic addition flushes subnormals. Explicit scalar addition and assignment preserves nextafter behavior.

## Results and interpretation

Final ordinary-training matrix: Warp 1.37–2.34x faster in measured d2–10 cases (n10k–1M,k32–1024), and 1.85x at d32. At d100 Warp is still slower (13.03ms vs 8.90ms). Final files are benchmark_small_final.json, benchmark_transposed.json, benchmark_large_final.json. Earlier ordinary timers could include previous CUDA graph cleanup; retain them as exploratory data only, never promote their speed claims.

The independent real-data test froze epoch budgets on validation, then used ten fresh paired seeds and a predeclared 0.2% loss margin. Bunny XYZ passes narrowly at 1.44x speed (25.05 vs 36.06ms), loss ratio 0.99869, paired 95% interval [0.99474,1.00195]. LabXY exceeds the margin; Intel, RGB and Bunny XYZ+normals are inconclusive. Report all five, and do not claim a universal equal-loss speedup. See docs/experiments/equal-loss.md.

Corpus preparation supports seven real feature datasets: Intel laser XY; BSDS RGB/LabXY; Stanford bunny/drill XYZ and XYZ+normals. All source hashes, preprocessing and splits are documented. Independent surface samples are not independent scanned objects; SLIC/VCCS encodings are surrogates without full application constraints. Source archives can be downloaded again; ignored data/processed contains the prepared arrays. Keep irreplaceable results in the project.

Scaling: 14,040-run pilot, 15,912-run larger study (97 configurations), 960-run revival ablation, 693 fresh-seed held-out evaluations and 840 real confirmation records. Dimensions cover 2–10,16,32,64,100. Fit data are uniform and anisotropic mixture at n100k/1M,k64/256,T1/3. Preserve conditional alpha/batch censoring: 226 alpha boundaries and 17 lower-boundary batch minima in the point fit. Interval candidates differ explicitly from the paper fitting method. No universal correct law is established; upstream defaults remain unchanged.

Frozen candidate laws both improve mean test MSE on six of seven real feature datasets in 30-seed confirmation; LabXY is inconclusive. The dimension candidate improves about 0.6–4.4% on those six. These are per-dataset initialization intervals, not population/domain uncertainty. Full coefficients, bootstrap intervals, source provenance, plots, failure cases and fresh-output reproduction commands are in docs/experiments/scaling.md. Evaluation timing is not performance evidence. The canonical sweep predates final layout/subnormal changes; its exact source is preserved as above. New resume guards reject source/protocol mismatch rather than overwriting original metadata.

## Remaining work

- #6: optimize d100 and establish equal-loss speed across more workloads; validate any new fast path explicitly.
- #4: refine censored valleys and study intrinsic dimension/covariance/imbalance/revival effects before claiming general scaling.
- #10: independently sampled source scenes, acquired clouds and robot environments, plus repeated synthetic distribution parameters and genuinely new held-out evaluations.
- #2/#3 implementation/corpus milestones and #8/#9 fixes are intended to close with this research PR; verify GitHub status.

Before a new run, inspect current Git state and GitHub issues. Resume only with matching provenance, otherwise choose a fresh output directory. Do not rerun the entire completed matrix just to regenerate plots: report_scaling.py then report_real_confirmation.py read archived evidence. Final task completion/merge outcome is also recorded in private Ledger for this run.
