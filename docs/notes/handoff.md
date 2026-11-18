# Warp e-means research handoff

Workspace: /home/horde/projects/emeans (persistent). Engine: Codex. Ledger run: 21cbfd40-dc1c-49f1-959a-bbc8c9bf5312. Current checkpoint: 2026-11-18 UTC; see the Lloyd baseline section below.

User authorized upstream reproduction, pure Warp implementation, low-dimensional scaling research, public issues/PRs and self-merges in https://github.com/alecjacobson/warp-emeans. Keep README limited to usage and the current flagship result. No subagents requested or used.

## Prior research milestone

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

## Pure Warp Lloyd baseline (Codex, current task)

User requested an optimized pure Warp k-means baseline for the same e-means configurations. Issue #14 tracks implementation. Branch warp-kmeans preserves runtime source 88855d5 and benchmark subset/test revision 5515cd9; retain these commits when merging so archived source fingerprints remain resolvable. Added Kmeans full-batch Lloyd, shared or Warp-random initialization, sharded reductions, empty-center retention, lowest-index FP32 ties, reusable CUDA graphs, and conservative FP64 bounds with active-point compaction for measured large low-dimensional workloads. Experimental losing alternatives remain outside the runtime in scripts/kmeans_variants.py. No new dependencies or native Warp build.

Validation: 72 tests passed on CPU/CUDA; Ruff passed; Compute Sanitizer memcheck passed nine graph/compaction/roundoff tests with zero errors. results/kmeans_validation.json verifies the archived performance/scaling source hashes against retained commits. Report is generated by scripts/report_kmeans.py; main README is limited to usage and current flagship result.

Benchmarks completed: 330 fixed-budget records across the original 15 synthetic configurations plus seven real datasets; 189 validation-selection records; 210 ten-seed confirmation records; 120 timing-replication records; 297 quality records adding warp_lloyd25/100 and rerunning upstream_auto on all 33 scaling holdouts. Original historical studies were preserved. Runtime examples: n=100k, k=256, d=3, Lloyd 25 1.53ms versus Warp e-means 3 25.47ms; d100 Lloyd 25 37.07ms versus Warp 12.87 ms / Triton 8.91 ms. Different budgets do not establish equivalent quality.

RGB flagship: validation selected 100 Lloyd iterations, then seeds 200–209 gave 3.46ms versus Warp e-means 25.28 ms and Triton 36.16 ms; mean loss 1.85% below Triton, paired 95% interval 0.34–3.26% below. Only RGB establishes the predeclared 0.2% margin versus Triton among seven cases. Intel and drill-normal did not meet validation target; LabXY, BunnyXYZ and BunnyXYZ+normals are inconclusive; drillXYZ fails confirmation. Report all cases. Four-case timing replication reproduced RGB (3.41/25.34/35.95ms) but removed elevated later Triton medians; original and repeated samples are both archived and the unexplained noise is disclosed. Quality analysis stays frozen.

All GPU work, documentation and archived results are committed. PR #16 is open: https://github.com/alecjacobson/warp-emeans/pull/16. GitHub rendering validated all five tables and report regeneration leaves no diff. One CPU CI run passed (21 seconds); the PR-triggered run is pending at this checkpoint. Remaining: verify the final checks, self-merge with a merge commit, and remove task scratch files. Consult PR #16 and private Ledger for the final merge receipt; this checkpoint precedes that external action. Follow-up issue #15 records initialization/local-minimum and wide-dimension limitations. Ledger log rejects this already-finished run; remember still delivers. Do not start another run.

Last checkpoint: 2026-11-18T03:14:36.163632+00:00

## Flash-KMeans comparison (Codex, active)

User authorized adding the paper's GPU Lloyd baseline. Prior PR #16 merged at 686d81d; issue #14 closed. Current branch flash-kmeans-baseline, issue #17. Installed pinned flash-kmeans 0.3.1 in the existing research environment. Small and tail dimensions work without source patches; package uses conservative split-D kernels on this unrecognized RTX PRO architecture. FP32 inputs default to TF32 products; benchmark uses separate processes for default TF32 and an IEEE control, plus paper-style FP16. Actual compiled TTIR precision is checked. Added scripts/flash_reference.py and scripts/benchmark_flash.py; optional CUDA tests are running. Final benchmarks, report, README update, PR and merge remain. Existing archives stay unchanged. Source must be committed before final measurements. No new Ledger run: supplied run is already sealed.

Last checkpoint: 2026-11-18T05:00:38.809480+00:00

Flash checkpoint: source e3d54cf0ed0af1293c981b2eac4f16393fc61253 is frozen. 86 full-suite tests pass with IEEE control; 14 Flash checks separately pass with default TF32. Matrix running in exec session2588; output results/flash_matrix.json, scratch log .cache/flash-matrix.log. Do not run GPU tests concurrently. Source committed before all final measurements. Report generator scripts/report_flash.py and changelog17 are uncommitted while matrix runs. Next launch confirmation and scaling sequentially, then generate report, audit provenance, publish and self-merge retaining e3d54cf. Preliminary d100: Warp Lloyd25 about37ms, FlashTF32 about12ms, FlashFP16 about10ms, FlashIEEE about37ms; fixed-budget descriptive quality only.

Last checkpoint: 2026-11-18T05:05:35.837031+00:00

Flash matrix complete:396 records/22cases in results/flash_matrix.json. Source still e3d54cf. Confirmation and then scaling are running sequentially in exec26511 (logs .cache/flash-confirmation.log and .cache/flash-scaling.log). RGB new seeds300–309: Warp100iterations3.43ms, FlashIEEE18.87ms,TF3219.27ms,FP1620.78ms. Quality margin0.2% passes vsIEEE/FP16 but notTF32. New seeds do not establish earlier RGB quality advantage over e-means; README now states this explicitly. scripts/report_flash.py generates full tables/plot/summary once all results exist; .cache/audit-flash.py verifies source and input hashes and writes validation JSON. Remaining: finish sequential benchmarks, generate/audit report, optional memory sanitizer, GitHub render validation, publish/self-merge retaining e3d54cf, then scratch cleanup.

Last checkpoint: 2026-11-18T05:08:46.780111+00:00

## Flash comparison ready to publish

All work is complete locally:396 matrix,420 confirmation,891 scaling records; exact source e3d54cf retained and verified against all code/input hashes. Report docs/experiments/flash-kmeans.md and plot are generated by scripts/report_flash.py; results/flash_summary.json contains full statistics. IEEE margin passes all7real cases, FP16passes6, TF32passes2. Fullsuite86pass; TF32-specific14pass; sanitizer4caseszeroerrors. GitHub Markdown rendered all5tables with consistent columns and README. Current main remains686d81d; intended next action: commit result/docs files, push branch, open PR closing17, verifyCI, then self-merge with merge commit preserving e3d54cf. PR body is .cache/flash-pr.md; follow-up results posted to issue15. Scratch files must be removed after publication; original archives and .cache/reference remain. Ledger remember delivered; log remains unavailable on sealed supplied run.

Last checkpoint: 2026-11-18T05:16:44.740844+00:00
