# Low-dimensional graphics and robotics corpus

The following are primary-source uses of k-means or constrained variants. A fast unconstrained Euclidean clusterer does not automatically reproduce the complete application.

| Use case and source | Features | Experiment and limitation |
|---|---|---|
| [Celebi, Improving the Performance of K-Means for Color Quantization](https://arxiv.org/abs/1101.0395) | 3D color | Berkeley sRGB palette clustering. Direct Euclidean k-means use; image-level reconstruction quality is a separate evaluation. |
| [Achanta et al., SLIC](https://www.epfl.ch/labs/ivrl/research/slic-superpixels/) | 5D Lab and pixel coordinates | Berkeley LabXY features. SLIC restricts spatial search and enforces connectivity; our global Euclidean surrogate measures optimization only. |
| [Jiang et al., K-means clustering for efficient and robust registration of multi-view point sets](https://arxiv.org/abs/1710.05193) | 3D XYZ | Stanford surfaces provide low-dimensional geometry. Their registration method alternates clustering with rigid alignment; our benchmark measures clustering of fixed geometry only. |
| [Vlassis et al., Learning the Voronoi Centers of a Mobile Robot’s Configuration Space](https://www.cs.cmu.edu/~motionplanning/papers/sbp_papers/integrated3/vlassis_voronoi.pdf) | 2D free-space positions for a disc robot | Motivates planar robot geometry. Intel laser endpoints are measured obstacle surfaces, not samples of collision-free space; uniform controls represent a different distribution. No navigation success claim. |
| [Papon et al., Voxel Cloud Connectivity Segmentation](https://www.cv-foundation.org/openaccess/content_cvpr_2013/papers/Papon_Voxel_Cloud_Connectivity_2013_CVPR_paper.pdf) | Position, appearance and geometric descriptors, with connectivity constraints | Motivates geometric-feature clustering. Our 6D position + normal datasets are explicitly application-inspired surrogates, not the paper’s feature vector or metric. |

## Prepared datasets

Run `uv run python scripts/build_corpus.py`. Source archives are checksum-verified against [sources.json](../datasets/sources.json). Generated arrays live in ignored `data/processed`; [prepared_manifest.json](../datasets/prepared_manifest.json) records exact scene IDs, sizes, preprocessing and output hashes. Source datasets keep their original terms and attribution; the repository distributes preparation code, not their content.

| Name | d | Training distribution | Validation/test |
|---|---:|---|---|
| bsds_rgb | 3 | Pixels from six original BSDS500 train images, sRGB in [0,1] | Three original validation and three original test images; no image crosses a split |
| bsds_labxy | 5 | Same scenes, D65 Lab/100 plus x,y divided by longest image side | Same original scene split; fixed feature weights independent of k |
| bunny_xyz, drill_xyz | 3 | 1M independent area-uniform samples of each Stanford reconstructed surface, centered and divided by a common bounding-box side | 100k independent samples per split of the same object; hold out the other object for cross-object law evaluation |
| bunny_xyzn, drill_xyzn | 6 | Above XYZ plus 0.1 × oriented face normal | Same independent surface draws; feature weight is part of the metric |
| intel_xy | 2 | Corrected Intel Lab laser endpoints in the first 60% of scans; ranges in (0,30)m | Next 20% and last 20% of scans; common scale/center fitted on training only |
| uniform, gaussian, mixture, manifold | 2–10,16,32,64,100 | Controlled generators with fixed parameters and independent samples | Separate RNG streams; n settings are nested prefixes |

[Berkeley BSDS](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/) provides original scene splits. This first corpus uses lexicographically first images, a declared convenience subset rather than a representative random sample of all 500 scenes. Repeated pixel colors are retained because their frequency defines the quantization objective; identical color vectors across images are not interpreted as scene leakage.

[Stanford Computer Graphics Laboratory](https://graphics.stanford.edu/data/3Dscanrep/) supplies the bunny and drill scans/reconstructions. Surface sampling creates new positions on existing triangles, not independent captured scenes. Geometric normalization uses the fixed source mesh; it is not learned from held-out evaluation samples. Original research-use terms apply.

The [Stachniss Lab laser archive](https://www.ipb.uni-bonn.de/datasets/) credits the Intel Research Lab log to Dirk Haehnel. The original HTTP download works; its HTTPS endpoint refused connections during preparation. The file checksum is pinned. Scan angles follow the legacy FLASER 180-degree convention, from -pi/2 with step pi/beam_count, transformed by the recorded corrected laser pose. Temporal partitions may revisit the same physical place; they test repeated observations, not unseen environments.

The synthetic mixture has twelve unequal-weight anisotropic components. The manifold control embeds two angular variables into a higher-dimensional space with small ambient noise, so ambient dimension can change without increasing the intrinsic signal dimension. Uniform/Gaussian controls increase intrinsic dimension along with ambient dimension. No per-coordinate whitening is applied. Arbitrarily encoding rotations as Euclidean vectors is outside this corpus.
