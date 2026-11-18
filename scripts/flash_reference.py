"""Adapt the paper's pinned, unmodified Flash-KMeans public training call."""

import hashlib
import importlib.metadata
import os
from pathlib import Path


def provenance():
    """Verify the installed reference and fingerprint its complete Python source."""
    import flash_kmeans
    import torch
    import triton

    version = importlib.metadata.version("flash-kmeans")
    if version != "0.3.1":
        raise ValueError(f"Expected flash-kmeans==0.3.1, found {version}")
    root = Path(flash_kmeans.__file__).parent
    return {
        "flash_kmeans": version,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda_runtime": torch.version.cuda,
        "dot_precision": os.environ.get("TRITON_F32_DEFAULT", "tf32"),
        "fp_fusion": os.environ.get("TRITON_DEFAULT_FP_FUSION"),
        "package_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*.py"))
        },
    }


def train(points, initial, iterations):
    """Run all Lloyd iterations with explicit centers and retain empty clusters."""
    from flash_kmeans.kmeans_triton_impl import batch_kmeans_Euclid

    # Import the exact implementation, avoiding the package's PyTorch fallback.
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if points.ndim != 2 or initial.ndim != 2 or points.shape[1] != initial.shape[1]:
        raise ValueError("Expected points (n,d) and centers (k,d) with matching d")
    if not 0 < len(initial) <= len(points):
        raise ValueError("Expected 0 < k <= n")
    if points.dtype != initial.dtype or points.device != initial.device:
        raise ValueError("Points and centers must share dtype and device")
    labels, centers, actual_iterations = batch_kmeans_Euclid(
        points[None],
        len(initial),
        max_iters=iterations,
        tol=-1,
        init_centroids=initial[None].clone(),
    )
    if actual_iterations != iterations:
        raise RuntimeError(f"Flash-KMeans stopped after {actual_iterations}, expected {iterations}")
    return centers[0], labels[0]


def compiled_assignment_evidence():
    """Read the actual compiled dot precision, including the architecture fallback."""
    from flash_kmeans.assign_euclid_triton import _euclid_assign_kernel, _euclid_assign_kernel_split_d

    evidence = []
    for function in (_euclid_assign_kernel, _euclid_assign_kernel_split_d):
        for cache in function.device_caches.values():
            for kernel in cache[0].values():
                ir = kernel.asm["ttir"]
                evidence.append(
                    {
                        "kernel": function.__name__,
                        "dot_operations": [line.strip() for line in ir.splitlines() if "tt.dot " in line],
                        "ttir_sha256": hashlib.sha256(ir.encode()).hexdigest(),
                        "ptx_sha256": hashlib.sha256(kernel.asm["ptx"].encode()).hexdigest(),
                    }
                )
    if not evidence:
        raise RuntimeError("No compiled Flash assignment kernel found; cannot verify execution path")
    return evidence
