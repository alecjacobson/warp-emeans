"""Prepare a checksum-verified corpus without redistributing source datasets."""

import gzip
import hashlib
import io
import json
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def fetch_sources():
    """Fetch and verify the explicitly listed research data archives."""
    manifest = json.loads((ROOT / "datasets/sources.json").read_text())
    raw = ROOT / "data/raw"
    raw.mkdir(parents=True, exist_ok=True)
    for name, metadata in manifest.items():
        path = raw / name
        if not path.exists():
            temporary = path.with_suffix(path.suffix + ".partial")
            with urllib.request.urlopen(metadata["url"], timeout=60) as response, temporary.open("wb") as out:
                while chunk := response.read(1 << 20):
                    out.write(chunk)
            temporary.replace(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
            raise ValueError(f"Checksum mismatch for {name}")
    return raw


def rgb_to_lab(rgb):
    """Convert sRGB in [0,1] to CIELAB with a D65 white point."""
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = (
        linear
        @ np.array(
            [
                [0.4124564, 0.3575761, 0.1804375],
                [0.2126729, 0.7151522, 0.0721750],
                [0.0193339, 0.1191920, 0.9503041],
            ]
        ).T
    )
    xyz /= [0.95047, 1, 1.08883]
    delta = 6 / 29
    value = np.where(xyz > delta**3, np.cbrt(xyz), xyz / (3 * delta**2) + 4 / 29)
    return np.stack(
        (
            116 * value[..., 1] - 16,
            500 * (value[..., 0] - value[..., 1]),
            200 * (value[..., 1] - value[..., 2]),
        ),
        axis=-1,
    )


def read_mesh(archive, member):
    """Read one ASCII triangle PLY from a tar archive without extraction."""
    with archive.extractfile(member) as file:
        vertex_count = face_count = 0
        while True:
            line = file.readline().decode("ascii").strip()
            if line.startswith("format") and line != "format ascii 1.0":
                raise ValueError("Expected ASCII PLY")
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
            if line.startswith("element face"):
                face_count = int(line.split()[-1])
            if line == "end_header":
                break
            if not line:
                raise ValueError("Truncated PLY header")
        vertices = np.array([[float(v) for v in file.readline().split()[:3]] for _ in range(vertex_count)])
        faces = []
        for _ in range(face_count):
            values = [int(v) for v in file.readline().split()]
            if values[0] != 3:
                raise ValueError("Expected triangular faces")
            faces.append(values[1:4])
    return vertices, np.asarray(faces)


def build():
    """Write deterministic train/validation/test arrays and preparation metadata."""
    raw = fetch_sources()
    destination = ROOT / "data/processed"
    destination.mkdir(parents=True, exist_ok=True)
    metadata = {}

    def save(name, splits, description, sources):
        info = {"description": description, "sources": sources, "splits": {}}
        for split, values in splits.items():
            values = np.ascontiguousarray(values, dtype=np.float32)
            path = destination / f"{name}_{split}.npy"
            np.save(path, values)
            info["splits"][split] = {
                "rows": len(values),
                "d": values.shape[1],
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        metadata[name] = info
        print(name, {s: i["rows"] for s, i in info["splits"].items()}, flush=True)

    with tarfile.open(raw / "BSR_bsds500.tgz") as archive:
        rgb_splits, labxy_splits, selected = {}, {}, {}
        for split, folder, count in [("train", "train", 6), ("validation", "val", 3), ("test", "test", 3)]:
            names = sorted(
                m.name
                for m in archive.getmembers()
                if f"/BSDS500/data/images/{folder}/" in m.name and m.name.endswith(".jpg")
            )[:count]
            selected[split] = names
            rgb_values, labxy_values = [], []
            for name in names:
                rgb = (
                    np.asarray(Image.open(io.BytesIO(archive.extractfile(name).read())).convert("RGB"))
                    / 255.0
                )
                h, w = rgb.shape[:2]
                yy, xx = np.mgrid[:h, :w]
                xy = np.stack((xx, yy), axis=-1) / max(h, w)
                rgb_values.append(rgb.reshape(-1, 3))
                labxy_values.append(np.concatenate((rgb_to_lab(rgb) / 100, xy), axis=-1).reshape(-1, 5))
            permutation = np.random.default_rng(725).permutation(sum(map(len, rgb_values)))
            rgb_splits[split] = np.concatenate(rgb_values)[permutation]
            labxy_splits[split] = np.concatenate(labxy_values)[permutation]
        save(
            "bsds_rgb",
            rgb_splits,
            "sRGB/255 pixels; distinct original scene splits; repeated colors retained",
            selected,
        )
        save(
            "bsds_labxy",
            labxy_splits,
            "D65 CIELAB/100 plus x,y/max(height,width); global Euclidean surrogate, without SLIC locality/connectivity",
            selected,
        )

    for shape, filename, member in [
        ("bunny", "bunny.tar.gz", "bunny/reconstruction/bun_zipper.ply"),
        ("drill", "drill.tar.gz", "drill/reconstruction/drill_shaft_vrip.ply"),
    ]:
        with tarfile.open(raw / filename) as archive:
            vertices, faces = read_mesh(archive, member)
        scale = np.ptp(vertices, axis=0).max()
        vertices = (vertices - vertices.mean(axis=0)) / scale
        triangles = vertices[faces]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        area2 = np.linalg.norm(normals, axis=1)
        valid = area2 > 0
        triangles, normals, area2 = triangles[valid], normals[valid], area2[valid]
        normals /= area2[:, None]
        xyz_splits, normal_splits = {}, {}
        for index, (split, count) in enumerate(
            [("train", 1_000_000), ("validation", 100_000), ("test", 100_000)]
        ):
            rng = np.random.default_rng(950 + index)
            ids = rng.choice(len(triangles), count, p=area2 / area2.sum())
            u = np.sqrt(rng.random(count))
            v = rng.random(count)
            points = (
                (1 - u)[:, None] * triangles[ids, 0]
                + (u * (1 - v))[:, None] * triangles[ids, 1]
                + (u * v)[:, None] * triangles[ids, 2]
            )
            xyz_splits[split] = points
            normal_splits[split] = np.concatenate((points, 0.1 * normals[ids]), axis=1)
        save(
            f"{shape}_xyz",
            xyz_splits,
            "Independent area-uniform surface samples; common bounding-box scale; same-object distribution; not new scans",
            [member],
        )
        save(
            f"{shape}_xyzn",
            normal_splits,
            "Surface samples plus oriented face normals weighted 0.1; Euclidean surrogate, not VCCS",
            [member],
        )

    scans = []
    with gzip.open(raw / "intel.gfs.log.gz", "rt") as file:
        for line in file:
            parts = line.split()
            if not parts or parts[0] != "FLASER":
                continue
            count = int(parts[1])
            ranges = np.array(parts[2 : 2 + count], dtype=float)
            px, py, heading = map(float, parts[2 + count : 5 + count])
            angles = heading - np.pi / 2 + np.arange(count) * np.pi / count
            valid = (ranges > 0) & (ranges < 30)
            points = np.stack((px + ranges * np.cos(angles), py + ranges * np.sin(angles)), axis=1)
            scans.append(points[valid])
    a, b = int(len(scans) * 0.6), int(len(scans) * 0.8)
    blocks = {"train": scans[:a], "validation": scans[a:b], "test": scans[b:]}
    train = np.concatenate(blocks["train"])
    center = train.mean(axis=0)
    scale = np.ptp(train, axis=0).max()
    splits = {name: ((np.concatenate(block) - center) / scale) for name, block in blocks.items()}
    for name, values in splits.items():
        splits[name] = values[np.random.default_rng(725).permutation(len(values))]
    save(
        "intel_xy",
        splits,
        "Corrected Intel Lab FLASER endpoints; 180-degree scan from -pi/2 in pi/count steps; 0<range<30m; contiguous 60/20/20 scan blocks; center/scale from training only",
        {
            "archive": "intel.gfs.log.gz",
            "scan_count": len(scans),
            "scan_boundaries": [a, b],
            "center": center.tolist(),
            "scale": float(scale),
        },
    )
    (ROOT / "datasets/prepared_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    build()
