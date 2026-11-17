"""Fetch unmodified reference sources at a fixed upstream revision."""

import hashlib
import json
import urllib.request
from pathlib import Path

REVISION = "c8f283867285af547db7a701c1b8dfa89631b3ea"
ROOT = Path(__file__).resolve().parents[1]


def fetch():
    """Download and verify pinned Faiss files into the ignored reference cache."""
    manifest_path = ROOT / "scripts/reference_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    destination = ROOT / ".cache/reference"
    for name, expected in manifest.items():
        path = destination / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            upstream = "LICENSE" if name == "LICENSE" else "contrib/" + name
            url = f"https://raw.githubusercontent.com/facebookresearch/faiss/{REVISION}/{upstream}"
            path.write_bytes(urllib.request.urlopen(url, timeout=60).read())
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Reference checksum mismatch: {name}")
    return destination


if __name__ == "__main__":
    print(fetch())
