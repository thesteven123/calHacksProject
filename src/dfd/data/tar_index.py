"""Torch-free byte-offset index of image members in plain tars (runs on the CPU-only Modal ingest image)."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

IMAGE_TAR_EXTS = (".webp", ".png", ".jpg", ".jpeg")


def build_image_tar_index(root: str | Path, tar_paths, exts: tuple[str, ...] = IMAGE_TAR_EXTS) -> list[dict]:
    """{tar (relative to root), member, offset (offset_data), size} for every regular image member."""
    root = Path(root)
    entries: list[dict] = []
    for tar_path in sorted(Path(p) for p in tar_paths):
        with tarfile.open(tar_path, "r:") as tf:
            for m in tf:
                if m.isfile() and m.name.lower().endswith(tuple(exts)):
                    entries.append({"tar": str(tar_path.relative_to(root)), "member": m.name,
                                    "offset": int(m.offset_data), "size": int(m.size)})
    return entries


def write_index(path: str | Path, entries: list[dict]) -> Path:
    path = Path(path)
    path.write_text(json.dumps(entries))
    return path
