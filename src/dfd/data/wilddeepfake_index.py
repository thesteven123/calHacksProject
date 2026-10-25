"""Torch-free WildDeepfake tar indexing (runs on the CPU-only Modal ingest image)."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path


def build_wilddeepfake_index(root: str | Path, split: str = "test") -> list[dict]:
    """Index every frame member of the *_<split> tars with byte offsets for direct reads.

    Files are named .tar.gz but are uncompressed tar, so members are seekable via offset_data.
    """
    root = Path(root)
    entries: list[dict] = []
    for kind, label in (("real", 0), ("fake", 1)):
        d = root / "deepfake_in_the_wild" / f"{kind}_{split}"
        if not d.is_dir():
            raise FileNotFoundError(f"WildDeepfake directory missing: {d}")
        for tar_path in sorted(d.glob("*.tar.gz")):
            with tarfile.open(tar_path, "r:*") as tf:
                for m in tf:
                    if not m.isfile() or not m.name.lower().endswith(".png"):
                        continue
                    parts = [p for p in m.name.split("/") if p not in (".", "")]
                    if len(parts) != 4:
                        continue
                    video, member_kind, clip, frame = parts
                    if member_kind not in ("real", "fake"):
                        continue
                    member_label = 0 if member_kind == "real" else 1
                    if member_label != label:
                        raise ValueError(
                            f"{tar_path}: member {m.name} labelled {member_kind} inside {kind}_{split}")
                    entries.append({
                        "tar": str(tar_path.relative_to(root)),
                        "member": m.name,
                        "offset": int(m.offset_data),
                        "size": int(m.size),
                        "label": label,
                        "video": video,
                        "clip": clip,
                        "frame": int(Path(frame).stem),
                    })
    return entries


def write_wilddeepfake_index(root: str | Path, index: list[dict], split: str = "test") -> Path:
    path = Path(root) / f"index_{split}.json"
    path.write_text(json.dumps(index))
    return path
