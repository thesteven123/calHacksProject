"""Video clips stored as pre-extracted frame tars (scripts/extract_video_frames.py) for temporal training.

VideoClipDataset yields (list[PIL.Image] of ALL stored frames, label, generator); TransformedDataset(frames_per_clip=T)
picks T frames and applies the per-clip-consistent transform.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from dfd.data.video_frames import extract_clip_frames, probe_video, square_resize, video_timestamps  # noqa: F401


class VideoClipDataset(Dataset):
    """Frames of the given clips (dicts from clips.json) read at indexed offsets in <root>/*.tar."""

    def __init__(self, root: str | Path, clips: list[dict], index_path: str | Path | None = None):
        self.root = Path(root)
        entries = json.loads((Path(index_path) if index_path else self.root / "index.json").read_text())
        by_clip: dict[str, list[tuple[int, dict]]] = {}
        for e in entries:
            clip_id, frame = e["member"].rsplit("/", 1)
            by_clip.setdefault(clip_id, []).append((int(Path(frame).stem), e))
        missing = [c["clip_id"] for c in clips if c["clip_id"] not in by_clip]
        if missing:
            raise ValueError(f"{len(missing)} clips have no frames in the index, e.g. {missing[:3]}")
        self.clips = list(clips)
        self.frames = [[e for _, e in sorted(by_clip[c["clip_id"]], key=lambda x: x[0])] for c in self.clips]
        self._handles: dict[str, object] = {}

    def __getstate__(self):
        d = self.__dict__.copy()
        d["_handles"] = {}
        return d

    def _handle(self, tar: str):
        if tar not in self._handles:
            self._handles[tar] = open(self.root / tar, "rb")
        return self._handles[tar]

    def __len__(self) -> int:
        return len(self.clips)

    def meta(self, idx: int) -> tuple[int, str]:
        c = self.clips[idx]
        return int(c["label"]), ("real" if int(c["label"]) == 0 else c["method"])

    def clip_id(self, idx: int) -> str:
        return self.clips[idx]["clip_id"]

    def __getitem__(self, idx: int) -> tuple[list[Image.Image], int, str]:
        frames = []
        for e in self.frames[idx]:
            f = self._handle(e["tar"])
            f.seek(e["offset"])
            frames.append(Image.open(io.BytesIO(f.read(e["size"]))).convert("RGB"))
        return frames, *self.meta(idx)
