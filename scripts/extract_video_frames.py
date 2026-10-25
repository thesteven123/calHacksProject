"""Extract time-sampled, format-aligned frames from TalkingHeadBench fakes + CelebV-HQ reals into frame tars.

    python scripts/extract_video_frames.py --thb-root /data/thb --celebvhq-root /data/celebvhq --out /data/video_frames
    python scripts/extract_video_frames.py --sources wdf --wdf-root /data/wilddeepfake   # merges into existing output

Every clip (real or fake) is sampled at 4 fps from t=0 (max 16 frames; shorter clips: 16 stamps spread over the
duration) and each frame is centre-square-cropped, resized to 256x256 (INTER_AREA) and JPEG q95-encoded, so source
resolution/fps (LivePortrait 320x320 @10 fps, CelebV-HQ 23.98 fps) are not label cues. WildDeepfake clips are
already PNG frame sequences: up to 16 frames evenly spaced over each clip, aligned identically. Celeb-DF v2 videos
are full frames, so each sampled frame is first cropped to the largest YuNet face (x1.3 square; centre-crop fallback
counted per class). Output:
<out>/<source>_<split>.tar (members <clip_id>/<k:02d>.jpg), index.json (tar byte offsets) and clips.json.
Torch-free: runs on the CPU-only Modal image.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import tarfile
import time
from multiprocessing import Pool
from pathlib import Path

from dfd.data.tar_index import build_image_tar_index, write_index
from dfd.data.video_frames import extract_clip_job, wdf_clips

THB_METHODS = ("EmoPortrait", "AniPortraitAudio", "LivePortrait", "Hallo2", "AniPortraitVideo", "Hallo")
THB_SPLITS = ("train", "val", "test")
WDF_SPLITS = ("train", "test")
SOURCES = ("thb", "celebvhq", "wdf", "celebdf")
CELEBDF_DIRS = {"Celeb-real": ("celebreal", 0), "YouTube-real": ("ytreal", 0), "Celeb-synthesis": ("synth", 1)}
_CELEBDF_TARGET = re.compile(r"^(id\d+)_")


def celebdf_clips(root: Path, face_onnx: str | None) -> list[dict]:
    """Celeb-DF v2 clips: test = List_of_testing_videos.txt paths (labels from the folder, not the list column),
    train = the rest. video_group = celebdf/idX (target video identity, so a real video and its fakes stay
    together) or celebdf/yt_<stem> for YouTube-real. idY (swapped-in face) identities still overlap across splits."""
    test = {line.split()[-1] for line in (root / "List_of_testing_videos.txt").read_text().splitlines() if line.strip()}
    clips = []
    for folder, (kind, label) in CELEBDF_DIRS.items():
        for p in sorted((root / folder).glob("*.mp4")):
            split = "test" if f"{folder}/{p.name}" in test else "train"
            m = _CELEBDF_TARGET.match(p.stem)
            group = f"celebdf/{m.group(1)}" if m else f"celebdf/yt_{p.stem}"
            clips.append({"clip_id": f"celebdf_{split}_{kind}_{p.stem}", "source": "celebdf", "split": split,
                          "method": "celebdf" if label else "real", "label": label, "video_group": group,
                          "path": str(p), "face_onnx": face_onnx})
    return clips
CELEBV_SPLITS = {"Train": "train", "Val": "val", "Test": "test"}


def list_clips(thb_root: Path, celebvhq_root: Path, sources: tuple[str, ...] = ("thb", "celebvhq"),
               wdf_root: Path | None = None, max_frames: int = 16, celebdf_root: Path | None = None,
               face_onnx: str | None = None) -> list[dict]:
    """THB fakes (clip_id = stem, method/split from the path), CelebV-HQ reals listed in real_dataset_split.json,
    and WildDeepfake train/test clips from index_<split>.json (frames chosen by dfd.data.video_frames.wdf_clips)."""
    clips = []
    if "thb" in sources:
        for method in THB_METHODS:
            for split in THB_SPLITS:
                for p in sorted((thb_root / "fake" / method / split).glob("*.mp4")):
                    clips.append({"clip_id": p.stem, "source": "thb", "method": method, "split": split, "label": 1,
                                  "path": str(p)})
    if "wdf" in sources:
        for split in WDF_SPLITS:
            index = wdf_root / f"index_{split}.json"
            if not index.exists():
                raise FileNotFoundError(f"{index} missing: ingest wilddeepfake (test) / wilddeepfake_train first")
            clips += [{**c, "root": str(wdf_root)} for c in wdf_clips(json.loads(index.read_text()), split, max_frames)]
    if "celebdf" in sources:
        if not face_onnx or not Path(face_onnx).is_file():
            raise FileNotFoundError(f"celebdf needs the YuNet model (--yunet); got {face_onnx!r}")
        clips += celebdf_clips(celebdf_root, str(face_onnx))
    if "celebvhq" not in sources:
        return clips
    split_json = json.loads((thb_root / "real" / "real_dataset_split.json").read_text())
    for key, split in CELEBV_SPLITS.items():
        for name in sorted(split_json.get(key, {}).get("CelebV-HQ", [])):
            stem = Path(name).stem
            clips.append({"clip_id": stem, "source": "celebvhq", "method": "real", "split": split, "label": 0,
                          "path": str(celebvhq_root / "clips" / f"{stem}.mp4")})
    return clips


def write_frames(clips: list[dict], out: Path, workers: int = 8, fps_sample: float = 4.0, max_frames: int = 16,
                 size: int = 256, quality: int = 95, sources: tuple[str, ...] | None = None) -> dict:
    """Extract all clips into <out>/<source>_<split>.tar, then write index.json, clips.json, failures.json.

    Merges with earlier runs: index.json covers every *.tar in `out`, and clips.json keeps the entries of sources
    not processed now (`sources`, default = the sources present in `clips`).
    """
    sources = tuple(sources or sorted({c["source"] for c in clips}))
    out.mkdir(parents=True, exist_ok=True)
    tars: dict[str, tarfile.TarFile] = {}
    done, failures = [], []
    jobs = [(c, fps_sample, max_frames, size, quality) for c in clips]
    with Pool(workers) as pool:
        for i, (meta, frames, err) in enumerate(pool.imap(extract_clip_job, jobs, chunksize=4)):
            if err:
                failures.append({"clip_id": meta["clip_id"], "source": meta["source"], "path": meta.get("path"),
                                 "error": err})
                continue
            name = f"{meta['source']}_{meta['split']}.tar"
            if name not in tars:
                tars[name] = tarfile.open(out / name, "w")
            for k, data in enumerate(frames):
                info = tarfile.TarInfo(f"{meta['clip_id']}/{k:02d}.jpg")
                info.size = len(data)
                tars[name].addfile(info, io.BytesIO(data))
            done.append(meta)
            if (i + 1) % 200 == 0:
                print(f"[frames] {i + 1}/{len(jobs)} clips, {len(failures)} failed", flush=True)
    for tf in tars.values():
        tf.close()
    write_index(out / "index.json", build_image_tar_index(out, sorted(out.glob("*.tar"))))
    old = json.loads((out / "clips.json").read_text()) if (out / "clips.json").exists() else []
    (out / "clips.json").write_text(json.dumps([c for c in old if c["source"] not in sources] + done))
    (out / "failures.json").write_text(json.dumps(failures, indent=2))
    return {"clips": len(done), "failed": len(failures), "tars": sorted(tars), "face_fallback": face_fallback(done)}


def face_fallback(clips: list[dict]) -> dict:
    """Centre-crop fallback (no face found) per class for face-cropped sources; a large real/fake difference would
    be a label shortcut."""
    out = {}
    for name, label in (("real", 0), ("fake", 1)):
        cs = [c for c in clips if "face_fallback_frames" in c and c["label"] == label]
        frames, total = sum(c["face_fallback_frames"] for c in cs), sum(c["n_frames"] for c in cs)
        out[name] = {"clips": len(cs), "fallback_frames": frames, "frames": total,
                     "rate": round(frames / total, 4) if total else None}
    return out


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thb-root", default="/data/thb")
    ap.add_argument("--celebvhq-root", default="/data/celebvhq")
    ap.add_argument("--wdf-root", default="/data/wilddeepfake")
    ap.add_argument("--celebdf-root", default="/data/celebdf")
    ap.add_argument("--yunet", default="/opt/yunet.onnx", help="YuNet ONNX for celebdf face crops")
    ap.add_argument("--sources", default="thb,celebvhq", help=f"comma-separated subset of {','.join(SOURCES)}")
    ap.add_argument("--out", default="/data/video_frames")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--max-frames", type=int, default=16)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--limit", type=int, default=None, help="first N clips only (smoke runs)")
    args = ap.parse_args(argv)

    sources = tuple(s for s in args.sources.split(",") if s)
    if set(sources) - set(SOURCES):
        raise SystemExit(f"Unknown sources {sorted(set(sources) - set(SOURCES))}; expected {SOURCES}")
    t0 = time.time()
    clips = list_clips(Path(args.thb_root), Path(args.celebvhq_root), sources, Path(args.wdf_root),
                       args.max_frames, Path(args.celebdf_root), args.yunet)[: args.limit]
    print(f"[frames] {len(clips)} clips "
          f"({sum(c['label'] == 1 for c in clips)} fake / {sum(c['label'] == 0 for c in clips)} real)", flush=True)
    r = write_frames(clips, Path(args.out), args.workers, args.fps, args.max_frames, args.size, args.quality, sources)
    r["wall_s"] = round(time.time() - t0, 1)
    print(f"[frames] done: {r}")
    return r


if __name__ == "__main__":
    main()
