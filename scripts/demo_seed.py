"""Seed the dfd-demo feed: held-out-only media under /demo + detection JSON per post.

Called from serve/modal_app.py's `seed` Modal function (volumes /data read, /demo write).
Every post is scored with the same Detector.detect path used by the API.
"""

from __future__ import annotations

import io
import json
import random
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

DATA = Path("/data")
DEMO = Path("/demo")
HANDLES = ["sunny.trails", "pixel.drift", "morning.frames", "quiet.harbor", "neon.orchard",
           "daily.dose", "studio.notes", "lens.and.light", "field.record", "urban.botany"]
CAPTIONS = ["weekend light", "new post", "archive find", "out take", "study 04", "morning walk",
            "back at it", "golden hour", "draft final v2", "unfiltered"]

SOURCES = {
    "fairface": "FairFace (CC BY 4.0)",
    "ffhq": "FFHQ (gaunernst/ffhq-1024-wds, held-out val)",
    "genimage_real": "GenImage test reals (CC-BY-NC-SA-4.0)",
    "sfhq": "SFHQ-T2I val (MIT)",
    "genimage_fake": "GenImage test fakes (CC-BY-NC-SA-4.0)",
    "celebvhq": "CelebV-HQ test (research only)",
    "wild_real": "WildDeepfake test reals (Apache-2.0)",
    "wild_fake": "WildDeepfake test fakes (Apache-2.0)",
    "thb": "TalkingHeadBench test (research only)",
}


def _resize_jpg(img: Image.Image, max_px: int = 1080, quality: int = 90) -> bytes:
    img = img.convert("RGB")
    img.thumbnail((max_px, max_px), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _fairface_pils(n: int, rng: random.Random) -> list[Image.Image]:
    import datasets

    ds = datasets.load_dataset("HuggingFaceM4/FairFace", "1.25", split="validation")
    idx = rng.sample(range(len(ds)), n)
    return [ds[i]["image"].convert("RGB") for i in idx]


def _ffhq_pils(n: int, rng: random.Random) -> list[Image.Image]:
    from dfd.data.external import FFHQDataset

    ds = FFHQDataset(DATA / "ffhq", split="val")
    idx = rng.sample(range(len(ds)), n)
    out = []
    for i in idx:
        e = ds.items[i]
        with tarfile.open(ds.root / e["tar"], "r:*") as tf:
            member = tf.getmember(e["member"])
            out.append(Image.open(tf.extractfile(member)).convert("RGB"))
    return out


def _genimage_pils(n: int, rng: random.Random, label: int, generators=("Midjourney",)) -> list[Image.Image]:
    from dfd.data.external import GenImageArrowDataset

    ds = GenImageArrowDataset(DATA / "genimage", "test", generators=generators,
                              balance=False, align_size=256)
    idxs = [i for i in range(len(ds)) if ds.meta(i)[0] == label]
    out = []
    for i in rng.sample(idxs, min(n, len(idxs))):
        di, r = ds._rows[i]
        raw = ds._dss[di][r]["image"]
        out.append(Image.open(io.BytesIO(raw)).convert("RGB"))
    return out


def _sfhq_pils(n: int, rng: random.Random, generators: tuple[str, ...]) -> list[Image.Image]:
    from dfd.data.external import SFHQT2IDataset

    ds = SFHQT2IDataset(DATA / "sfhq_t2i" / "SyntheticFacesHighQuality-T2I.zip",
                        generators=generators, split="val")
    idx = rng.sample(range(len(ds)), min(n, len(ds)))
    with zipfile.ZipFile(ds.zip_path) as zf:
        return [Image.open(io.BytesIO(zf.read(ds.items[i][0]))).convert("RGB") for i in idx]


def _avatar(name: str, seed_i: int) -> bytes:
    rng = np.random.default_rng(seed_i)
    size = 128
    img = Image.new("RGB", (size, size))
    c1, c2 = rng.integers(40, 200, 3), rng.integers(40, 200, 3)
    arr = np.linspace(0, 1, size)[:, None]
    grad = (c1 * (1 - arr) + c2 * arr).astype(np.uint8)
    img = Image.fromarray(np.tile(grad[:, None, :], (1, size, 1)))
    d = ImageDraw.Draw(img)
    initials = "".join(w[0] for w in name.replace(".", " ").split())[:2].upper()
    d.ellipse((8, 8, size - 8, size - 8), outline=(255, 255, 255), width=3)
    d.text((size // 2, size // 2), initials, anchor="mm", fill=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _encode_mp4_from_frames(frames: list[Image.Image], fps: float, out: Path) -> None:
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w, h = frames[0].size
    tmp = out.with_suffix(".raw.mp4")
    vw = cv2.VideoWriter(str(tmp), fourcc, fps, (w, h))
    for f in frames:
        vw.write(cv2.cvtColor(np.asarray(f.convert("RGB")), cv2.COLOR_RGB2BGR))
    vw.release()
    _ffmpeg_h264(tmp, out)
    tmp.unlink(missing_ok=True)


def _ffmpeg_h264(src: Path, out: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-t", "10", "-an", "-vf",
         "scale='min(720,iw)':-2", "-pix_fmt", "yuv420p", "-c:v", "libx264",
         "-preset", "fast", "-crf", "23", str(out)],
        check=True, capture_output=True)


def _wilddeepfake_frames(root: Path, rng: random.Random, label: int, n_clips: int = 1):
    entries = json.loads((root / "index_test.json").read_text())
    clips: dict[tuple, list[dict]] = {}
    for e in entries:
        if e["label"] == label:
            clips.setdefault((e["tar"], e["video"], e["clip"]), []).append(e)
    keys = sorted(clips)
    rng.shuffle(keys)
    for tar_rel, video, clip in keys[:n_clips]:
        frames = sorted(clips[(tar_rel, video, clip)], key=lambda e: e["frame"])
        imgs = []
        with open(root / tar_rel, "rb") as f:
            for e in frames:
                f.seek(e["offset"])
                imgs.append(Image.open(io.BytesIO(f.read(e["size"]))).convert("RGB"))
        yield f"{video}/{clip}", imgs


def run(detect) -> dict:
    """detect(data: bytes, filename: str) -> contract dict (Detector().detect.remote on Modal)."""
    rng = random.Random(0)
    media, feed = DEMO / "media", DEMO / "feed"
    for d in ("instagram", "tiktok", "avatars"):
        (media / d).mkdir(parents=True, exist_ok=True)
    feed.mkdir(parents=True, exist_ok=True)

    handles = HANDLES * 4
    rng.shuffle(handles)
    posts = {"instagram": [], "tiktok": []}
    summary = {"instagram": {"n": 0, "match": 0}, "tiktok": {"n": 0, "match": 0}}

    def post(feed_name: str, i: int, payload: bytes, ext: str, media_type: str,
             ground_truth: str, source: str):
        pid = f"{'ig' if feed_name == 'instagram' else 'tt'}_{i + 1:03d}"
        fn = f"{pid}.{ext}"
        (media / feed_name / fn).write_bytes(payload)
        handle = handles[(len(posts[feed_name])) % len(handles)]
        av = f"a{len(posts[feed_name]) % 10 + 1:02d}.jpg"
        av_path = media / "avatars" / av
        if not av_path.exists():
            av_path.write_bytes(_avatar(handle, len(posts[feed_name])))
        det = detect(payload, fn)
        ok = (det.get("verdict") == ground_truth) or (ground_truth == "real" and det.get("verdict") == "real") \
            or (ground_truth == "fake" and det.get("verdict") == "fake")
        summary[feed_name]["n"] += 1
        summary[feed_name]["match"] += int(ok)
        posts[feed_name].append({
            "id": pid, "handle": handle, "avatar": f"/media/avatars/{av}",
            "caption": rng.choice(CAPTIONS), "media": f"/media/{feed_name}/{fn}",
            "media_type": media_type, "likes": rng.randint(100, 9000),
            "comments": rng.randint(1, 400), "ground_truth": ground_truth,
            "source": source, "detection": det})

    # ---- Instagram: 15 real + 15 fake images ----
    real_imgs = (
        [(im, "fairface") for im in _fairface_pils(7, rng)]
        + [(im, "ffhq") for im in _ffhq_pils(4, rng)]
        + [(im, "genimage_real") for im in _genimage_pils(4, rng, 0, ("Midjourney",))]
    )
    fake_imgs = (
        [(im, "sfhq") for im in _sfhq_pils(4, rng, ("FLUX1-dev",))]
        + [(im, "sfhq") for im in _sfhq_pils(3, rng, ("FLUX1-pro",))]
        + [(im, "sfhq") for im in _sfhq_pils(2, rng, ("DALLE3",))]
        + [(im, "genimage_fake") for im in _genimage_pils(3, rng, 1, ("Midjourney",))]
        + [(im, "genimage_fake") for im in _genimage_pils(3, rng, 1, ("SD14",))]
    )
    for i, (im, src) in enumerate(real_imgs + fake_imgs):
        post("instagram", i, _resize_jpg(im), "jpg", "image",
             "real" if i < len(real_imgs) else "fake", SOURCES[src])

    # ---- TikTok: 8 real + 8 fake videos ----
    wild = DATA / "wilddeepfake"
    real_videos: list[tuple[Path, str]] = []
    split = json.loads((DATA / "thb" / "real" / "real_dataset_split.json").read_text())
    test_clips = [c for c in (split.get("Test") or split.get("test") or [])]
    rng.shuffle(test_clips)
    clip_dir = DATA / "celebvhq" / "clips"
    for c in test_clips[:5]:
        p = clip_dir / c if not str(c).startswith("/") else Path(c)
        if p.exists():
            real_videos.append((p, "celebvhq"))
    for i, (_, frames) in enumerate(_wilddeepfake_frames(wild, rng, 0, 3)):
        out = media / "tiktok" / f"_wild_real_{i}.mp4"
        _encode_mp4_from_frames(frames[:80], 8.0, out)
        real_videos.append((out, "wild_real"))

    fake_videos: list[tuple[Path, str]] = []
    for gen in ("EmoPortrait", "Hallo2"):
        d = DATA / "thb" / "fake" / gen / "test"
        vids = sorted(p for p in d.rglob("*.mp4")) if d.is_dir() else []
        take = 3 if gen == "EmoPortrait" else 2
        fake_videos += [(p, "thb") for p in rng.sample(vids, min(take, len(vids)))]
    for i, (_, frames) in enumerate(_wilddeepfake_frames(wild, rng, 1, 3)):
        out = media / "tiktok" / f"_wild_fake_{i}.mp4"
        _encode_mp4_from_frames(frames[:80], 8.0, out)
        fake_videos.append((out, "wild_fake"))

    i = 0
    for videos, gt in ((real_videos, "real"), (fake_videos, "fake")):
        for p, src in videos:
            out = media / "tiktok" / f"tt_{i + 1:03d}.mp4"
            if p.suffix == ".mp4" and not str(p).startswith(str(media)):
                _ffmpeg_h264(p, out)
            else:
                out.write_bytes(p.read_bytes()) if p != out else None
            post("tiktok", i, out.read_bytes(), "mp4", "video", gt, SOURCES[src])
            i += 1

    for name, posts_list in posts.items():
        (feed / f"{name}.json").write_text(json.dumps({"posts": posts_list}, indent=1))
    return {"posts": {k: len(v) for k, v in posts.items()}, "summary": summary}
