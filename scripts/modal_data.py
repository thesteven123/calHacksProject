"""Ingest external datasets into the `dfd-data` Modal volume and eval checkpoints against them.

    .venv/bin/modal run scripts/modal_data.py --action plan                     # sizes only, no compute
    .venv/bin/modal run scripts/modal_data.py --action ingest --dataset genimage
    .venv/bin/modal run scripts/modal_data.py --action report
    .venv/bin/modal run scripts/modal_data.py --action eval --dataset wilddeepfake \
        --checkpoint /runs/<tag>/<rotation>/best.pt
    .venv/bin/modal run scripts/modal_data.py --action ood --run-tag full7_dinov2_conformal   # OOD-gate experiment
    .venv/bin/modal run scripts/modal_data.py --action ingest --dataset thb        # then --dataset celebvhq
    .venv/bin/modal run scripts/modal_data.py --action frames                      # -> /data/video_frames

Volume layout: /data/genimage/data/<split>/<DIR>/..., /data/wilddeepfake/deepfake_in_the_wild/... (+index_test.json),
/data/sfhq_t2i/SyntheticFacesHighQuality-T2I.zip, /data/ffhq/<NNNNN>.tar (+index.json),
/data/thb/{fake/<method>/<split>/*.mp4, real/*.json}, /data/celebvhq/{videos.tar, clips/<clip>.mp4},
/data/video_frames/<source>_<split>.tar (+index.json, clips.json).
Each dataset dir gets MANIFEST.json with the pinned revision sha.
`plan` runs fully locally (HfApi metadata only) — no Modal compute is spawned.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import modal

PROJECT = Path(__file__).resolve().parents[1]
REMOTE = "/root/project"
DATASETS = {
    "genimage": {"repo": "nebula/GenImage-arrow", "licence": "CC-BY-NC-SA-4.0", "dest": "/data/genimage"},
    "wilddeepfake": {"repo": "xingjunm/WildDeepfake", "licence": "Apache-2.0", "dest": "/data/wilddeepfake"},
    "sfhq_t2i": {"repo": "bitmind/SyntheticFacesHQ", "licence": "MIT", "dest": "/data/sfhq_t2i"},
    "ffhq": {"repo": "gaunernst/ffhq-1024-wds", "licence": "CC-BY-NC-SA-4.0", "dest": "/data/ffhq"},
    "thb": {"repo": "luchaoqi/TalkingHeadBench", "licence": "MIT", "dest": "/data/thb"},
    "celebvhq": {"repo": "SwayStar123/CelebV-HQ", "dest": "/data/celebvhq",
                 "licence": "non-commercial research (official CelebV-HQ terms; unofficial HF mirror)"},
}
THB_METHODS = ("EmoPortrait", "AniPortraitAudio", "LivePortrait", "Hallo2", "AniPortraitVideo", "Hallo")
_THB_FAKE = re.compile(r"^fake/(" + "|".join(THB_METHODS) + r")/(train|val|test)/[^/]+\.mp4$")
_THB_REAL_JSON = re.compile(r"^real/[^/]+\.json$")
THB_SPLIT_JSON = "/data/thb/real/real_dataset_split.json"
CELEBVHQ_FILES = ("videos.tar", "celebvhq_info.json")
SFHQ_ZIP = "SyntheticFacesHighQuality-T2I.zip"
EVAL_DATASETS = ("wilddeepfake", "sfhq_t2i", "fairface")
GENIMAGE_DIRS = {"ADM": "ADM", "BigGAN": "BigGAN", "GLIDE": "glide", "Midjourney": "Midjourney",
                 "SD14": "stable_diffusion_v_1_4", "SD15": "stable_diffusion_v_1_5", "VQDM": "VQDM",
                 "Wukong": "wukong"}
_SHARD = re.compile(r"data-\d+-of-\d+\.arrow$")
_FFHQ_TAR = re.compile(r"^\d+\.tar$")
CPU_N, MEM_MB, TIMEOUT_S = 4.0, 8192, 4 * 3600
# Modal list prices (USD/sec), modal.com/pricing 2026-10-24 — same rates as scripts/modal_logo.py.
CPU_PRICE_PER_S = CPU_N * 0.0000131 + (MEM_MB / 1024) * 0.00000222
GPU_PRICE_PER_S = 0.000222 + 8.0 * 0.0000131 + 32 * 0.00000222
TIB = 1024 ** 4

cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(  # pins mirror requirements.lock
        "huggingface-hub==2.2.0", "datasets==5.1.0", "pyarrow==26.0.0",
        "pillow==13.0.0", "numpy==2.5.4", "pyyaml==6.0.3", "opencv-python-headless==5.0.0.93",
    )
    .env({"PYTHONPATH": f"{REMOTE}/src", "HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(PROJECT / "src", f"{REMOTE}/src")
    .add_local_dir(PROJECT / "scripts", f"{REMOTE}/scripts")
)
gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .uv_pip_install(  # same pins as scripts/modal_logo.py image
        "torch==2.14.1", "torchvision==0.29.1", "transformers==5.20.1", "datasets==5.1.0",
        "huggingface-hub==2.2.0", "timm==1.0.30", "pillow==13.0.0", "numpy==2.5.4", "scikit-learn==1.9.1",
        "pyyaml==6.0.3", "opencv-python-headless==5.0.0.93", "safetensors==0.8.0", "tqdm==4.70.1",
        "pyarrow==26.0.0",
    )
    .env({"PYTHONPATH": f"{REMOTE}/src", "HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(PROJECT / "src", f"{REMOTE}/src")
    .add_local_dir(PROJECT / "scripts", f"{REMOTE}/scripts")
)
app = modal.App("dfd-data", image=cpu_image)
data_vol = modal.Volume.from_name("dfd-data", create_if_missing=True)
runs_vol = modal.Volume.from_name("dfd-runs", create_if_missing=True)
hf_cache = modal.Volume.from_name("dfd-hf-cache", create_if_missing=True)


def _hf_secret() -> list:
    """Ephemeral secret carrying ONLY HF_TOKEN. Read locally from env or .env."""
    import os

    token = os.environ.get("HF_TOKEN", "")
    env_file = PROJECT / ".env"
    if not token and modal.is_local() and env_file.is_file():
        for line in env_file.read_text().splitlines():
            if line.strip().startswith("HF_TOKEN="):
                token = line.split("=", 1)[1].strip().strip("\"'")
    return [modal.Secret.from_dict({"HF_TOKEN": token})] if token else []


def select_shards(n: int, k: int | None) -> list[int]:
    """Same spreading rule as dfd.data.external.select_shards (kept dependency-free for plan)."""
    import numpy as np

    if k is None or k >= n:
        return list(range(n))
    return sorted(set(int(i) for i in np.linspace(0, n - 1, k).round()))


def plan_files(dataset: str, tree: list[tuple[str, int]], generators: tuple[str, ...] = (),
               train_shards: int = 4, smoke: bool = False, ffhq_shards: int = 15) -> list[tuple[str, int]]:
    """Pure file selection over a repo's (path, size_bytes) listing; unit-testable offline."""
    out: list[tuple[str, int]] = []
    if dataset == "genimage":
        gens = ["ADM"] if smoke else (list(generators) or list(GENIMAGE_DIRS))
        for g in gens:
            d = GENIMAGE_DIRS[g]
            for split, k in (("test", 1 if smoke else None), ("train", 1 if smoke else train_shards)):
                leaf = f"data/{split}/{d}/"
                leaf_files = sorted(p for p, _ in tree
                                    if p.startswith(leaf) and "/" not in p[len(leaf):])
                shards = [p for p in leaf_files if _SHARD.search(p)]
                meta = [p for p in leaf_files if p.endswith(".json")]
                chosen = {shards[i] for i in select_shards(len(shards), k)} | set(meta)
                out += [(p, s) for p, s in tree if p in chosen]
    elif dataset == "wilddeepfake":
        for kind in ("real_test", "fake_test"):
            prefix = f"deepfake_in_the_wild/{kind}/"
            files = sorted((p, s) for p, s in tree if p.startswith(prefix))
            out += files[:1] if smoke else files
    elif dataset == "sfhq_t2i":
        if not smoke:  # smoke = no download; plan reads the remote zip central directory instead
            out += [(p, s) for p, s in tree if p.endswith(SFHQ_ZIP)]
    elif dataset == "ffhq":
        tars = sorted((p, s) for p, s in tree if _FFHQ_TAR.match(p))
        out += [tars[i] for i in select_shards(len(tars), 1 if smoke else ffhq_shards)]
    elif dataset == "thb":
        fakes = sorted((p, s) for p, s in tree if _THB_FAKE.match(p))
        if smoke:
            fakes = [f for m in THB_METHODS for f in [f for f in fakes if f[0].startswith(f"fake/{m}/test/")][:2]]
        out += fakes + sorted((p, s) for p, s in tree if _THB_REAL_JSON.match(p))
    elif dataset == "celebvhq":
        if not smoke:  # smoke = no download; plan/ingest print the needed-clip count instead
            out += [(p, s) for p, s in tree if p in CELEBVHQ_FILES]
    else:
        raise ValueError(f"Unknown dataset {dataset!r}")
    return out


def _repo_tree(api, repo: str) -> tuple[str, list[tuple[str, int]], dict[str, str]]:
    info = api.dataset_info(repo)
    entries = [f for f in api.list_repo_tree(repo, repo_type="dataset", recursive=True)
               if getattr(f, "size", None) is not None]
    files = [(f.path, int(f.size)) for f in entries]
    sha256s = {f.path: f.lfs.sha256 for f in entries
               if getattr(getattr(f, "lfs", None), "sha256", None)}
    return info.sha, files, sha256s


def _eval_names(dataset: str) -> list[str]:
    if dataset == "all":
        return list(EVAL_DATASETS)
    if dataset in EVAL_DATASETS:
        return [dataset]
    raise SystemExit(f"Unknown eval dataset {dataset!r}; supported: {list(EVAL_DATASETS)} or 'all'")


def _download_files(repo: str, files: list[tuple[str, int]], revision: str, dest,
                    download_fn, commit=None, workers: int = 8, commit_every: int = 200) -> tuple[int, int, int]:
    """Download `files` into `dest` via hf_hub_download's local_dir path, which revalidates per-file
    commit-hash/etag metadata instead of trusting size alone. Returns (downloaded, reused, bytes).

    Files are fetched by `workers` threads (per-file HTTP overhead dominates for many small files); the volume is
    committed from this thread every `commit_every` new downloads and at the end."""
    from concurrent.futures import ThreadPoolExecutor

    dest = Path(dest)

    def fetch(item: tuple[str, int]) -> bool:
        path, size = item
        local = dest / path
        before = (local.stat().st_mtime_ns, local.stat().st_size) if local.exists() else None
        download_fn(repo_id=repo, filename=path, repo_type="dataset", revision=revision, local_dir=str(dest))
        if not local.exists() or local.stat().st_size != size:
            raise IOError(f"Size mismatch after download: {local}")
        return before is not None and before == (local.stat().st_mtime_ns, local.stat().st_size)

    downloaded = reused = n_bytes = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for (path, size), was_reused in zip(files, pool.map(fetch, files)):
            if was_reused:
                reused += 1
                continue
            downloaded += 1
            n_bytes += size
            if commit and downloaded % commit_every == 0:
                commit()
    if commit:
        commit()
    return downloaded, reused, n_bytes


def celebvhq_wanted(split_json: dict) -> set[str]:
    """Clip stems listed under "CelebV-HQ" in TalkingHeadBench's real_dataset_split.json (all splits)."""
    return {Path(n).stem for part in split_json.values() for n in (part or {}).get("CelebV-HQ", [])}


def extract_tar_members(tar_path: str | Path, wanted: set[str], out_dir: str | Path) -> dict:
    """Stream the tar once and copy only <...>/<stem>.mp4 members with stem in `wanted` to <out_dir>/<stem>.mp4.

    An existing file is skipped only if its sha256 equals the member's (the tar is streamed anyway, so hashing is
    free); same-size changed content is rewritten. Returns {extracted, skipped, missing (sorted stems never seen)}.
    """
    import hashlib
    import tarfile

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seen, extracted, skipped = set(), 0, 0
    with tarfile.open(tar_path, "r|*") as tf:
        for m in tf:
            stem = Path(m.name).stem
            if not m.isfile() or Path(m.name).suffix != ".mp4" or stem not in wanted:
                continue
            seen.add(stem)
            dest = out_dir / f"{stem}.mp4"
            with tf.extractfile(m) as src:
                data = src.read()
            if dest.exists() and hashlib.sha256(dest.read_bytes()).digest() == hashlib.sha256(data).digest():
                skipped += 1
                continue
            tmp = dest.with_suffix(".mp4.part")
            tmp.write_bytes(data)
            tmp.replace(dest)
            extracted += 1
    return {"extracted": extracted, "skipped": skipped, "missing": sorted(wanted - seen)}


def _celebvhq_needed_count() -> None:
    """Smoke/plan preview: how many CelebV-HQ clips TalkingHeadBench's real split needs (reads one small json)."""
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(DATASETS["thb"]["repo"], "real/real_dataset_split.json", repo_type="dataset")
        print(f"[plan] celebvhq needed clips: {len(celebvhq_wanted(json.loads(Path(path).read_text())))}")
    except Exception as e:  # noqa: BLE001 - best-effort preview
        print(f"[plan] celebvhq needed-clip count failed: {type(e).__name__}: {e}")


def _plan(dataset: str, generators: tuple[str, ...], train_shards: int, smoke: bool, ffhq_shards: int = 15) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    total = 0
    for name in ([dataset] if dataset != "all" else list(DATASETS)):
        spec = DATASETS[name]
        sha, tree, _ = _repo_tree(api, spec["repo"])
        files = plan_files(name, tree, generators, train_shards, smoke, ffhq_shards)
        size = sum(s for _, s in files)
        total += size
        # ~30 MB/s sustained HF download on the CPU ingest container -> cost estimate.
        est_s = size / 30e6
        print(f"[plan] {name:12s} repo={spec['repo']} rev={sha[:10]} files={len(files)} "
              f"bytes={size / 1e9:.1f} GB  est ingest ~{est_s / 3600:.1f}h CPU ~${est_s * CPU_PRICE_PER_S:.2f}")
        if name == "sfhq_t2i" and smoke:
            _sfhq_smoke_counts(spec["repo"])
        if name == "celebvhq":
            _celebvhq_needed_count()
    print(f"[plan] total {total / 1e9:.1f} GB ({total / TIB:.2%} of the 1 TiB included volume)")


def _sfhq_smoke_counts(repo: str) -> None:
    """Read the remote zip central directory only (no download) and count per-generator files."""
    try:
        import zipfile

        from huggingface_hub import HfFileSystem

        from dfd.data.external import _SFHQ_PREFIX, SFHQ_GENERATORS

        fs = HfFileSystem()
        with fs.open(f"datasets/{repo}/{SFHQ_ZIP}") as f, zipfile.ZipFile(f) as zf:
            counts: dict[str, int] = {}
            for name in zf.namelist():
                m = _SFHQ_PREFIX.match(Path(name).name)
                if m:
                    counts[SFHQ_GENERATORS[m.group(1)]] = counts.get(SFHQ_GENERATORS[m.group(1)], 0) + 1
        print(f"[plan] sfhq_t2i remote zip counts: {counts}")
    except Exception as e:  # noqa: BLE001 - best-effort smoke preview
        print(f"[plan] sfhq_t2i remote zip listing failed: {type(e).__name__}: {e}")


@app.function(cpu=CPU_N, memory=MEM_MB, timeout=TIMEOUT_S,
              volumes={"/data": data_vol, "/cache": hf_cache}, secrets=_hf_secret())
def ingest(dataset: str, files: list[tuple[str, int]], revision: str,
           selection: dict, sha256s: dict) -> dict:
    from huggingface_hub import hf_hub_download

    t0 = time.time()
    spec = DATASETS[dataset]
    dest = Path(spec["dest"])
    if dataset == "celebvhq" and not Path(THB_SPLIT_JSON).exists():
        raise FileNotFoundError(f"{THB_SPLIT_JSON} missing: ingest thb first (it lists the CelebV-HQ clips needed)")
    downloaded, reused, n_bytes = _download_files(spec["repo"], files, revision, dest,
                                                hf_hub_download, commit=data_vol.commit)
    if dataset == "wilddeepfake":
        from dfd.data.wilddeepfake_index import build_wilddeepfake_index, write_wilddeepfake_index

        write_wilddeepfake_index(dest, build_wilddeepfake_index(dest, split="test"), split="test")
    if dataset == "ffhq":
        from dfd.data.tar_index import build_image_tar_index, write_index

        write_index(dest / "index.json", build_image_tar_index(dest, [dest / p for p, _ in files]))
    if dataset == "celebvhq":
        tar_path = next(dest / p for p, _ in files if p.endswith("videos.tar"))
        wanted = celebvhq_wanted(json.loads(Path(THB_SPLIT_JSON).read_text()))
        selection = {**selection, "extraction": {"wanted": len(wanted),
                                                 **extract_tar_members(tar_path, wanted, dest / "clips")}}
        data_vol.commit()
    manifest = {"repo": spec["repo"], "revision": revision, "licence": spec["licence"],
                "selection": selection,
                "files": [{"path": p, "size": s, "sha256": sha256s.get(p)} for p, s in files],
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (dest / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    data_vol.commit()
    wall = time.time() - t0
    return {"dataset": dataset, "downloaded": downloaded, "reused": reused, "bytes": n_bytes,
            "wall_s": round(wall, 1), "est_cost_usd": round(wall * CPU_PRICE_PER_S, 3)}


FRAMES_CPU, FRAMES_MEM_MB = 8.0, 16384
FRAMES_PRICE_PER_S = FRAMES_CPU * 0.0000131 + (FRAMES_MEM_MB / 1024) * 0.00000222


@app.function(cpu=FRAMES_CPU, memory=FRAMES_MEM_MB, timeout=4 * 3600, volumes={"/data": data_vol})
def extract_frames(limit: int | None = None) -> dict:
    import subprocess

    t0 = time.time()
    cmd = ["python", f"{REMOTE}/scripts/extract_video_frames.py", "--thb-root", "/data/thb",
           "--celebvhq-root", "/data/celebvhq", "--out", "/data/video_frames", "--workers", str(int(FRAMES_CPU))]
    if limit:
        cmd += ["--limit", str(limit)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    data_vol.commit()
    wall = time.time() - t0
    return {"returncode": proc.returncode, "wall_s": round(wall, 1),
            "est_cost_usd": round(wall * FRAMES_PRICE_PER_S, 3), "log_tail": (proc.stdout + proc.stderr)[-4000:]}


@app.function(cpu=2.0, memory=4096, timeout=600, volumes={"/data": data_vol})
def report_volume() -> dict:
    out = {}
    for name, spec in DATASETS.items():
        dest = Path(spec["dest"])
        mf = dest / "MANIFEST.json"
        out[name] = json.loads(mf.read_text()) if mf.exists() else {"missing": str(mf)}
    usage = {}
    for d in Path("/data").iterdir():
        if d.is_dir():
            usage[d.name] = sum(p.stat().st_size for p in d.rglob("*") if p.is_file())
    return {"manifests": out, "bytes_by_dir": usage}


@app.function(gpu="L4", cpu=8.0, memory=32768, timeout=2 * 3600, image=gpu_image,
              volumes={"/data": data_vol, "/runs": runs_vol, "/cache": hf_cache}, secrets=_hf_secret())
def eval_external(dataset: str, checkpoint: str, frames_per_clip: int, max_per_generator: int | None,
                  sfhq_split: str = "all") -> dict:
    import subprocess

    t0 = time.time()
    out_path = str(Path(checkpoint).with_name(f"eval_external_{dataset}.json"))
    cmd = ["python", f"{REMOTE}/scripts/eval_external.py", "--dataset", dataset,
           "--checkpoint", checkpoint, "--data-root", "/data", "--output", out_path,
           "--frames-per-clip", str(frames_per_clip), "--num-workers", "6"]
    if max_per_generator:
        cmd += ["--max-per-generator", str(max_per_generator)]
    if dataset == "sfhq_t2i":
        cmd += ["--sfhq-split", sfhq_split]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    runs_vol.commit()
    if dataset == "fairface":
        hf_cache.commit()  # keep the FairFace download for later evals
    return {"dataset": dataset, "checkpoint": checkpoint, "returncode": proc.returncode,
            "wall_s": round(time.time() - t0, 1), "est_cost_usd": round((time.time() - t0) * GPU_PRICE_PER_S, 3),
            "output": out_path, "report": json.loads(Path(out_path).read_text()) if Path(out_path).exists() else None,
            "log_tail": (proc.stdout + proc.stderr)[-4000:]}


@app.function(gpu="L4", cpu=8.0, memory=32768, timeout=2 * 3600, image=gpu_image,
              volumes={"/data": data_vol, "/runs": runs_vol, "/cache": hf_cache}, secrets=_hf_secret())
def ood_gate_eval(checkpoint: str, alphas: str, frames_per_clip: int, max_per_generator: int) -> dict:
    import subprocess

    t0 = time.time()
    out_path = str(Path(checkpoint).with_name("ood_gate_eval.json"))
    cmd = ["python", f"{REMOTE}/scripts/ood_gate_eval.py", "--checkpoint", checkpoint, "--alphas", alphas,
           "--data-root", "/data", "--output", out_path, "--frames-per-clip", str(frames_per_clip),
           "--max-per-generator", str(max_per_generator), "--num-workers", "6"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    runs_vol.commit()
    return {"checkpoint": checkpoint, "returncode": proc.returncode,
            "wall_s": round(time.time() - t0, 1), "est_cost_usd": round((time.time() - t0) * GPU_PRICE_PER_S, 3),
            "report": json.loads(Path(out_path).read_text()) if Path(out_path).exists() else None,
            "log_tail": (proc.stdout + proc.stderr)[-4000:]}


def _run_checkpoints(run_tag: str) -> list[str]:
    """Every /runs/<run_tag>/*/best.pt on the dfd-runs volume (listed from the client, no compute)."""
    return sorted(f"/runs/{e.path.lstrip('/')}" for e in runs_vol.listdir(run_tag, recursive=True)
                  if e.path.endswith("/best.pt") and e.path.lstrip("/").count("/") == 2)


@app.local_entrypoint()
def main(action: str = "plan", dataset: str = "all", generators: str = "", train_shards: int = 4,
         smoke: bool = False, checkpoint: str = "", frames_per_clip: int = 8,
         max_per_generator: int = 2000, tag: str = "", run_tag: str = "", alphas: str = "0.05,0.10",
         sfhq_split: str = "all", ffhq_shards: int = 15):
    gens = tuple(g for g in generators.split(",") if g)
    if action == "plan":
        _plan(dataset, gens, train_shards, smoke, ffhq_shards)
        return
    if action == "report":
        r = report_volume.remote()
        for name, mf in r["manifests"].items():
            files, size = mf.get("files", []), None
            total = sum(f["size"] for f in files) / 1e9 if files else 0
            print(f"[report] {name:12s} rev={str(mf.get('revision'))[:10]} files={len(files)} "
                  f"~{total:.1f} GB created={mf.get('created_utc')}")
        print(f"[report] /data usage: { {k: f'{v / 1e9:.1f} GB' for k, v in r['bytes_by_dir'].items()} }")
        return
    if action == "ingest":
        names = [dataset] if dataset != "all" else list(DATASETS)
        unknown = [n for n in names if n not in DATASETS]
        if unknown:
            raise SystemExit(f"Unknown dataset(s) {unknown}; supported: {list(DATASETS)} or 'all'")
        from huggingface_hub import HfApi

        api = HfApi()
        for name in names:
            sha, tree, sha256s = _repo_tree(api, DATASETS[name]["repo"])
            files = plan_files(name, tree, gens, train_shards, smoke, ffhq_shards)
            print(f"[ingest] {name}: {len(files)} files, {sum(s for _, s in files) / 1e9:.1f} GB")
            if not files:
                if name == "sfhq_t2i" and smoke:
                    _sfhq_smoke_counts(DATASETS[name]["repo"])
                if name == "celebvhq" and smoke:
                    _celebvhq_needed_count()
                continue
            r = ingest.remote(name, files, sha,
                              {"generators": list(gens), "train_shards": train_shards, "smoke": smoke,
                               "ffhq_shards": ffhq_shards},
                              sha256s)
            print(f"[ingest] {name}: {r}")
        return
    if action == "eval":
        if not checkpoint:
            raise SystemExit("--action eval requires --checkpoint (e.g. /runs/<tag>/<rotation>/best.pt)")
        names = _eval_names(dataset)
        tag = tag or f"eval_{time.strftime('%Y%m%d-%H%M%S')}"
        local = PROJECT / "runs" / "modal" / tag
        local.mkdir(parents=True, exist_ok=True)
        failed = []
        for name in names:
            r = eval_external.remote(name, checkpoint, frames_per_clip, max_per_generator or None, sfhq_split)
            (local / f"eval_{name}.json").write_text(json.dumps(r, indent=2))
            print(f"[eval] {name}: rc={r['returncode']} ~${r['est_cost_usd']:.2f} -> {local / f'eval_{name}.json'}")
            if r["returncode"] != 0:
                failed.append(name)
                print(r["log_tail"][-1500:])
        if failed:
            raise SystemExit(f"eval failed for {failed} (see {local})")
        return
    if action == "ood":
        if bool(checkpoint) == bool(run_tag):
            raise SystemExit("--action ood requires exactly one of --checkpoint or --run-tag")
        ckpts = [checkpoint] if checkpoint else _run_checkpoints(run_tag)
        if not ckpts:
            raise SystemExit(f"No /runs/{run_tag}/*/best.pt found on the dfd-runs volume")
        if len(ckpts) > 7:
            raise SystemExit(f"{len(ckpts)} checkpoints exceed the 7-job cap: {ckpts}")
        tag = tag or run_tag or f"ood_{time.strftime('%Y%m%d-%H%M%S')}"
        local = PROJECT / "runs" / "modal" / tag
        local.mkdir(parents=True, exist_ok=True)
        args = [(c, alphas, frames_per_clip, max_per_generator) for c in ckpts]
        failed = []
        for c, r in zip(ckpts, ood_gate_eval.starmap(args, return_exceptions=True)):
            rotation = Path(c).parent.name
            path = local / f"ood_{rotation}.json"
            if isinstance(r, BaseException):
                path.write_text(json.dumps({"checkpoint": c, "error": f"{type(r).__name__}: {r}"}, indent=2))
                failed.append(rotation)
                print(f"[ood] {rotation}: EXCEPTION {type(r).__name__}: {r}")
                continue
            path.write_text(json.dumps(r, indent=2))
            print(f"[ood] {rotation}: rc={r['returncode']} ~${r['est_cost_usd']:.2f} -> {path}")
            if r["returncode"] != 0:
                failed.append(rotation)
                print(r["log_tail"][-1500:])
        if failed:
            raise SystemExit(f"ood eval failed for {failed} (see {local})")
        return
    if action == "frames":
        r = extract_frames.remote(24 if smoke else None)
        print(f"[frames] rc={r['returncode']} wall={r['wall_s'] / 60:.1f}min ~${r['est_cost_usd']:.2f}")
        print(r["log_tail"][-1500:])
        if r["returncode"] != 0:
            raise SystemExit("frame extraction failed")
        return
    raise SystemExit(f"Unknown action {action!r}; expected plan|ingest|report|eval|ood|frames")


if __name__ == "__main__":
    # Local-only fallback for `python scripts/modal_data.py plan [--dataset X] [--smoke]`.
    argv = sys.argv[1:]
    if not argv or argv[0] != "plan":
        raise SystemExit("Local mode supports only: python scripts/modal_data.py plan [--dataset X] [--smoke]")
    kv = {a.lstrip("-").replace("-", "_"): v for a, v in zip(argv[1::2], argv[2::2]) if not v.startswith("--")}
    smoke_flag = "--smoke" in argv
    _plan(kv.get("dataset", "all"),
          tuple(g for g in kv.get("generators", "").split(",") if g),
          int(kv.get("train_shards", 4)), smoke_flag, int(kv.get("ffhq_shards", 15)))
