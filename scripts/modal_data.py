"""Ingest external datasets into the `dfd-data` Modal volume and eval checkpoints against them.

    .venv/bin/modal run scripts/modal_data.py --action plan                     # sizes only, no compute
    .venv/bin/modal run scripts/modal_data.py --action ingest --dataset genimage
    .venv/bin/modal run scripts/modal_data.py --action report
    .venv/bin/modal run scripts/modal_data.py --action eval --dataset wilddeepfake \
        --checkpoint /runs/<tag>/<rotation>/best.pt

Volume layout: /data/genimage/data/<split>/<DIR>/..., /data/wilddeepfake/deepfake_in_the_wild/... (+index_test.json),
/data/sfhq_t2i/SyntheticFacesHighQuality-T2I.zip. Each dataset dir gets MANIFEST.json with the pinned revision sha.
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
}
SFHQ_ZIP = "SyntheticFacesHighQuality-T2I.zip"
EVAL_DATASETS = ("wilddeepfake", "sfhq_t2i")
GENIMAGE_DIRS = {"ADM": "ADM", "BigGAN": "BigGAN", "GLIDE": "glide", "Midjourney": "Midjourney",
                 "SD14": "stable_diffusion_v_1_4", "SD15": "stable_diffusion_v_1_5", "VQDM": "VQDM",
                 "Wukong": "wukong"}
_SHARD = re.compile(r"data-\d+-of-\d+\.arrow$")
CPU_N, MEM_MB, TIMEOUT_S = 4.0, 8192, 4 * 3600
# Modal list prices (USD/sec), modal.com/pricing 2026-10-24 — same rates as scripts/modal_logo.py.
CPU_PRICE_PER_S = CPU_N * 0.0000131 + (MEM_MB / 1024) * 0.00000222
GPU_PRICE_PER_S = 0.000222 + 8.0 * 0.0000131 + 32 * 0.00000222
TIB = 1024 ** 4

cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(  # pins mirror requirements.lock
        "huggingface-hub==2.2.0", "datasets==5.1.0", "pyarrow==26.0.0",
        "pillow==13.0.0", "numpy==2.5.4", "pyyaml==6.0.3",
    )
    .env({"PYTHONPATH": f"{REMOTE}/src", "HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(PROJECT / "src", f"{REMOTE}/src")
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
               train_shards: int = 4, smoke: bool = False) -> list[tuple[str, int]]:
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
                    download_fn, commit=None) -> tuple[int, int, int]:
    """Download `files` into `dest` via hf_hub_download's local_dir path, which revalidates per-file
    commit-hash/etag metadata instead of trusting size alone. Returns (downloaded, reused, bytes)."""
    dest = Path(dest)
    downloaded = reused = n_bytes = 0
    for path, size in files:
        local = dest / path
        before = (local.stat().st_mtime_ns, local.stat().st_size) if local.exists() else None
        download_fn(repo_id=repo, filename=path, repo_type="dataset",
                    revision=revision, local_dir=str(dest))
        if not local.exists() or local.stat().st_size != size:
            raise IOError(f"Size mismatch after download: {local}")
        if before is not None and before == (local.stat().st_mtime_ns, local.stat().st_size):
            reused += 1
        else:
            downloaded += 1
            n_bytes += size
            if commit and downloaded % 20 == 0:
                commit()
    if commit:
        commit()
    return downloaded, reused, n_bytes


def _plan(dataset: str, generators: tuple[str, ...], train_shards: int, smoke: bool) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    total = 0
    for name in ([dataset] if dataset != "all" else list(DATASETS)):
        spec = DATASETS[name]
        sha, tree, _ = _repo_tree(api, spec["repo"])
        files = plan_files(name, tree, generators, train_shards, smoke)
        size = sum(s for _, s in files)
        total += size
        # ~30 MB/s sustained HF download on the CPU ingest container -> cost estimate.
        est_s = size / 30e6
        print(f"[plan] {name:12s} repo={spec['repo']} rev={sha[:10]} files={len(files)} "
              f"bytes={size / 1e9:.1f} GB  est ingest ~{est_s / 3600:.1f}h CPU ~${est_s * CPU_PRICE_PER_S:.2f}")
        if name == "sfhq_t2i" and smoke:
            _sfhq_smoke_counts(spec["repo"])
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
    downloaded, reused, n_bytes = _download_files(spec["repo"], files, revision, dest,
                                                hf_hub_download, commit=data_vol.commit)
    if dataset == "wilddeepfake":
        from dfd.data.wilddeepfake_index import build_wilddeepfake_index, write_wilddeepfake_index

        write_wilddeepfake_index(dest, build_wilddeepfake_index(dest, split="test"), split="test")
    manifest = {"repo": spec["repo"], "revision": revision, "licence": spec["licence"],
                "selection": selection,
                "files": [{"path": p, "size": s, "sha256": sha256s.get(p)} for p, s in files],
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (dest / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    data_vol.commit()
    wall = time.time() - t0
    return {"dataset": dataset, "downloaded": downloaded, "reused": reused, "bytes": n_bytes,
            "wall_s": round(wall, 1), "est_cost_usd": round(wall * CPU_PRICE_PER_S, 3)}


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
def eval_external(dataset: str, checkpoint: str, frames_per_clip: int, max_per_generator: int | None) -> dict:
    import subprocess

    t0 = time.time()
    out_path = str(Path(checkpoint).with_name(f"eval_external_{dataset}.json"))
    cmd = ["python", f"{REMOTE}/scripts/eval_external.py", "--dataset", dataset,
           "--checkpoint", checkpoint, "--data-root", "/data", "--output", out_path,
           "--frames-per-clip", str(frames_per_clip), "--num-workers", "6"]
    if max_per_generator:
        cmd += ["--max-per-generator", str(max_per_generator)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    runs_vol.commit()
    return {"dataset": dataset, "checkpoint": checkpoint, "returncode": proc.returncode,
            "wall_s": round(time.time() - t0, 1), "est_cost_usd": round((time.time() - t0) * GPU_PRICE_PER_S, 3),
            "output": out_path, "report": json.loads(Path(out_path).read_text()) if Path(out_path).exists() else None,
            "log_tail": (proc.stdout + proc.stderr)[-4000:]}


@app.local_entrypoint()
def main(action: str = "plan", dataset: str = "all", generators: str = "", train_shards: int = 4,
         smoke: bool = False, checkpoint: str = "", frames_per_clip: int = 8,
         max_per_generator: int = 2000, tag: str = ""):
    gens = tuple(g for g in generators.split(",") if g)
    if action == "plan":
        _plan(dataset, gens, train_shards, smoke)
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
            files = plan_files(name, tree, gens, train_shards, smoke)
            print(f"[ingest] {name}: {len(files)} files, {sum(s for _, s in files) / 1e9:.1f} GB")
            if not files:
                if name == "sfhq_t2i" and smoke:
                    _sfhq_smoke_counts(DATASETS[name]["repo"])
                continue
            r = ingest.remote(name, files, sha,
                              {"generators": list(gens), "train_shards": train_shards, "smoke": smoke},
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
            r = eval_external.remote(name, checkpoint, frames_per_clip, max_per_generator or None)
            (local / f"eval_{name}.json").write_text(json.dumps(r, indent=2))
            print(f"[eval] {name}: rc={r['returncode']} ~${r['est_cost_usd']:.2f} -> {local / f'eval_{name}.json'}")
            if r["returncode"] != 0:
                failed.append(name)
                print(r["log_tail"][-1500:])
        if failed:
            raise SystemExit(f"eval failed for {failed} (see {local})")
        return
    raise SystemExit(f"Unknown action {action!r}; expected plan|ingest|report|eval")


if __name__ == "__main__":
    # Local-only fallback for `python scripts/modal_data.py plan [--dataset X] [--smoke]`.
    argv = sys.argv[1:]
    if not argv or argv[0] != "plan":
        raise SystemExit("Local mode supports only: python scripts/modal_data.py plan [--dataset X] [--smoke]")
    kv = {a.lstrip("-").replace("-", "_"): v for a, v in zip(argv[1::2], argv[2::2]) if not v.startswith("--")}
    smoke_flag = "--smoke" in argv
    _plan(kv.get("dataset", "all"),
          tuple(g for g in kv.get("generators", "").split(",") if g),
          int(kv.get("train_shards", 4)), smoke_flag)
