"""Run nested leave-one-generator-out (LOGO) rotations of the core detector on Modal GPUs.

    .venv/bin/modal run scripts/modal_logo.py                                   # 1 rotation: GLIDE -> Midjourney
    .venv/bin/modal run scripts/modal_logo.py --rotations all                   # all 7 test generators, in parallel
    .venv/bin/modal run scripts/modal_logo.py --rotations all --backbone facebook/dinov3-vitb16-pretrain-lvd1689m

Each rotation = (selection generator, test generator): train on the other five, pick the checkpoint on the
selection generator, score the test generator once (scripts/train.py nested protocol). Reports land in
runs/modal/<tag>/<rotation>/ locally and on the `dfd-runs` Volume.

Cost controls: L4 GPU + 8 cores + 32 GiB (~$1.43/h list), hard per-job timeout, capped concurrency, and an
estimated cost per job printed from measured wall time. Only HF_TOKEN is sent to Modal (ephemeral secret),
never TINKER_API_KEY or GITHUB_TOKEN.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import modal

PROJECT = Path(__file__).resolve().parents[1]
REMOTE = "/root/project"
GENERATORS = ["ADM", "BigGAN", "GLIDE", "Midjourney", "SD15", "VQDM", "Wukong"]  # SD14 absent in Tiny-GenImage
GPU, CPU, MEM_MB, TIMEOUT_S, MAX_PARALLEL = "L4", 8.0, 32768, 2 * 3600, 7
# Modal list prices (USD/sec), modal.com/pricing 2026-10-24
PRICE_PER_S = 0.000222 + CPU * 0.0000131 + (MEM_MB / 1024) * 0.00000222

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .uv_pip_install(  # pins mirror requirements.lock
        "torch==2.14.1", "torchvision==0.29.1", "transformers==5.20.1", "datasets==5.1.0",
        "huggingface-hub==2.2.0", "timm==1.0.30", "pillow==13.0.0", "numpy==2.5.4", "scikit-learn==1.9.1",
        "pyyaml==6.0.3", "opencv-python-headless==5.0.0.93", "safetensors==0.8.0", "tqdm==4.70.1",
    )
    .env({"PYTHONPATH": f"{REMOTE}/src", "HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(PROJECT / "src", f"{REMOTE}/src")
    .add_local_dir(PROJECT / "scripts", f"{REMOTE}/scripts")
    .add_local_dir(PROJECT / "configs", f"{REMOTE}/configs")
)
app = modal.App("dfd-logo", image=image)
hf_cache = modal.Volume.from_name("dfd-hf-cache", create_if_missing=True)
runs_vol = modal.Volume.from_name("dfd-runs", create_if_missing=True)


def _hf_secret() -> list:
    """Ephemeral secret carrying ONLY HF_TOKEN (needed for gated DINOv3). Read locally from env or .env."""
    token = os.environ.get("HF_TOKEN", "")
    env_file = PROJECT / ".env"
    if not token and modal.is_local() and env_file.is_file():
        for line in env_file.read_text().splitlines():
            if line.strip().startswith("HF_TOKEN="):
                token = line.split("=", 1)[1].strip().strip("\"'")
    return [modal.Secret.from_dict({"HF_TOKEN": token})] if token else []


@app.function(gpu=GPU, cpu=CPU, memory=MEM_MB, timeout=TIMEOUT_S, max_containers=MAX_PARALLEL,
              volumes={"/cache": hf_cache, "/runs": runs_vol}, secrets=_hf_secret())
def train_rotation(selection: str, test: str, backbone: str, epochs: int, tag: str, base_config: str) -> dict:
    import subprocess

    import yaml

    t0 = time.time()
    cfg = yaml.safe_load(Path(f"{REMOTE}/{base_config}").read_text())
    name = f"{selection}-to-{test}"
    out_dir = f"/runs/{tag}/{name}"
    cfg["model"]["backbone"] = backbone
    cfg["data"].update(selection_generators=[selection], holdout_generators=[test], num_workers=6)
    cfg["train"].update(epochs=epochs, output_dir=out_dir)
    cfg_path = Path(f"/tmp/{name}.yaml")
    cfg_path.write_text(yaml.safe_dump(cfg))
    proc = subprocess.run(["python", f"{REMOTE}/scripts/train.py", "--config", str(cfg_path)],
                          capture_output=True, text=True)
    hf_cache.commit()  # persist downloaded datasets/weights for later rotations
    runs_vol.commit()
    wall = time.time() - t0
    result = {"rotation": name, "selection": selection, "test": test, "backbone": backbone, "epochs": epochs,
              "returncode": proc.returncode, "wall_s": round(wall, 1), "est_cost_usd": round(wall * PRICE_PER_S, 3),
              "log_tail": (proc.stdout + proc.stderr)[-4000:]}
    report = Path(out_dir) / "report.json"
    if report.exists():
        result["report"] = json.loads(report.read_text())
    return result


def _rotations(spec: str) -> list[tuple[str, str]]:
    """'all' -> each generator is the test once, selection = the next generator in the cycle."""
    if spec == "all":
        return [(GENERATORS[(i + 1) % len(GENERATORS)], g) for i, g in enumerate(GENERATORS)]
    pairs = [tuple(p.split(":")) for p in spec.split(",")]
    for sel, test in pairs:
        if sel == test or {sel, test} - set(GENERATORS):
            raise SystemExit(f"Bad rotation {sel}:{test}; generators are {GENERATORS}")
    return pairs


@app.local_entrypoint()
def main(rotations: str = "GLIDE:Midjourney", backbone: str = "facebook/dinov2-base", epochs: int = 2,
         tag: str = "", base_config: str = "configs/tiny_genimage_nested.yaml", warm_first: bool = True):
    pairs = _rotations(rotations)
    tag = tag or f"{backbone.split('/')[-1]}_e{epochs}_{time.strftime('%Y%m%d-%H%M%S')}"
    print(f"[modal] {len(pairs)} rotation(s), backbone={backbone}, epochs={epochs}, tag={tag}, "
          f"~${PRICE_PER_S * 3600:.2f}/GPU-hour list")
    args = [(s, t, backbone, epochs, tag, base_config) for s, t in pairs]
    results = []
    if warm_first and len(args) > 1:
        # First job downloads data/weights into the shared cache; the rest then read it instead of racing.
        results.append(train_rotation.remote(*args[0]))
        args = args[1:]
    results += list(train_rotation.starmap(args, return_exceptions=True))

    local = PROJECT / "runs" / "modal" / tag
    local.mkdir(parents=True, exist_ok=True)
    total = 0.0
    for r in results:
        if isinstance(r, Exception):
            print(f"[modal] job failed: {type(r).__name__}: {r}")
            continue
        total += r["est_cost_usd"]
        (local / f"{r['rotation']}.json").write_text(json.dumps(r, indent=2))
        rep = r.get("report") or {}
        t = (rep.get("eval_test") or {}).get("overall", {})
        print(f"[{r['rotation']}] rc={r['returncode']} wall={r['wall_s']/60:.1f}min ~${r['est_cost_usd']:.2f} "
              f"test AUC={t.get('auc')} TPR@1%FPR={t.get('tpr@1%fpr')}")
        if r["returncode"] != 0:
            print(r["log_tail"][-1500:])
    print(f"[modal] estimated total ${total:.2f} (list price; actual in Modal dashboard). Results: {local}")
