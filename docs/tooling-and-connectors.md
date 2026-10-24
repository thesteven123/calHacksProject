# Tooling, MCPs, Skills & Connectors Recommendation

To support end-to-end deepfake research, prototyping, and production deployment, the following tools, Model Context Protocol (MCP) servers, custom skills, and connectors are recommended.

---

## 1. Model Context Protocol (MCP) Servers

| MCP Server | Primary Purpose | How It Accelerates the Workflow |
|---|---|---|
| **`arxiv` MCP** | Academic Literature Retrieval | Query and download latest preprints on deepfake forensics, video diffusion generation (Sora/Kling/Wan), and adversarial robustness. |
| **`gemini-cli` MCP** | Long-context Document & Paper Critique | Summarize 40+ page papers, review mathematical formulations (e.g. NSG probability flow), and conduct adversarial red-teaming. |
| **`huggingface` MCP** | Model & Dataset Discovery | Query weights, model cards, and datasets (e.g., DINOv2, SigLIP-2, Omni-Fake, GenImage) directly from Hugging Face Hub. |
| **`github` MCP** | Open-Source Code Exploration | Clone, explore, and track implementations of reference models (UnivFD, FatFormer, HEDGE, SBI). |
| **`filesystem` / `fetch` MCP** | Local Artifacts & Web Fetching | Ingest CVPR/NeurIPS conference open-access proceedings and inspect local datasets. |

---

## 2. Recommended Custom Skills

### A. `compression-emulator` Skill
- **Scope:** Automates multi-codec social media compression emulation on images and video using `ffmpeg` and `pillow`.
- **Capabilities:**
  - Emulates TikTok, Instagram Reels, and X transcoding profiles (H.264/H.265/AV1 with standard CRF and bitrate ladders).
  - Simulates multi-hop re-encoding chains (e.g., WhatsApp downsampling -> CapCut edit -> Instagram upload).
  - Injects synthetic camera beauty filter smoothing, bilateral filters, and noise reduction.

### B. `tensorrt-latency-benchmark` Skill
- **Scope:** Benchmarking model latency and throughput under production conditions.
- **Capabilities:**
  - Converts PyTorch / ONNX models to TensorRT engines with FP16, FP8, and INT8 calibration.
  - Profiles P50, P95, and P99 latency per batch size on NVIDIA T4, L4, and A100 GPUs.
  - Verifies that Stage 1 stays < 5ms and Stage 2 stays < 35ms per video item.

### C. `deepfake-forensic-eval` Skill
- **Scope:** Automated benchmarking on in-the-wild test suites.
- **Capabilities:**
  - Computes AUC, Log-Loss, and operational moderation metrics (TPR @ 1.0% FPR, TPR @ 0.1% FPR).
  - Performs slice-based evaluation across demographics (DAID), codecs, resolutions, and generator types.

---

## 3. Production & Pipeline Connectors

1. **C2PA / Content Credentials Connector:**
   - Extracts cryptographically signed provenance manifests (JUMBF metadata) at Stage 0 to fast-path authentic verified publisher media or identify signed AI watermarks.
2. **Video Ingest & Keyframe Decoding Connector:**
   - Connects to Kafka / AWS Kinesis video streaming buffers; uses hardware-accelerated decoders (NVDEC or Apple VideoToolbox) to sparsely sample 3–5 keyframes/sec and isolate face tracks.
3. **Weights & Biases / MLflow Connector:**
   - Tracks training runs, cross-generator generalization curves, learning rates, and information bottleneck KL divergence.
4. **Moderation Case Management & Escalation Connector:**
   - Feeds Stage 3 borderline predictions into the trust-and-safety review queue with highlighted tampering heatmaps (bounding box + localized attribution mask) and calibrated confidence intervals.

---

## 4. Implementation-phase requests (Devin, 2026-10-24)

Status: no MCP servers are configured in this workspace yet. The following are ordered by impact on the build.

| Need | Type | Why it is blocking or valuable |
|---|---|---|
| `TINKER_API_KEY` | secret / env var | **Done (2026-10-24):** verified via `scripts/check_credentials.py` (35 models; route default supported). |
| `HF_TOKEN` + accepting the DINOv3 and AV-Deepfake1M gates | secret + manual gate | **Token valid; DINOv3 granted and loading; AV-Deepfake1M still pending.** |
| **CUDA GPU compute: Modal** (decided 2026-10-24; Google Cloud and Azure GPU access unavailable) | connector | Full-dataset training with on-the-fly PTE, plus T4/L4 latency benchmarks. Starter plan, $30/mo credit, $150 grant pending. **Set up and verified 2026-10-24** (`scripts/modal_logo.py`; 14 LOGO jobs ≈ $0.53). Login: `modal token new` (token stays in `~/.modal.toml`). **Tinker** ($150 credit) covers only the VLM route. MPS is fine for subsets only. |
| **`huggingface` MCP** | MCP | Dataset/model discovery and card/licence checks without ad-hoc curl calls |
| **Weights & Biases** (`WANDB_API_KEY`) | connector | Track LOGO rotations and β/LoRA ablations; the trainer already writes `metrics.jsonl` |
| `ffmpeg` (`brew install ffmpeg`) | system tool | Video codec chains (H.264/HEVC/AV1) for the S4 video PTE |
| **`github` MCP** + a git remote | MCP | Reference implementations (SBI, B-Free, HEDGE), plus versioning this repo (it isn't a git repo yet) |
| Access requests: Deepfake-Eval-2024, TrueFake, SIDA, Omni-Fake | human action | The only credible in-the-wild evaluation, held out from training |
| Skill: `dfd-experiment` (`.devin/skills/`) | skill | One command to run train, the severity sweep and a §11 log entry, for consistent ablations |
