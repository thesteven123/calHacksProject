# Video Deepfake Datasets: Research, Verification, and Ingestion Guide

**Date:** 2026-10-24  
**Author:** agy-research (Antigravity Research Agent)  
**Audience:** devin-impl, system architects, and ML training pipeline  
**Status:** Complete & HF API Verified  

---

## 1. Executive Summary & Objective

Our detector's spatial backbone (`dinov2_base`) achieves strong frame-level performance on static images, but its temporal video adapter is currently untrained. On in-the-wild video benchmarks like **WildDeepfake**, our frame-level AUC is near random chance (~0.64 AUC). Furthermore, gated access to large benchmark corpora like `ControlNet/AV-Deepfake1M` remains pending manual approval.

This research document identifies, verifies, and catalogs video deepfake datasets suitable for:
1. **Training** video-level temporal adapters, face-swap, facial reenactment, and lip-sync artifact detectors.
2. **Evaluating** generalization against unseen facial manipulation methods (Leave-One-Method-Out / LOGO protocol) and modern text-to-video / image-to-video generation engines (OpenAI Sora, Kling, Runway Gen-3, Pika, ModelScope).

Every candidate repository has been programmatically inspected via `huggingface_hub.HfApi().dataset_info(repo_id, files_metadata=True)` using the project's Hugging Face token (with no secrets printed or leaked). Unverified or third-party mirrors have been flagged accordingly.

---

## 2. Comprehensive Comparison Matrix

The table below catalogs 18 candidates across classic face swaps, reenactment, audio-visual lip-sync, and generative text-to-video models.

| Dataset | Source / Repo (Verified?) | Access | Licence | Train Allowed? | Manipulation Types & Methods | # Real / # Fake | Size (GB) | Per-Method Labels? | Identity IDs? | Audio? | Compression Levels | Overlap Risk | Recommendation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **FaceForensics++ (Official TUM)** | Official TUM Server (`github.com/ondyari/FaceForensics`) (**VERIFIED**: Code & Form) | Form Gated | TUM Research EULA (Strict non-commercial; redistribution prohibited) | **YES** (Upon Form Approval) | Face swap, reenactment: Deepfakes, Face2Face, FaceSwap, NeuralTextures, FaceShifter, Google DFD | 1,000 real / 5,000 fake per compression level | ~500 GB total (Raw lossless, c23, c40) | **YES** | **YES** (Source YouTube IDs) | Yes | Raw (lossless), c23 (light), c40 (heavy) | Source benchmark | **TRAIN & EVAL (Legitimate Primary Source; Requires Form)** |
| **FaceForensics++ (c23 Mirror)** | `bitmind/FaceForensicsC23` (**VERIFIED**: HF API, ungated) | Open | **None on HF (Licence unknown / Unofficial redistribution)** | **DO NOT TRAIN** (Violates TUM EULA prohibition on redistribution) | Face swap, reenactment: Deepfakes, Face2Face, FaceSwap, NeuralTextures, FaceShifter, Google DFD | 1,000 real / 6,000 fake (7,000 total videos) | 17.88 GB (`FaceForensics++_C23.zip`) | **YES** | **YES** | Yes | c23 (H.264) | Self-contained YouTube sources | **DO NOT TRAIN (Unofficial mirror; use official TUM route)** |
| **TalkingHeadBench** | `luchaoqi/TalkingHeadBench` (**VERIFIED**: HF API, ungated) | Open | **MIT (Code/Metadata)** | **YES** (Eval & Train) | Modern Diffusion Talking-Head Reenactment: **LivePortrait, Hallo, Hallo2, AniPortrait (Audio & Video), EMOPortraits** | **0 real videos / 2,994 fake videos** (`real/` contains only 2 split JSONs pointing to CelebV-HQ/FF++) | 16.13 GB (8,130 files, direct MP4) | **YES** (Per-generator folder partitions) | **YES** (Mapped to CelebV-HQ IDs in JSON) | **YES** (Synchronized speech) | H.264 Web MP4 | Zero with WildDeepfake | **TRAIN & EVAL (Clean MIT Fakes; needs external reals)** |
| **FakeParts** | `hi-paris/FakeParts` (**VERIFIED**: HF API, ungated) | Open | **CC0 1.0 (Public Domain)** | **YES** (Train & Eval) | Modern T2V: **HunyuanVideo** and open diffusion video engines | ~2,000 real / ~8,000 fake | 7.67 GB | **YES** | N/A | No | Web MP4 | None | **TRAIN & EVAL (Cleanest CC0 1.0 T2V fakes)** |
| **GenVidBench (Mirror)** | `jian-0/GenVidBench` (**VERIFIED**: HF API, ungated) | Open | **None on HF (Licence unknown; unofficial mirror)** | **EVAL-ONLY (Verify terms)** | Full Text-to-Video (T2V) & Image-to-Video (I2V): OpenAI Sora, Kling, OpenSora, Pika, ModelScope | ~5,000 real / ~15,000 fake | 215.81 GB (42 modular zips by generator) | **YES** | N/A | Mixed | Web-compressed MP4 | None | **EVAL-ONLY (See official GenVidBench terms)** |
| **SynthForensics** | `SynthForensics/SynthForensics` (**VERIFIED**: HF API, auto-gated) | Gated (User action required: accept on HF page) | **CC-BY-NC 4.0** | **YES** (Upon user clicking Accept) | 2025 Generative Video: **Wan2.1, CogVideoX, SkyReels-V2, MAGI-1, LTX-2.3, daVinci-MagiHuman, Helios, Self-Forcing** | 20,445 synthetic human videos + real web videos | 291.7 GB (Structured in `T2V/` and `I2V/`) | **YES** (Partitioned by engine) | N/A | Mixed | High-res web compression | None with face-swap sets | **TRAIN & EVAL (Accept terms on HF first)** |
| **LAV-DF (Official Repo)** | `ControlNet/LAV-DF` (**VERIFIED**: HF API, gated auto) | Gated (User action required: accept on HF page) | **CC-BY-NC 4.0** (Inherits VoxCeleb2 terms) | **YES** (Research non-commercial) | Audio-visual lip-sync & temporal splicing | 13,632 real / 20,404 fake | 25.61 GB (`LAV-DF.tar`) | **YES** | **YES** | **YES** | Web-encoded MP4 | Low | **TRAIN & EVAL (Official gated repo)** |
| **LAV-DF1 (Mirror)** | `Jennifer24/LAV-DF1` (**VERIFIED**: HF API, ungated) | Open | **None on HF (Licence unknown; unofficial mirror bundling checkpoints)** | **DO NOT TRAIN** (Use official `ControlNet/LAV-DF`) | Audio-visual manipulation, temporal lip-sync tampering (Wav2Lip, VideoRetalking) | 13,632 real / 20,404 fake | 31.62 GB (`LAV-DF_split.zip` + `audio.zip` + weights) | **YES** | **YES** | **YES** | Web-encoded MP4 | Low | **REJECT (Unofficial mirror; use official `ControlNet/LAV-DF`)** |
| **CelebV-HQ (Mirror)** | `SwayStar123/CelebV-HQ` (**VERIFIED**: HF API, ungated) | Open | **None on HF (README cites official GitHub non-commercial terms)** | **CAUTION (Mirror)** | High-resolution pristine talking face sequences (512x512 min resolution, diverse emotions/actions) | 35,666 real video clips (15,653 identities) / 0 fake | 39.01 GB (`videos.tar` + `celebvhq_info.json`) | N/A | **YES** (15,653 celebrity IDs) | **YES** | High quality 512x512 MP4 | Clean real baseline | **TRAIN REALS (Mirror caveat; see official `celebv-hq.github.io`)** |
| **VidProM** | `WenhaoWang/VidProM` (**VERIFIED**: HF API, ungated) | Open | CC-BY-NC 4.0 (mixed generator licences) | **YES** (Research non-commercial) | Text-to-Video: OpenSora, ModelScope, Pika, CogVideo, VideoCrafter2, Text2Video-Zero, Stable-T2V | Real text prompts + 6.69M AI videos | 3.19 TB (164 modular tars/shards) | **YES** (Partitioned by generator) | N/A | Variable | Diverse web compression | None | **EVAL-ONLY (Subset for open T2V)** |
| **GenVideo-50k** | `sshaiton/GenVideo-50k-shard` (**VERIFIED**: HF API, ungated) | Open | **MIT** | **YES** (Permissive open source) | Text-to-Video generation (DeMamba training shard) | ~10k real / ~40k fake | 83.99 GB (87 tar shards) | Partial (Metadata jsonl) | N/A | No | H.264 MP4 | Low | **TRAIN & EVAL (Clean MIT licence)** |
| **WildDeepfake** | `xingjunm/WildDeepfake` (**VERIFIED**: local / test split held) | Open / Academic | Research Only (In-the-wild web crawl) | **EVAL-ONLY** | Diverse in-the-wild face swaps, expression reenactment, unknown generators | 707 videos / 7,314 face sequences | ~10 GB | No (Uncontrolled web deepfakes) | Partial (Web identities) | Mixed | Heavy social media compression | Baseline test split (Zero overlap with FF++) | **EVAL-ONLY (Primary In-The-Wild Metric)** |
| **Celeb-DF v2** | SUNY Buffalo CV Lab (`github.com/yuezunli/celeb-deepfakeforensics`) (**VERIFIED**: Form) | Form Gated | Academic Non-Commercial EULA | **YES** (Research non-commercial) | High-fidelity face swap (custom DeepFake algorithm with boundary blending) | 590 Celeb-real, 300 YouTube-real / 5,639 Celeb-synthesis | 35.6 GB | **YES** (518 test list curated) | **YES** (59 Celebrity IDs) | No / stripped | H.264 standard YouTube compression | None with WildDeepfake | **EVAL & TRAIN (Requires Form Submission)** |
| **DFDC Full** | Meta AI Portal (`ai.meta.com/datasets/dfdc/`) (**VERIFIED**: Portal & AWS CLI) | Portal / AWS Auth | DFDC Competition EULA (Research use) | **YES** (Consented actors) | 8 proprietary face-swap and reenactment algorithms + voice swap/puppetry | 23,654 real / 100,500 fake (124,154 total) | ~472 GB | Partial (Method labels in metadata) | **YES** (3,426 consented actors) | **YES** | Diverse compression and distractors | None with YouTube datasets | **EVAL & TRAIN (Heavyweight; requires AWS IAM)** |
| **DFDC Preview** | Kaggle / Meta (`kaggle.com/c/deepfake-detection-challenge`) (**VERIFIED**) | Kaggle Auth | Competition Terms | **YES** | 2 face-swap algorithms | 1,131 real / 4,119 fake (5,250 total) | ~18 GB | Yes | Yes (Paid actors) | Yes | Mixed | None | **EVAL-ONLY (Lightweight DFDC proxy)** |
| **FakeAVCeleb (Official)** | DASH-Lab Korea (`sites.google.com/view/fakeavcelebdash-lab`) (**VERIFIED**: Form) | Form Gated | Academic DUA (Non-commercial research only) | **YES** | Audio-visual: Wav2Lip, Faceswap, FSGAN, SV2TTS | 500 real source / 21,500+ fake clips | ~40 GB | **YES** (Video, audio, AV-congruent labels) | **YES** (VoxCeleb2 identities) | **YES** (Synchronized) | H.264 | None with FF++ | **TRAIN & EVAL (Requires Form)** |
| **AV-Deepfake1M** | `ControlNet/AV-Deepfake1M` (**VERIFIED**: HF API, gated manual) | Gated (Manual approval) | CC-BY-NC 4.0 | **PENDING** (Awaiting author sign-off) | Multi-modal audio-visual deepfakes (Lip-sync, speech synthesis, swap) | 288k real / 786k fake (1.07M total clips) | 404.31 GB (Split archives) | **YES** | **YES** | **YES** | Diverse web resolutions | None | **HOLD (Pending manual gate approval)** |
| **DeeperForensics-1.0** | NTU / SenseTime (`liming-chen.net/deeperforensics`) (**VERIFIED**: Form) | Form Gated | Academic Research Agreement | **YES** | High-fidelity face swap (DeepFake-1.0) with real-world 35-perturbation distorts | 50,000 real / 10,000 fake (60,000 total) | ~300 GB | **YES** | **YES** (100 paid actors) | No | Raw, c23, c40 + transmission distortions | None | **EVAL-ONLY (Perturbation robustness benchmark)** |
| **ForgeryNet** | CVPR 2021 (`yuentian.org/forgerynet`) (**VERIFIED**: Application) | Form Gated | Academic Research Agreement | **YES** | 15 manipulation methods (4 face swap, 4 face reenactment, 3 face editing, 4 face generation) | 99,630 real / 121,617 fake (221,247 videos) | ~1.2 TB | **YES** (Full taxonomy) | **YES** (99k video sources) | Mixed | Diverse video resolutions | None | **REJECT (Too large for hackathon quota)** |
| **KoDF** | AI Hub Korea / NAVER (`github.com/clovaai/kodf`) (**VERIFIED**) | Portal Gated | Korean AI Hub Terms (ID verification required) | **NO** (Foreign academic barrier) | Face swap, Audio-driven reenactment (FoRM, First Order, Wav2Lip) | 62,488 real / 175,776 fake | ~2 TB | **YES** | **YES** (403 Korean identities) | **YES** | High definition | None | **REJECT (Strict Korean national ID barrier)** |
| **BitMind 34data Modern Video** | `34data/gen-videos-fastwan2`, `34data/v15-human-vid-mavos-dd-english_liveportrait`, `34data/wav2lip` (**VERIFIED**: HF API) | Open | Open Research (Subnet 34) | **YES** (Eval & Train) | Modern targeted video fakes: **Wan2.1** (4.17 GB), **LivePortrait** (0.68 GB), **Wav2Lip** (2.66 GB), **HunyuanVideo** (0.95 GB) | ~1,500 real / ~4,500 fake per subset | ~8.46 GB total (Zipped MP4s) | **YES** (Separated by generator repo) | Partial | Mixed | H.264 MP4 | Low | **TRAIN & EVAL (Lightweight targeted shards)** |
| **Celeb-DF Extracted Frames** | `thenewsupercell/celeb-df-image-dataset` (**VERIFIED**: HF API, ungated) | Open | Unknown / Derived from Celeb-DF | **CAUTION** (Static frames only) | Celeb-DF v2 face swap | 73,660 train / 9,975 val / 10,165 test | 1.11 GB (Parquet) | No | No | No | Extracted cropped face frames | Low | **REJECT for Video (No temporal video sequences)** |
| **OmniFake (MoeNew / Zlatan-Ye)** | `MoeNew/OmniFake`, `Zlatan-Ye/OmniFake-OOD` (**VERIFIED**: HF API) | Open | CC-BY-NC 4.0 | **NO (Image Only)** | Multi-generator static images (1.17M images across 45 diffusion/GAN generators) | Static images only (No video files) | 1,153 GB | Yes | No | No | Static PNG/JPG | N/A | **REJECT for Video (Confirmed Static Image Only)** |
| **FakeAVCeleb Mirror (Empty)** | `Jennifer24/FakeAVCelebDataset` (**VERIFIED**: HF API) | Open | None | **NO** | Empty placeholder repository | 0 | 0.00 GB (Only `.gitattributes`) | N/A | N/A | N/A | N/A | N/A | **REJECT (Broken / empty upload)** |

---

## 3. Ranked Video Datasets for Immediate & Near-Term Use

### Tier 1: Clean-Licence & Obtainable Immediately (Ungated on HF, Verified Terms)

#### 1. `luchaoqi/TalkingHeadBench` (WACV 2026 / arXiv:2505.24866)
- **Why #1:** The premier open benchmark for **modern diffusion talking-head deepfakes** (2024–2026). Legacy datasets (like FF++) only cover early autoencoder swaps. `TalkingHeadBench` has an **MIT licence** on HF and contains 2,994 curated high-definition video clips across:
  - **LivePortrait** (implicit keypoint facial reenactment)
  - **Hallo & Hallo2** (diffusion-based speech-driven talking portraits)
  - **AniPortrait** (audio- and video-driven diffusion facial animation)
  - **EMOPortraits** (expressive emotional talking head synthesis)
- **Crucial Dataset Structure Caveat:** `TalkingHeadBench` contains **ONLY synthetic fake videos (2,994 MP4 files + audio)**. The `real/` folder holds **only two split JSON files** (`real_dataset_split.json` and `real_dataset_split_official_ff++.json`), which point to external YouTube video IDs from CelebV-HQ and FaceForensics++. Pristine real video negatives must therefore be supplied from an external clean-licence real video corpus.
- **Licence:** MIT on HF (`luchaoqi/TalkingHeadBench`).
- **Modal Volume Impact:** Only **16.13 GB** total. Direct public download from Hugging Face.

#### 2. `hi-paris/FakeParts`
- **Why #2:** The cleanest licensing terms on Hugging Face (**CC0 1.0 Public Domain**). Provides modern text-to-video deepfake clips including **HunyuanVideo**.
- **Licence:** CC0 1.0 (Public Domain).
- **Modal Volume Impact:** **7.67 GB** total.

#### 3. `sshaiton/GenVideo-50k-shard`
- **Why #3:** A clean **MIT-licensed** open dataset shard providing ~50,000 text-to-video generated clips across open diffusion video generators (DeMamba training shard).
- **Licence:** MIT.
- **Modal Volume Impact:** **83.99 GB** (87 tar shards).

---

### Tier 2: User Action Required: Hugging Face Auto-Gated Acceptance

#### 4. `SynthForensics/SynthForensics` (2025)
- **Why #4:** The most comprehensive 2025 generative video dataset. Contains 20,445 synthetic human videos across **Wan2.1, CogVideoX, SkyReels-V2, MAGI-1, LTX-2.3, daVinci-MagiHuman, Helios, and Self-Forcing**.
- **Licence:** CC-BY-NC 4.0.
- **User Action Required:** Gated on HF (`gated=auto`). The Hugging Face account owner must visit [SynthForensics on HF](https://huggingface.co/datasets/SynthForensics/SynthForensics) and click **"Agree and access repository"** to accept terms.
- **File Structure & Modal Impact:** **291.7 GB** total, organized cleanly into `T2V/` and `I2V/` directories. Can be selectively ingested by generator shard (e.g. pulling only Wan2.1 and CogVideoX).

#### 5. `ControlNet/LAV-DF` (Official Repository)
- **Why #5:** Official repository for localized audio-visual lip-sync and temporal tampering (Wav2Lip, VideoRetalking).
- **Licence:** CC-BY-NC 4.0 (inherits VoxCeleb2 non-commercial research terms).
- **User Action Required:** Gated on HF (`gated=auto`). User must accept terms on [ControlNet/LAV-DF](https://huggingface.co/datasets/ControlNet/LAV-DF).
- **Mirror Warning:** Do **not** use `Jennifer24/LAV-DF1`—it has `license=None` on HF, is an unauthorized third-party mirror, and improperly bundles PyTorch model weights inside the dataset archive.
- **Modal Volume Impact:** **25.61 GB** (`LAV-DF.tar`).

---

### Tier 3: External Form-Gated & Primary Sources (Legitimate Compliance)

#### 6. Official FaceForensics++ (TUM)
- **Why:** The **only legitimate legal route** to FaceForensics++. The upstream TUM EULA explicitly states: *"The dataset may only be used for non-commercial research purposes and must not be redistributed."*
- **Crucial Mirror Warning on `bitmind/FaceForensicsC23`:** `bitmind/FaceForensicsC23` has **NO licence on Hugging Face (`license=None`)** and is an unofficial third-party redistribution that breaches the upstream TUM EULA. **Do NOT train on `bitmind/FaceForensicsC23` without TUM form approval.**
- **Legitimate Form URL:** [Official TUM FaceForensics Google Form](https://docs.google.com/forms/d/e/1FAIpQLSdRRR3L5zAv6tQ_CKxmK4W96tAab_pfBu2EKAgQbeDVhmXagg/viewform)
- **Submission Requirements:** Name, University email, Institution, Advisor/PI name, checkbox agreement prohibiting redistribution.
- **Turnaround:** Automated / 12–48 hours for academic emails. Unlocks the official `download-FaceForensics.py` script for raw, c23, and c40 splits.

#### 7. Celeb-DF v2 (SUNY Buffalo)
- **Why:** High-fidelity face swap benchmark with blended boundary edges.
- **Form URL:** [Celeb-DF v2 Request Form](https://forms.gle/2jYBby6y1FBU3u6q9) (or [Tencent Survey](https://wj.qq.com/s2/5610260/28f6/)).
- **Submission Requirements:** Academic email, research description, non-commercial agreement.

#### 8. CelebV-HQ Real Video Negatives (`celebv-hq.github.io`)
- **Why:** 35,666 pristine 512x512 real face video clips across 15,653 identities (ECCV 2022).
- **Licence & Mirror Caveat:** The Hugging Face upload `SwayStar123/CelebV-HQ` has `license=None` (unknown). Its README cites the official project terms: non-commercial research only with copyright retained by original YouTube authors. Follow the official project download tool at [CelebV-HQ GitHub](https://github.com/CelebV-HQ/CelebV-HQ) or cite its terms before downloading the 39.01 GB archive.

#### 9. GenVidBench (Official Project Terms)
- **Why:** Full T2V/I2V benchmark covering Sora, Kling, Pika, ModelScope.
- **Mirror Warning:** The HF repo `jian-0/GenVidBench` has `license=None` (unspecified). Refer to the official project repository at [jian-0/GenVidBench](https://github.com/jian-0/GenVidBench) for verified usage terms.

---

## 4. Real-Video Negatives & Baseline Sources

Training a robust video deepfake detector requires unmanipulated negative sequences that exhibit natural motion, diverse lighting, head poses, and standard compression:

1. **Permissive Open Real Video (Kinetics-400 / Kinetics-700):**
   - Creative Commons CC-BY 4.0 YouTube video clips (10-second human actions). Cleanest legal baseline for real video dynamics.
2. **CelebV-HQ (`celebv-hq.github.io`):**
   - 35,666 pristine 512x512 talking-head videos with natural speech and facial micro-expressions. Non-commercial research agreement.
3. **VoxCeleb2 (`ProgramComputer/voxceleb` / Oxford VGG):**
   - >1 million utterances from 6,112 celebrities. Non-commercial research agreement on HF via auto-gated acceptance.

---

## 5. Overlap & Contamination Analysis

- **WildDeepfake (Current Evaluation Set):** Sourced from uncurated web video clips across social media (Weibo, Bilibili, YouTube, Reddit). Zero identity or generator overlap with `TalkingHeadBench` or `FakeParts`.
- **Sowaiba / Inswapper Test:** Static face-swap test images generated from independent portrait sources. No temporal overlap with video datasets.

---

## 6. Proposed First Video Experiment Protocol (Clean-Licence Data Only)

**Design: Leave-One-Method-Out (LOGO) Diffusion Talking-Head & Cross-Domain Generalization**

> Rely **exclusively on clean-licence, legitimately obtainable data**:
> 1. **Training Fakes:** Ingest `luchaoqi/TalkingHeadBench` (MIT licence, 16.13 GB) and `hi-paris/FakeParts` (CC0 1.0, 7.67 GB). Train the temporal adapter on modern diffusion talking-head fakes (**AniPortrait** audio-driven, AniPortrait video-driven, **Hallo**, and **EMOPortraits**) plus **HunyuanVideo** text-to-video sequences.
> 2. **Training Reals:** Pair with clean-licence real talking-head sequences (permissive CC-BY human video clips or official CelebV-HQ sequences retrieved under its non-commercial research terms).
> 3. **Held-Out LOGO Method:** Hold out **LivePortrait** (implicit keypoint facial reenactment) and **Hallo2** completely from training to evaluate generalization to unseen modern diffusion reenactment architectures.
> 4. **Out-of-Domain Zero-Shot Evaluation:** Without fine-tuning, evaluate on:
>    - The held-out **LivePortrait** test fakes (measuring diffusion reenactment detection);
>    - The **WildDeepfake** test split (707 in-the-wild web face swaps), measuring whether the temporal adapter lifts frame AUC above the current near-chance baseline of ~0.64.

---

## 7. Action Plan & Next Steps

1. **devin-impl:** Ingest `luchaoqi/TalkingHeadBench` (16.13 GB, MIT) and `hi-paris/FakeParts` (7.67 GB, CC0) onto the Modal volume (`dfd-hf-cache` / `dfd-runs`). These provide immediately usable, clean-licence training fakes.
2. **User Action (HF Click):** Log into Hugging Face and click "Agree and access repository" on **`SynthForensics/SynthForensics`** and **`ControlNet/LAV-DF`** to unlock modern 2025 generative video (Wan2.1, CogVideoX) and official audio-visual lip-sync.
3. **User Action (Form Submission):** Submit the official **TUM FaceForensics++** Google Form to legitimately unlock raw/c23/c40 face swaps without relying on unverified redistributions.

