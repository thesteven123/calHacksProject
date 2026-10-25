# Open-Access Deepfake & Synthetic Image Datasets

**Author:** agy-research  
**Date:** 2026-10-24  
**Purpose:** Survey of publicly accessible, non-gated datasets suitable for training / evaluating the DFD core detector.  
All datasets were verified via `https://huggingface.co/api/datasets/<id>` unless a non-HF source is noted.  
Status key: ✅ VERIFIED PUBLIC · ⚠️ PUBLIC BUT CAVEATED · ❌ GATED/INACCESSIBLE

---

## Priority Tier 1 — Drop-in, HF-native, No Gating

These datasets can be streamed directly with `load_dataset()` on Modal/HF with no login or approval step.

---

### 1. `nebula/GenImage-arrow` ✅ VERIFIED PUBLIC
- **Link:** https://huggingface.co/datasets/nebula/GenImage-arrow
- **Licence:** CC-BY-NC-SA 4.0 (research only, same as `TheKernel01/Tiny-GenImage`)
- **Size:** ~1 M real/fake image pairs in Arrow format (~160 GB full; per-generator configs allow partial loading)
- **Generators (fakes):** ADM · BigGAN · GLIDE · Midjourney · Stable Diffusion v1.4 · Stable Diffusion v1.5 · VQDM · Wukong
- **Real images:** ImageNet-matched crops (same class labels as fakes)
- **Format:** HF Arrow shards; each generator is a named `config_name` (`adm`, `biggan`, `glide`, `midjourney`, `sd14`, `sd15`, `vqdm`, `wukong`). Also `all-train` / `all-test` configs for cross-generator training.
- **Splits:** `train`, `test`, `validation` per generator. `all-train` has ~1.1 M images.
- **Shortcut risk:** ⚠️ Fakes are square ImageNet prompts; reals are natural-aspect-ratio photos. **Must apply the same `format_align` centre-crop + JPEG re-encode as DeepFakeFace before mixing with face datasets.**
- **Generator metadata column:** Each row includes the generator label, enabling LOGO splits.
- **Relevance:** This IS the dataset already in use as `TheKernel01/Tiny-GenImage`. The `nebula/GenImage-arrow` repo is a more usable HF-native mirror with per-generator configs. Use it when you need the full 1 M scale.
- **Integration note for Devin:** Replace or supplement `TheKernel01/Tiny-GenImage` in configs by setting `hf_dataset: nebula/GenImage-arrow` and `hf_config: all-train` (or a specific generator config for ablations).

---

### 2. `JamieWithofs/Deepfake-and-real-images` ✅ VERIFIED PUBLIC
- **Link:** https://huggingface.co/datasets/JamieWithofs/Deepfake-and-real-images
- **Licence:** Not specified (community dataset)
- **Size:** ~1.79 GB; 140,002 train / 10,905 test / 39,428 validation
- **Format:** Parquet, single `image` + `label` column (`0=Fake`, `1=Real`)
- **Generators:** Unknown provenance — community-aggregated real vs. AI-generated faces (circa early 2024). No per-generator metadata column.
- **Splits:** Pre-split train/test/validation in Parquet.
- **Shortcut risk:** ⚠️ No per-generator labels → cannot do LOGO; unknown generator mix → treat as fixed held-out eval only, not training. Provenance undocumented; may contain format shortcuts.
- **Relevance:** Useful as an additional held-out binary-label evaluation set. Not suitable for nested LOGO training without generator metadata.
- **Integration note:** Load as `source: hf` with `hf_dataset: JamieWithofs/Deepfake-and-real-images`, `generator_column: null`. Treat as external eval only.

---

### 3. `Hemg/deepfake-and-real-images` ✅ VERIFIED PUBLIC
- **Link:** https://huggingface.co/datasets/Hemg/deepfake-and-real-images
- **Licence:** Not specified
- **Size:** ~1.79 GB; 190,335 train samples only
- **Format:** Parquet, `image` + `label` (`0=Fake`, `1=Real`)
- **Generators:** Unknown — very similar in structure to `JamieWithofs/`; likely same or overlapping source. No generator metadata.
- **Splits:** Train only (no pre-built test split).
- **Shortcut risk:** Same caveats as above — no LOGO, unknown provenance.
- **Relevance:** Extra training volume if Devin wants a quick additional source. Recommend using `JamieWithofs/` instead since it has test/validation splits.

---

### 4. `yashduhan/DeepFakeDetection` ✅ VERIFIED PUBLIC
- **Link:** https://huggingface.co/datasets/yashduhan/DeepFakeDetection
- **Licence:** Not specified
- **Size:** Single zip archive (`DeepFakeDetection.zip`); 112,000 train / 14,000 val / 14,000 test
- **Format:** ImageFolder-style zip; columns `image` + `label` (`real` / `fake`)
- **Generators:** Undocumented. 140k face images across diverse demographic profiles, balanced.
- **Splits:** Pre-split train/val/test.
- **Shortcut risk:** ⚠️ No per-generator column; unknown provenance.
- **Relevance:** Useful as an additional held-out binary eval. Same recommendation: external eval, not LOGO training.

---

### 5. `dragonintelligence/CIFAKE-image-dataset` ✅ VERIFIED PUBLIC
- **Link:** https://huggingface.co/datasets/dragonintelligence/CIFAKE-image-dataset
- **Paper:** Bird & Lotfi, "CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images," IEEE Access 2024. arXiv:2303.14126.
- **Licence:** Not specified on HF card (original Kaggle release is public domain/CC0)
- **Size:** 50 MB; 100,000 train + 20,000 test
- **Format:** Parquet; `image` + `label` (`REAL` / `FAKE`)
- **Generators (fakes):** Stable Diffusion v1.4 mirroring CIFAR-10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Real images:** CIFAR-10 (32×32 px)
- **Splits:** Train / Test
- **Shortcut risk:** ⚠️ Low-resolution (32×32). Conceptually clean (one generator, known provenance) but extremely small images — not suitable for face forensics or DINOv2 fine-tuning. Best used as a sanity-check or toy benchmark.
- **Relevance:** Low for this project; note for completeness.

---

## Priority Tier 2 — High-Value HF Datasets Needing Minor Setup

---

### 6. `xingjunm/WildDeepfake` ✅ VERIFIED PUBLIC
- **Link:** https://huggingface.co/datasets/xingjunm/WildDeepfake
- **Paper:** Zi et al., "WildDeepfake: A Challenging Real-World Dataset for Deepfake Detection," ACM MM 2020. arXiv:2101.01456.
- **Licence:** Apache-2.0 (confirmed in HF `cardData`)
- **Size:** ~72.8 GB; tags say `1M < n < 10M` (image count across face sequences)
- **Format:** WebDataset `.tar.gz` shards; 4 directories: `fake_train` (592 shards), `real_train` (371 shards), `fake_test` (115 shards), `real_test` (42 shards). Total 1,120 tar.gz files.
- **Dataset stats (paper):** 707 wild videos; 7,314 face sequences; scraped from internet — diverse, unconstrained lighting / compression / pose.
- **Generators:** **Unknown heterogeneous methods** — videos collected in the wild from the internet (DeepFaceLab, FaceSwap, and others). No per-generator labels. Generator identity is unavailable by design (in-the-wild).
- **Splits:** Pre-split fake_train/real_train/fake_test/real_test.
- **Shortcut risk:** Low — collected from real internet, highly diverse.
- **Relevance:** **Highly recommended as a cross-dataset OOD eval set.** Because generators are unknown/heterogeneous, this tests real-world generalisation. Cannot be used for LOGO. Best as the "internet deepfake" held-out split.
- **Integration note for Devin:** Download via `hf_hub_download` with `repo_type="dataset"` and pattern `"deepfake_in_the_wild/**/*.tar.gz"`. A WebDataset-compatible loader is needed. The `webdataset` Python library (already in `requirements.lock`?) streams tar shards without full extraction.

---

### 7. `bitmind/SyntheticFacesHQ` (SFHQ) ✅ VERIFIED PUBLIC
- **Link:** https://huggingface.co/datasets/bitmind/SyntheticFacesHQ
- **Original repo:** https://github.com/SelfishGene/SFHQ-dataset
- **Licence:** MIT (per GitHub) — fully open for research and commercial use
- **Size:** ~547,000 total 1024×1024 face images across 4 parts + T2I extension. HF files: `SFHQ-part1.zip` through `SFHQ-part4.zip` + `SyntheticFacesHQ-T2I.zip`
- **Format:** ImageFolder zip archives
- **Generators by part:**
  - Part 1 (~89.8k): Inspiration images → StyleGAN render (AAHQ / Close-Up Humans / UIBVFED)
  - Part 2 (~91.4k): Face Synthetics + Stable Diffusion v1.4 portraits
  - Part 3 (~118.4k): StyleGAN2 high-truncation sampling
  - Part 4 (~125.8k): Stable Diffusion v2.1 text-to-image
  - T2I ext. (~122.7k): **Flux.1 (Pro/Dev/Schnell) + SDXL + DALL-E 3** with diverse prompts
- **Real images:** None — fakes only. Must be paired with a real face set (e.g., `bitmind/bm-real` below)
- **Splits:** None pre-defined (single dump)
- **Shortcut risk:** ⚠️ All synthetic are 1024×1024 high-quality. Real set must be size-matched. Use `format_align` to standardize.
- **Relevance:** **Very high.** Covers the most modern generators (Flux.1 — what current social-media fakes use). T2I extension is especially valuable. Pair with `bitmind/bm-real` for a clean real/fake split.
- **Integration note for Devin:** Treat as FAKE source. Pair with `bitmind/bm-real` as REAL source. Create an HFImageDataset pointing to both repos, or a custom FolderDataset after downloading zips. No per-generator metadata column in HF; Devin may want to add a `generator` tag per part number.

---

### 8. `bitmind/bm-real` ✅ VERIFIED PUBLIC
- **Link:** https://huggingface.co/datasets/bitmind/bm-real
- **Licence:** "Other" (Bittensor Subnet 34 terms — research use permitted, no redistribution of derived models)
- **Size:** 6.71 GB; 28,393 real face images in Parquet (`train` split, 45 shards)
- **Format:** Parquet, single `image` column
- **Content:** Curated real-world face photographs (diverse demographics, poses, lighting conditions)
- **Splits:** Train only
- **Shortcut risk:** Low — real photographs with natural diversity
- **Relevance:** The natural partner to `bitmind/SyntheticFacesHQ`. Provides ~28k clean real faces. Much smaller than SFHQ — consider augmenting with `OpenRL/DeepFakeFace`'s `wiki.zip` (30k IMDB-WIKI reals) for scale balance.

---

## Priority Tier 3 — Non-HF, Accessible With Effort

---

### 9. WildDeepfake (original, non-HF)
- **GitHub:** https://github.com/xingjunm/wild-deepfake
- **Direct download:** Listed on GitHub project page (Google Drive links)
- **Note:** Same content as `xingjunm/WildDeepfake` on HF. Use the HF mirror (Tier 2, #6) for easier programmatic access.

### 10. DF40 (NeurIPS 2024)
- **Paper:** Yan et al., "DF40: Toward Next-Generation Deepfake Detection," NeurIPS 2024. arXiv:2406.13495.
- **GitHub:** https://github.com/YZY-stack/DF40
- **Licence:** CC-BY-NC 4.0 (non-commercial research only)
- **Size:** ~93 GB preprocessed (frame crops) + raw videos separately. Google Drive + Baidu Disk (password: `DF40`).
- **Scale:** 40 generation techniques — face-swapping, face-reenactment, full synthesis, face editing. Covers both GAN and diffusion generators.
- **Access:** No form required. Public Google Drive links in the GitHub repo README.
- **HF mirror:** `01Yassine/DF40` (160 GB JSONL+shards; gating status unclear — may require HF login for large files).
- **Relevance:** Extremely broad generator coverage for generalization testing. Best used for evaluation; 40-class generator label column enables per-generator AUC breakdown.
- **Integration note for Devin:** Download preprocessed crops from Google Drive. Build a `FolderDataset` pointing to the extracted directory. Do not use as primary training data (CC-BY-NC prohibits commercial use); use as a comprehensive cross-dataset eval.

### 11. Celeb-DF v2 (Kaggle mirrors)
- **Paper:** Li et al., "Celeb-DF: A Large-Scale Challenging Dataset for DeepFake Forensics," CVPR 2020.
- **Kaggle mirrors (no account required to browse):** Search "Celeb-DF v2" on Kaggle — community-uploaded mirrors exist.
- **Licence:** Academic research only (original TU authors' terms)
- **Access:** Official download requires form fill (GitHub); Kaggle mirrors are unofficial but functional.
- **Size:** ~2 GB video clips; 590 real + 5,639 deepfake videos, 300k+ frames total. Generator: custom high-quality face-swap pipeline.
- **Relevance:** Standard benchmark — models trained on other datasets are evaluated against Celeb-DF for cross-dataset AUC. Good eval-only set.
- **Integration note for Devin:** Use unofficial Kaggle mirror or use `Thien0103/DeepFake_Extracted_Face_Images` on HF (see below) which contains pre-extracted Celeb-DF v2 face crops.

### 12. `Thien0103/DeepFake_Extracted_Face_Images` ✅ VERIFIED PUBLIC
- **Link:** https://huggingface.co/datasets/Thien0103/DeepFake_Extracted_Face_Images
- **Licence:** Not specified
- **Size:** 30 MB (~2,798 face crops from Celeb-DF v2 video frames); Parquet format
- **Format:** `image` + `label` (`0=fake`, `1=real`)
- **Content:** Pre-extracted face crops from Celeb-DF v2. Also includes raw zip archives (`Celeb-DF-v2_splitted.zip`, `celeb_df_extracted.zip`) in the file listing.
- **Shortcut risk:** Small scale (2.8k samples) — evaluation only; too small for training.
- **Relevance:** Convenient quick Celeb-DF v2 eval without the full download. Good sanity check.

---

## Summary & Prioritised Recommendations for Devin

| Priority | Repo ID | Size | Use | LOGO? | Licence | Key Risk |
|---|---|---|---|---|---|---|
| **1** | `nebula/GenImage-arrow` | ~160 GB | Train + LOGO | ✅ (generator col) | CC-BY-NC-SA 4.0 | Size/aspect shortcut (non-face) |
| **2** | `bitmind/SyntheticFacesHQ` + `bitmind/bm-real` | ~5 GB + 6.7 GB | Train (face fakes + reals) | ❌ no gen col | MIT + "other" | Must pair manually; format align needed |
| **3** | `xingjunm/WildDeepfake` | ~73 GB | **OOD eval only** | ❌ no gen col | Apache-2.0 | WebDataset shards; streaming loader needed |
| **4** | `JamieWithofs/Deepfake-and-real-images` | 1.8 GB | Eval only | ❌ no gen col | Unknown | Unknown provenance |
| **5** | `yashduhan/DeepFakeDetection` | zip | Eval only | ❌ no gen col | Unknown | Undocumented generators |
| **6** | DF40 (Google Drive) | 93 GB | Eval only | ✅ (40-class) | CC-BY-NC 4.0 | Manual download; no HF streaming |
| **7** | `Thien0103/DeepFake_Extracted_Face_Images` | 30 MB | Eval only (Celeb-DF v2) | ❌ | Unknown | Too small for training |

### Recommended minimal addition for Devin to integrate next:

**Option A — Expand training coverage (diffusion faces):**  
Add `bitmind/SyntheticFacesHQ` (SFHQ T2I part — Flux.1 + SDXL fakes) paired with `bitmind/bm-real`.  
This adds modern Flux/SDXL generator coverage absent from GenImage and DeepFakeFace.

**Option B — Add a true in-the-wild eval split:**  
Download `xingjunm/WildDeepfake` Apache-2.0 shards and add a `"wild"` eval key in `eval_splits`.  
This gives the first OOD result on real-internet deepfakes (heterogeneous, unknown generators).

**Option C — Full-scale GenImage:**  
Swap `TheKernel01/Tiny-GenImage` for `nebula/GenImage-arrow` with `config=all-train`. Doubles the scale and brings generator labels needed for LOGO.

---

## Notes on Excluded Datasets

| Dataset | Reason Excluded |
|---|---|
| `nuriachandra/Deepfake-Eval-2024` | Gated (manual approval). Access pending. |
| `kd13/deepfakebench-unfiltered` | Does not exist on HF API (HTTP 404). |
| `dbristow/inpaint_bench` | Does not exist on HF API (HTTP 404). |
| `mueller91/In-The-Wild` | **Audio deepfake only** — not image/face. |
| `FaceForensics++` | Requires form fill (email registration). Use `bitmind/FaceForensicsC23` instead. |
| DFFD (MSU) | Academic-only request form, no public download. |
| `saakshigupta/deepfake-detection-dataset-v2/v3` | Only 1k–10k images; too small. |

---

## Format-Shortcut Risks Summary (for Devin)

All non-paired datasets carry the risk that real and fake images differ in resolution, JPEG quality, or aspect ratio.  
The `format_align` strategy implemented in `DeepFakeFaceDataset` (centre-crop → 384×384 bicubic → group-seeded JPEG re-encode) should be applied wherever real/fake format is not pre-matched by construction.

- `nebula/GenImage-arrow`: Fakes are square ImageNet-prompted outputs; reals are natural-aspect crops. Apply `format_align` or equivalent before mixing with face datasets.
- `bitmind/SyntheticFacesHQ` + `bitmind/bm-real`: SFHQ is 1024×1024; bm-real may vary. Downsample both to 384×384.
- `xingjunm/WildDeepfake`: In-the-wild face sequences, naturally varied — minimal shortcut risk.
- `JamieWithofs/`, `yashduhan/`: Unknown format — Devin should sample and measure size/quality distributions before training.
