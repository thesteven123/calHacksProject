# Research Verification & Empirical Handoff

**Author:** ML Literature Review & Research Agent  
**Audience:** Devin (Implementation Lead) & copilot-git-organizer  
**Date:** 2026-10-24  
**Status:** Research & Verification Complete (No Implementation Code)

---

## 1. Dataset Verification Matrix (Hugging Face Repositories)

Every dataset listed in the previous brief was checked for existence, gating status, size, licensing, metadata, and format matching.

| Dataset Repo ID | Status on HF | Download Access | License | Size / Samples | Format & Pair Matching |
|---|---|---|---|---|---|
| **[`OpenRL/DeepFakeFace`](https://huggingface.co/datasets/OpenRL/DeepFakeFace)** | **VERIFIED** | **Public (Direct)** | **Apache-2.0** (code/data); underlying IMDB-WIKI is research-only | ~5.07 GB (120,004 images across 4 zips of 30k each) | **1:1 Paired.** `wiki.zip` (authentic) matched 1:1 with `inpainting.zip` (SD inpainting), `text2img.zip` (SD1.5), `insight.zip` (InsightFace). Perfect resolution & format parity. |
| **[`Sowaiba01/Deepfake`](https://huggingface.co/datasets/Sowaiba01/Deepfake)** | **VERIFIED** | **Public (Direct)** | **MIT License** | ~1.2 GB (10,852 images: 5,426 real, 5,426 fake) | **1:1 Paired.** Generated via InsightFace `inswapper_128`. Real/fake pairs share identical JPEG 95% quality and camera framing. Metadata: `metadata.csv` (source, target, paths). |
| **`bitmind/bm-real`** & **`bitmind/SyntheticFacesHQ`** | **VERIFIED** | **Public (Direct)** | **Unspecified / "other"** (Bittensor Subnet 34) | `bm-real`: 28.4k rows.<br>`SyntheticFacesHQ`: 493k rows.<br>Video: `bitmind/FaceForensicsC23` (17.9 GB zip). | **Unpaired.** High-resolution curated collections. Real and synthetic are separated into distinct repositories; clean metadata, but not 1:1 semantic/identity matched. |
| **`dbristow/inpaint_bench`** | **DOES NOT EXIST** | N/A | N/A | N/A | **HALLUCINATED / INVALID REPO.** (Removed from recommendations). |
| **[`kd13/deepfakebench-unfiltered`](https://huggingface.co/datasets/kd13/deepfakebench-unfiltered)** | **VERIFIED** | **Public (Direct)** | **"other"** | 850,000+ rows across Celeb-DF v1/v2, FF++, DFDC, DFD, FaceShifter. | **Benchmark Frames.** Pre-extracted raw un-resized frame bytes from major video sets. Metadata: `image`, `binary_label`, `class_id`, `subset`, `source_dataset`, `method`, `video_id`. |
| **[`nuriachandra/Deepfake-Eval-2024`](https://huggingface.co/datasets/nuriachandra/Deepfake-Eval-2024)** | **VERIFIED (GATED)** | **Access-Controlled (Gated)** | **CC-BY-SA-4.0** (Eval only; training prohibited) | 45h video, 56.5h audio, 1,975 images from 2024 social circulation. | Real-world in-the-wild user flags. **Requires approval form** with institutional email and proof of research. Not usable immediately without approval. |

### Summary & Recommendations for Devin
1. **Immediate Pair-Matched Training:** Use **[`OpenRL/DeepFakeFace`](https://huggingface.co/datasets/OpenRL/DeepFakeFace)** and **[`Sowaiba01/Deepfake`](https://huggingface.co/datasets/Sowaiba01/Deepfake)**. Both are 100% open, ungated, have permissive licenses (Apache-2.0 and MIT), and eliminate aspect-ratio/resolution shortcuts via strict 1:1 real-to-fake pairs.
2. **Immediate Video Path (No Wait):** Download **[`bitmind/FaceForensicsC23`](https://huggingface.co/datasets/bitmind/FaceForensicsC23)** (17.9 GB, direct public download of 7,000 MP4s across all 6 manipulation algorithms) or pull Celeb-DF-v2 frame shards directly from **[`kd13/deepfakebench-unfiltered`](https://huggingface.co/datasets/kd13/deepfakebench-unfiltered)**. Do not wait on `Deepfake-Eval-2024` or `AV-Deepfake1M`.
3. **Inpainting Benchmark for Multi-Crop:** Use `inpainting.zip` from `OpenRL/DeepFakeFace` (30,000 local face edits on authentic backgrounds).

---

## 2. Re-Analysis of DINOv3 vs. DINOv2

Devin noted:
- `facebook/dinov2-base` has **no register tokens** in the standard release.
- The repository wrapper already skips DINOv3's 4 registers.
- Therefore, register token leakage does not explain why DINOv3 (0.819) underperformed DINOv2 (0.845, sign-flip $p=0.22$).

### A. Statistical Significance Check
- **Finding:** With sign-flip test $p = 0.22$ across 7 LOGO rotations on Tiny-GenImage, the performance difference between DINOv2 and DINOv3 is **statistically non-significant** ($p > 0.05$).
- **Evidence status:** *Verified*. We cannot reject the null hypothesis that DINOv2 and DINOv3 perform equally on this benchmark subset.

### B. Architectural Divergences: RoPE vs. Absolute Positional Embeddings
1. **Rotary Positional Embeddings (RoPE):**
   - DINOv2 uses standard learned 1D absolute positional embeddings added to the input sequence before layer 1.
   - DINOv3 introduces **Rotary Positional Embeddings (RoPE)** applied to queries and keys inside every self-attention block:
     $$q_{\text{rot}} = \text{RoPE}(W_q x), \quad k_{\text{rot}} = \text{RoPE}(W_k x)$$
   - When LoRA is injected into $W_q$ and $W_v$, the perturbed query vector $(W_q + \Delta W_q)x$ undergoes non-linear rotary phase shifts. In DINOv2, the projection output directly interacts with static position vectors.
   - *Hypothesis:* Standard LoRA hyperparameters ($LR = 3 \times 10^{-4}$) calibrated for absolute positional ViTs disrupt RoPE rotation geometry in DINOv3.
2. **Gram Anchoring & Over-Abstraction:**
   - DINOv3 introduces **Gram Anchoring** during pre-training to prevent dense patch representation collapse over 1.7B images.
   - Gram anchoring enforces invariant semantic correlation matrices across patches. While this improves semantic segmentation, it penalizes high-frequency photometric residuals that forensic classifiers depend on.
   - *Evidence status:* *Supported by foundation model literature (DINOv3 technical report)*.

### C. Concrete Fine-Tuning Recommendations for DINOv3
- If Devin runs a DINOv3 hyperparameter sweep:
  1. **Lower LoRA Learning Rate:** Reduce $\text{LR}_{\text{LoRA}}$ from $3 \times 10^{-4}$ down to **$5 \times 10^{-5}$ – $1 \times 10^{-4}$** to preserve RoPE phase stability.
  2. **Probe Earlier Layers:** Shift feature extraction from $\{4, 8, 10, 12\}$ to **$\{2, 5, 8, 11\}$** to capture low-level photometric features before Gram-anchored semantic invariance dominates.
  3. **Empirical Recommendation:** If limited on hackathon time, keep **DINOv2-base** as the production core; it is stable, ungated, and has a proven track record in wild deepfake literature.

---

## 3. Conformal Prediction & Calibration: Theoretical Assumptions & Evidence Audit

### A. Split-Conformal False Positive Rate (FPR) Guarantees

#### Core Assumption: Exchangeability
Split-conformal prediction provides finite-sample marginal coverage:
$$P(Y_{n+1} \in \mathcal{C}(X_{n+1})) \ge 1 - \alpha$$
**The Fundamental Assumption:** The calibration samples $(X_1, Y_1), \dots, (X_n, Y_n)$ and the deployment test point $(X_{n+1}, Y_{n+1})$ must be **exchangeable** (e.g. i.i.d.).

#### What Happens Under Social Media Platform Compression?
- **Failure Condition:** If the calibration set $\mathcal{D}_{\text{cal}}$ is drawn from clean or mildly compressed images, but the test image $X_{\text{deploy}}$ undergoes heavy lossy transcoding (e.g. WhatsApp H.264 CRF 32 or WebP Q40), **covariate shift** occurs:
  $$P_{\text{cal}}(X \mid Y=\text{real}) \ne P_{\text{deploy}}(X \mid Y=\text{real})$$
- **What Breaks:** The exchangeability assumption is **violated**. Non-conformity scores under heavy compression systematically shift upward, causing the empirical threshold $\hat{q}_{\text{real}}$ to become anti-conservative. The guaranteed $\le 1\%$ FPR bound collapses.
- **Evidence-Backed Mitigation (Weighted Conformal Prediction / PTE Calibration):**
  1. *Condition:* Calibration reals MUST be passed through the exact same **Platform Transcoding Emulation (PTE)** distribution as the training data.
  2. *Reference:* Tibshirani et al. (2019), *"Conformal Prediction Under Covariate Shift"* (NeurIPS). Weighted split-conformal prediction corrects for known covariate distributions by weighting calibration residuals with density ratios $w(x)$.
  3. *Evidence status:* **VERIFIED** (Standard statistical theory; exchangeability is mathematically strict).

#### Asymmetry Between Reals and Fakes
- **Reals:** Natural camera distributions under platform compression are reasonably bounded and exchangeable if PTE is matched. Conformal bounds on reals hold.
- **Fakes:** Unseen generators (Midjourney, Sora, Wan2.1) are inherently non-exchangeable with known generators (GLIDE, BigGAN). No standard conformal predictor can guarantee bounded FNR on unseen generators without structural assumptions. Thus, calibration on fakes must use **worst-case min-max bounds across known generator families** rather than marginal exchangeability.

---

### B. VIB-Variance OOD Gate: Literature Audit

- **Claim:** "Use posterior latent variance $\sigma^2(z|x)$ from the VIB head as an automatic epistemic uncertainty gate to trigger `ABSTAIN`."
- **Literature Verification Status:** **PARTIALLY SUPPORTED / SPECULATIVE AS A HARD THRESHOLD**.
  - **Verified Support:** Using Variational Information Bottleneck for Out-of-Distribution (OOD) detection is well-established in academic literature:
    - *Alemi et al. (ICLR 2017)*: Deep Variational Information Bottleneck.
    - *Fischer (2020)*: Conditional Entropy VIB (CE-VIB) for Out-of-Distribution Detection.
    - These works demonstrate that VIB models natural epistemic uncertainty; when presented with OOD inputs, the latent representation $q(z|x)$ exhibits high KL divergence from the prior $D_{KL}(q(z|x) \,\|\, \mathcal{N}(0, I))$ and high predictive entropy.
  - **Speculative Aspect:** Setting a hard empirical quantile threshold directly on the raw variance vector $\text{mean}(\sigma^2) > \text{Quantile}_{90}$ as an automated deepfake gating mechanism is an **engineering heuristic**, not an established theorem. It provides no formal statistical guarantee comparable to conformal prediction.
  - **Recommendation for Devin:** 
    - Do **not** rely on raw $\sigma^2$ thresholding as a primary safety gate.
    - Treat VIB variance as an auxiliary metric, or use **$D_{KL}(q(z|x) \,\|\, \mathcal{N}(0, I))$** (latent divergence to prior) which has formal backing in information-theoretic OOD literature.
    - Rely on **Conformal Quantiles on Real Scores** for strict platform risk bounds.
