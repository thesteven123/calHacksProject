# Research Proposal: Innovative Architectural Enhancements for Deepfake Detection (DFD)

**Authors:** agy-research & codex-web-search  
**Date:** 2026-10-24  
**Target:** Devin (Architecture Design & Implementation)  
**Status:** Research Proposal & Evaluation Blueprint (No code modified)  
**Sources:** CVPR 2024/2025, ECCV 2024, NeurIPS 2024, IJCAI 2025  

---

## 1. Executive Summary & Diagnostic of Current Baseline

The current DFD core architecture consists of:
- **Backbone:** Frozen `facebook/dinov2-base` + LoRA ($r=8$) on $q,v$ projections of the last 3 ViT blocks (layers 9, 10, 11).
- **Features:** Multi-layer concatenated representations: $[\text{CLS}_l \,;\, \text{mean-patch}_l]$ across layers $l \in \{4, 8, 10, 12\}$ (dim $6144$).
- **Head:** Variational Information Bottleneck (VIB) head ($6144 \to 512 \to 2 \cdot 128 \to \mu, \log\sigma^2$) + linear classifier + auxiliary beauty-filter head.
- **Data & Augmentation:** Platform Transcoding Emulation (PTE) applied identically to real and fake classes; split-conformal calibration for 3-way verdicts (real / abstain / fake).

### Baseline Performance (7-Rotation Nested LOGO on Full Tiny-GenImage)
- **Mean unseen-generator AUC:** **0.915** [0.847, 0.965]
- **Mean unseen-generator TPR@1%FPR:** **0.420** [0.259, 0.578]
- **Conformal FPR on Reals:** 1.2% (close to 1.0% target)

### Critical Bottlenecks & Failure Modes
1. **Midjourney Outlier Failure:** While BigGAN (0.978), VQDM (0.980), and SD1.5 (0.958) generalize well, **Midjourney drops to AUC 0.741, TPR@1%FPR 0.088, and missed-fake rate (FNR) 32.6%**. Midjourney's high visual photorealism and proprietary latent upscaling minimize obvious spatial artifacts, evading semantic DINOv2 features.
2. **Inpainting / Local Edit Dilution:** Mean-pooling all $14 \times 14 = 196$ patch tokens in layers $\{4, 8, 10, 12\}$ dilutes localized manipulations (e.g. DeepFakeFace inpainting or facial component swaps) with 80–90% authentic background tokens. Multi-crop scoring provided +0.05 AUC clean, but degraded under compression.
3. **Semantic vs. High-Frequency Dilemma:** DINOv2 self-supervision naturally abstracts away high-frequency pixel residuals to build semantic invariance. Conversely, naive frequency methods (SRM filters, raw DCT) collapse under cascaded JPEG/WebP transcoding (PTE).
4. **Iterative Diffusion Inversion Latency:** State-of-the-art diffusion forensics (DIRE, Fake-Inversion, FIRE) require 20–50 reverse diffusion steps per image (500ms–2s), violating the high-throughput moderation SLA ($<50$ms per image).

---

## 2. Four Innovative Architectural Directions

Below are four concrete, low-latency innovations sourced from 2024–2025 literature, evaluated for compatibility with our frozen DINOv2 + LoRA setup.

```
                    ┌────────────────────────────────────────────────────────┐
                    │                      Input Image                       │
                    └───────────┬────────────────────────────────┬───────────┘
                                │ (RGB)                          │ (Fast VAE / SRM)
                                ▼                                ▼
              ┌───────────────────────────────────┐    ┌───────────────────┐
              │  Frozen DINOv2 + LoRA(q,v last 3) │    │  1-Step VAE / SRM │
              │   Extract Patch Tokens (14×14)    │    │ Residual (AeroB)  │
              └───────────────┬───────────────────┘    └─────────┬─────────┘
                              │                                  │
                              ▼                                  ▼
              ┌───────────────────────────────────┐    ┌───────────────────┐
              │  Innovation 1: Attention-MIL /    │    │  Innovation 2:    │
              │  Differentiable Top-k Pooling     │    │  Residual Feature │
              └───────────────┬───────────────────┘    └─────────┬─────────┘
                              │                                  │
                              └───────────────┬──────────────────┘
                                              ▼
                               ┌─────────────────────────────┐
                               │  VIB Head (d=128 Latent)    │
                               │  + Innovation 4: GMM-OOD    │
                               └──────────────┬──────────────┘
                                              ▼
                               ┌─────────────────────────────┐
                               │ Conformal 3-Way Decision:   │
                               │  Real / Abstain / Fake      │
                               └─────────────────────────────┘
```

---

### Innovation 1: Patch-Level Attention-MIL & Top-k Pooling (Zero FLOP Overhead)

* **Problem Solved:** Overcomes dilution of localized inpainting and face-swaps caused by global `mean(patch_tokens)`.
* **Literature Sources:**
  - *Contrastive Learning for DeepFake Classification and Localization via Multi-Label Ranking* (CVPR 2024). [CVPR Paper](https://openaccess.thecvf.com/content/CVPR2024/html/Hong_Contrastive_Learning_for_DeepFake_Classification_and_Localization_via_Multi-Label_Ranking_CVPR_2024_paper.html).
  - *Detecting Localized Deepfake Manipulations Using Action Unit-Guided Representations* (CVPR Workshops 2025). [Paper](https://openaccess.thecvf.com/content/CVPR2025W/SAIAD/html/Anand_Detecting_Localized_Deepfake_Manipulations_Using_Action_Unit-Guided_Video_Representations_CVPRW_2025_paper.html).

#### Architectural Mechanism
Instead of reducing layer-12 patch tokens $P \in \mathbb{R}^{196 \times 768}$ to $\bar{P} \in \mathbb{R}^{768}$ via mean-pooling, formulate the patch tokens as a bag of instances in Multiple Instance Learning (MIL):
1. **Patch Classifier:** Project each patch token to an instance logit: $s_i = W_{\text{patch}} P_i + b_{\text{patch}} \in \mathbb{R}$.
2. **Attention-Weighted Pooling (Ilse et al. formulation):**
   $$a_i = \frac{\exp\left(w^\top \tanh(V P_i^\top) \odot \text{sigm}(U P_i^\top)\right)}{\sum_{j=1}^{196} \exp\left(w^\top \tanh(V P_j^\top) \odot \text{sigm}(U P_j^\top)\right)}$$
   $$f_{\text{MIL}} = \sum_{i=1}^{196} a_i P_i$$
3. **Auxiliary Top-$k$ Loss:** Pool the top-$k$ logits ($k \in \{2, 4, 8\}$):
   $$s_{\text{top-}k} = \frac{1}{k} \sum_{i \in \text{Top-}k(s)} s_i$$
   For an inpainting fake, if even 4 patches carry manipulation boundaries, $s_{\text{top-}k}$ triggers without needing the entire face to be fake.

#### Complexity & Trade-offs
- **FLOPs / Latency:** $< 0.1\%$ increase. Reuses already-computed DINOv2 layer-12 patch tokens.
- **Inference Time:** Identical to baseline.
- **Calibration Caveat:** Top-$k$ pooling can slightly elevate false alarms on natural high-contrast regions (e.g. jewelry, eyeglasses). Must be gated by the CLS token and calibrated on `fit` reals.

---

### Innovation 2: Fast 1-Step Latent Autoencoder Residuals (AeroBlade-Style)

* **Problem Solved:** Targets the Midjourney / Latent Diffusion failure mode without incurring the prohibitive 50-step inversion latency of DIRE / Fake-Inversion / FIRE.
* **Literature Sources:**
  - *AEROBLADE: Training-Free Detection of Latent Diffusion Images Using Autoencoder Reconstruction Error* (CVPR 2024). [arXiv:2401.17879](https://arxiv.org/abs/2401.17879), [Code: jonasricker/aeroblade](https://github.com/jonasricker/aeroblade).
  - *D3: Scaling Up Deepfake Detection by Learning from Discrepancy* (CVPR 2025). [Paper Record](https://mlanthology.org/cvpr/2025/yang2025cvpr-scaling/).

#### Core Forensic Insight
Latent Diffusion Models (SD1.4, SD1.5, SDXL, and Midjourney) synthesize images directly within the latent space of a pre-trained autoencoder (specifically, the standard SD VAE / VQ-VAE). 
- **The Asymmetry:** Synthetic images reside on the reconstruction manifold of the VAE decoder. When passed through a frozen VAE:
  $$\hat{x} = \mathcal{D}(\mathcal{E}(x))$$
  **AI-generated images are reconstructed with significantly lower $L_1$ and perceptual error than authentic camera photographs.** Real camera photos contain sensor shot noise, Bayer demosaicing artifacts, and non-latent textures that the VAE struggles to reconstruct.
- **Speed:** Requires only **one single feedforward pass** through a lightweight VAE ($\sim 10$–$15$ ms on L4 GPU), compared to 20–50 iterative denoising steps in DIRE ($>500$ ms).

#### Proposed Integration into DFD Core
1. Run a frozen 1-step VAE pass: $\Delta x = |x - \mathcal{D}(\mathcal{E}(x))|$.
2. Downsample $\Delta x$ via a 3-layer stride-2 convolutional adapter to produce residual token embeddings $R \in \mathbb{R}^{196 \times 128}$.
3. Concatenate $R$ with DINOv2 patch tokens before the VIB head.
4. For images where DINOv2 is uncertain (e.g. Midjourney), $\Delta x$ provides the orthogonal latent-prior footprint.

---

### Innovation 3: Consistency-Driven Frequency Debiasing (FreqDebias)

* **Problem Solved:** Prevents forensic frequency adapters from collapsing under platform compression (JPEG / WebP / PTE).
* **Literature Sources:**
  - *FreqDebias: Towards Generalizable Deepfake Detection via Consistency-Driven Frequency Debiasing* (CVPR 2025). [arXiv:2509.22412](https://arxiv.org/abs/2509.22412), [CVPR 2025 Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Kashiani_FreqDebias_Towards_Generalizable_Deepfake_Detection_via_Consistency-Driven_Frequency_Debiasing_CVPR_2025_paper.pdf).
  - *Rethinking the Up-Sampling Operations in CNN-based Generative Network for Generalizable Deepfake Detection (NPR)* (CVPR 2024). [CVPR Paper](https://openaccess.thecvf.com/content/CVPR2024/html/Tan_Rethinking_the_Up-Sampling_Operations_in_CNN-based_Generative_Network_for_Generalizable_Deepfake_CVPR_2024_paper.html).
  - *WaveDIF: Wavelet Sub-band Based Deepfake Identification in Frequency Domain* (CVPR Workshops 2025). [CVPRW Paper](https://openaccess.thecvf.com/content/CVPR2025W/CVEU/html/Dutta_WaveDIF_Wavelet_sub-band_based_Deepfake_Identification_in_Frequency_Domain_CVPRW_2025_paper.html).

#### Architectural Mechanism
1. **Fixed Up-sampling / Residual Extractor:** Apply Neighboring Pixel Relationship (NPR) or multi-scale Haar Discrete Wavelet Transform (DWT) decomposition ($LL, LH, HL, HH$) on the input.
2. **Spectral Consistency Loss Under PTE:**
   Conventional frequency branches fail because they memorize a specific JPEG Q-table. FreqDebias applies Forgery Mixup and minimizes the divergence between the features of clean image $x$ and platform-transcoded image $\text{PTE}(x)$:
   $$\mathcal{L}_{\text{consistency}} = \mathcal{D}_{\text{cos}}(f_{\text{freq}}(x), f_{\text{freq}}(\text{PTE}(x)))$$
3. **Late Gated Fusion:** A lightweight learnable scalar gate $\alpha \in [0, 1]$ modulates the frequency branch contribution:
   $$z_{\text{fused}} = z_{\text{semantic}} + \sigma(\alpha) \cdot z_{\text{freq}}$$
   If PTE compression is severe, $\alpha$ automatically attenuates the frequency contribution, falling back gracefully to the semantic DINOv2 branch.

---

### Innovation 4: Latent Real-Manifold OOD Gating for Conformal Abstention

* **Problem Solved:** Halts the 32.6% missed-fake rate on unseen generators by formalizing OOD abstention.
* **Literature Sources:**
  - *Beyond Generation: A Diffusion-based Low-level Feature Extractor for Detecting AI-generated Images* (CVPR 2025). [CVPR Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhong_Beyond_Generation_A_Diffusion-based_Low-level_Feature_Extractor_for_Detecting_AI-generated_CVPR_2025_paper.pdf).
  - *Think Twice before Adaptation: Improving Adaptability of DeepFake Detection via Online Test-Time Adaptation (T2A)* (IJCAI 2025). [arXiv:2505.18787](https://arxiv.org/abs/2505.18787).

#### Technical Formulation
1. **Fit One-Class Manifold on Calibration Reals:**
   In the 128-dimensional VIB latent space $z$, authentic camera images form a compact, smooth cluster. Fit a 6-component Gaussian Mixture Model (GMM) or Mahalanobis distance metric on the $\mu(z)$ vectors of the calibration reals:
   $$D_{\text{real}}(x) = \min_k \left( (\mu(x) - \mu_k)^\top \Sigma_k^{-1} (\mu(x) - \mu_k) \right)$$
2. **Dual-Condition Conformal Rule:**
   Instead of checking only the classification probability $p(y=1|x)$, check both probability and manifold distance:
   - **Real Verdict:** $p < \tau_{\text{real}}$ **AND** $D_{\text{real}}(x) < \gamma_{\text{real}}$ (it looks real AND lies on the real data manifold).
   - **Fake Verdict:** $p > \tau_{\text{fake}}$.
   - **Abstain (Human Review):** If $p < \tau_{\text{fake}}$ but $D_{\text{real}}(x) > \gamma_{\text{real}}$ (the image looks like a real photo to DINOv2, but its latent representation lies far off the authentic camera manifold — e.g. Midjourney).
3. **Impact:** Unseen generator fakes that trick the linear boundary are caught by the OOD manifold check and diverted to the Abstain queue rather than released as false negatives.

---

## 3. Prioritized Implementation Roadmap for Devin

To maintain rapid iteration and adhere to the project's verification standards (`.venv/bin/pytest -q` in 5s, smoke test in 1 min), the innovations are prioritized by complexity and risk:

| Phase | Innovation | Complexity | Added Latency | Expected Impact | Recommended Action |
|---|---|---|---|---|---|
| **Phase 1** | **Patch Attention-MIL & Top-$k$ Pooling** | Minimal ($\sim 40$ lines in `heads.py`) | $0.0$ ms | Eliminates inpainting dilution; boosts localized fake detection | **Immediate Implementation** |
| **Phase 2** | **VIB Real-Manifold OOD Abstention** | Low ($\sim 50$ lines in `calibration.py`) | $< 0.5$ ms | Slashes Midjourney FNR from 32.6% by routing to human review | **Immediate Implementation** |
| **Phase 3** | **1-Step VAE Latent Residual (AeroBlade)** | Medium (add frozen VAE forward in `backbone.py`) | $\approx 12$ ms (L4 GPU) | Directly attacks Midjourney / SDXL latent prior shortcut | **Implement after Phase 1/2** |
| **Phase 4** | **FreqDebias / Wavelet Consistency Adapter** | High (new dual-branch training & loss) | $\approx 5$ ms | Adds cross-generator frequency robustness under compression | **Ablation candidate** |

---

## 4. Evaluation Protocol & Verification Gates

Per the project protocol in `AGENTS.md` and `docs/implementation-architecture.md`:
1. **Nested LOGO Integrity:**
   - Candidate hyper-parameters (top-$k$ value, VAE residual weight, OOD threshold $\gamma$) must be tuned **only** on `data.selection_generators` (e.g. GLIDE or SD1.5).
   - Midjourney and held-out generators are evaluated strictly once after checkpoint selection.
2. **Metric Yardstick:**
   - Report paired results on the 7-rotation LOGO suite: **Mean AUC**, **TPR@1%FPR**, **TPR@0.1%FPR**, and **Conformal FNR**.
   - With $N \approx 1,167$ test reals, report bootstrap 95% confidence intervals.
3. **Platform Degradation Sweep:**
   - Every candidate change must pass the PTE degradation sweep (`scripts/evaluate.py --severities all`) to verify that gains on clean images do not evaporate under compression.
