# Social Media Deepfake Detection Architecture Specification

**Status:** Proposed Architecture  
**Target Ingestion:** Live Social Media Ingestion (Images + Short/Long-form Video)  
**Primary Constraints:**
1. High throughput (millions of uploads/day)
2. Low latency (Stage 1 screener < 5ms; Stage 2 core < 35ms; Stage 3 async escalation < 500ms)
3. Robustness against aggressive lossy transcoding (H.264/HEVC/AV1, downscaling, multi-pass recompression)
4. Generalization across unseen generators (GANs, SDXL, Flux, Midjourney, Sora, Kling, Wan2.1)

---

## 1. System Pipeline: Tiered Ingestion Funnel

```
                          ┌───────────────────────────┐
                          │ Incoming Upload (Img/Vid) │
                          └─────────────┬─────────────┘
                                        │
                                        ▼
          ┌───────────────────────────────────────────────────────────┐
          │ STAGE 0: Fast Metadata, Provenance & Deduplication (CPU)   │
          │ • C2PA / Content Credentials cryptographic verification   │
          │ • Perceptual Hash (PDQ / pHash) vs. Known-Fake / Real DB │
          │ • Face Detection (MediaPipe/RetinaFace) + Bounding BBox   │
          │   with 1.35x padding (capturing blend boundaries)         │
          └─────────────────────────────┬─────────────────────────────┘
                                        │
               ┌────────────────────────┴────────────────────────┐
        (Known Match)                                      (New Media)
               ▼                                                 │
      [Instant Verdict]                                          ▼
                                      ┌───────────────────────────────────────────────────┐
                                      │ STAGE 1: Front-Door Screener (GPU, < 5ms)         │
                                      │ • MobileNetV4-Small / EfficientNet-B0 on          │
                                      │   compressed-domain DCT / residual representations│
                                      │ • High-Confidence Threshold: Passes obvious       │
                                      │   authentic media immediately (80-85% of traffic) │
                                      └─────────────────────────┬─────────────────────────┘
                                                                │
                                            ┌───────────────────┴───────────────────┐
                                      (Confident Real)                        (Uncertain / Flagged)
                                            ▼                                       │
                                     [Approve Upload]                               ▼
                                      ┌───────────────────────────────────────────────────┐
                                      │ STAGE 2: Core Forensic Detector (~25–35ms)        │
                                      │ • Backbone: Frozen DINOv2-Base / DINOv3 (ViT-B/14)│
                                      │ • Video Temporal Adapter: Spatiotemporal Adapter  │
                                      │   (StA) 3D-conv modules capturing feature drift   │
                                      │ • Classifier Head: Variational Information        │
                                      │   Bottleneck (VIB) MLP (suppresses platform noise)│
                                      │ • Top-K Logit Pooling across 3-5 fps keyframes    │
                                      └─────────────────────────┬─────────────────────────┘
                                                                │
                                     ┌──────────────────────────┴──────────────────────────┐
                             (Confident Real)          (Borderline / Abstain)        (Confident Fake)
                                     ▼                          │                           ▼
                              [Approve Upload]                  ▼                     [Tag / Quarantine]
                                      ┌───────────────────────────────────────────────────┐
                                      │ STAGE 3: Escalation Tier (Async Queue, < 500ms)   │
                                      │ • Heterogeneous Ensemble (HEDGE-style: DINOv3 +   │
                                      │   MetaCLIP2-Giant + high-res path)               │
                                      │ • Physics Prior: Normalized Spatiotemporal        │
                                      │   Gradient (NSG) checking diffusion continuity    │
                                      │ • Audio-Visual Sync (AVFF/CAD) if speech present  │
                                      │ • Human Moderation Routing with Heatmap Overlay   │
                                      └───────────────────────────────────────────────────┘
```

---

## 2. Core Model Architecture Details (Stage 2)

### 2.1 Vision Foundation Backbone
- **Base Encoder:** Frozen `DINOv2-Base` (86M params, patch size 14x14) or `DINOv3-Base`.
- **Rational:** DINO features possess self-supervised semantic-forensic duality. They capture subtle photometric and boundary anomalies without fine-tuning, dramatically reducing overfitting to specific generator fingerprints.
- **Precision:** FP8 / INT8 post-training quantization via TensorRT for ultra-low VRAM footprint and high batch throughput.

### 2.2 Spatiotemporal Adapter (StA) for Video
- Rather than processing 3D volumes through heavy 3D-ViTs (e.g. VideoMAE), lightweight **Spatiotemporal Adapters** are injected between intermediate frozen transformer blocks.
- **Mechanism:** Dual-branch 1D temporal + 2D spatial depthwise separable convolutions with varying kernel sizes (k=3, 5).
- **Target:** Captures *Facial Feature Drift* (FFD) and inter-frame identity jitter caused by frame-by-frame generative models, while maintaining low latency.

### 2.3 Variational Information Bottleneck (VIB) Head
- Standard linear probes exploit dataset artifacts (e.g. camera sensor noise, color saturation, compression differences between datasets).
- **VIB Formulation:**
  $$\mathcal{L}_{VIB} = \mathcal{L}_{CE}(y, \hat{y}) + \beta \cdot D_{KL}(q(z|x) \,\|\, p(z))$$
- Constrains the latent feature space $z$ to discard authentication-irrelevant details (such as platform compression blockiness, illumination variations) while retaining predictive manipulation markers.

---

## 3. Social Media Compression Resilience Pipeline

### 3.1 Platform Transcoding Emulation (PTE)
Social networks (TikTok, Instagram, X, Facebook) apply aggressive multi-stage transcoding pipelines:
1. **Dynamic Resolution Resizing:** Rescaling to standard display buckets (e.g., 720p, 1080p, square 1:1, vertical 9:16).
2. **Cascaded Lossy Codecs:** Transcoding via H.264 (libx264, crf 23-32), HEVC (H.265), VP9, and AV1.
3. **Chroma Subsampling & Quantization:** 4:2:0 YUV downsampling and dynamic DCT coefficient quantization.
4. **Platform-Realistic Data Augmentation:** Every training sample undergoes stochastic simulation of multi-hop compression chains (e.g. WhatsApp compression -> CapCut edit re-encode -> Platform upload).

### 3.2 Anti-Shortcut Strategy: Symmetrized Blending & Format Alignment
- **Aligned Pairs (B-Free / SBI):** Generate fakes directly from authentic social media frames using inpainting, diffusion-regeneration, and synthetic blending (SBI) to balance image semantics.
- **Format Normalization (DDA):** Synchronize the distribution of JPEG quality factors, WebP compression, and color profiles between real and fake training buckets.
- **Beautification Jitter:** Inject synthetic skin smoothing, bilateral filtering, and low-light sharpening into real training samples to prevent false-positive alarms on mobile beauty filters.

---

## 4. Benchmark & Validation Protocol

Evaluation avoids saturated lab datasets (e.g., raw FF++ c0/c23) and evaluates strictly against:
1. **Omni-Fake (CVPR 2026):** Multi-modal OOD evaluation on 1M+ samples across images, video, and audio.
2. **SIDA (CVPR 2025):** Social media distribution and localization benchmark.
3. **Deepfake-Eval-2024:** Real user-flagged in-the-wild media.
4. **TrueFake:** Multi-platform (Facebook, X, Telegram) verified pipeline uploads.
5. **DAID Demographic Fairness Audit:** Evaluating equalized TPR@0.1%FPR across age, ethnicity, and gender subgroups.
