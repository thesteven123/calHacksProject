# Adversarial ML & Systems Architecture Critique

**Date:** 2026-10-24  
**Role:** Adversarial ML Researcher & Social Media Systems Architect  
**Subagent Review ID:** `abca1e5a-21f8-4e9b-a23b-78839738cbe8`

---

## Executive Summary

The proposed architecture correctly pivots away from fragile academic forensic CNNs (which collapse under compression) toward foundation vision backbones (DINOv2/CLIP) and a tiered cost-aware funnel. However, four critical structural vulnerabilities were identified and require mitigations:

1. **Stage 1 Fast-Path Bypass & False-Negative Leakage:** Compressed-domain bitstream screeners fail completely on re-encoded photorealistic fakes because the bitstream syntax reflects the encoder (FFmpeg/libx264), not the generator.
2. **White-Box Foundation Transfer & Universal Adversarial Perturbations:** Standard frozen foundation backbones have publicly known weights, making gradient-based evasion trivially portable.
3. **The Beauty-Filter / Computational-Photography Collision:** Skin smoothing, bilateral filtering, and OEM camera AI share the exact latent characteristics targeted by naive forensic classifiers, causing severe False Positive Rates (FPR).
4. **Ingestion Bottlenecks:** Real-world inference cost is dominated by decode, tracking, and PCIe bandwidth—not ViT FLOPs.

---

## 1. Concrete Failure Modes & Vulnerabilities

### 1.1 White-Box Transferability & UAPs
- **The Frozen Backbone Liability:** DINOv2, DINOv3, and CLIP are open-weight models. An attacker does not need access to the proprietary classification head; they can craft Projected Gradient Descent (PGD) or universal adversarial perturbations (UAPs) against the public backbone to minimize distance between synthetic face tokens and authentic face cluster centroids.
- **Mitigation:** Introduce **Low-Rank Adaptation (LoRA)** on the projection layers ($W_q, W_v$) of the top 3 transformer blocks, co-trained with the Spatiotemporal Adapters. This perturbs the public foundation manifold, breaking off-the-shelf transferability of adversarial white-box attacks.

### 1.2 Variational Information Bottleneck (VIB) Pitfalls
- Deepfake forensic cues often reside in subtle, low-variance residuals (blending seams, micro-texture artifacts, spectral roll-off). High-level semantics (facial identity, expression, pose) dominate mutual information $I(Z; X)$. If $\beta$ is slightly too high, the bottleneck aggressively prunes subtle forensic traces under heavy social media compression (**semantic collapse**).
- **Mitigation:** Use schedule-annealed $\beta$ with a lower bound and add spatial boundary guidance to anchor the bottleneck to manipulation contours.

### 1.3 Spatiotemporal Adapters (StA) Temporal Evasion
- Attackers can confine synthetic manipulation or identity swapping to burst intervals or non-keyframe spans, or use neural frame interpolators (RIFE, FILM) to smooth motion vectors.
- **Mitigation:** Use top-$k$ frame pooling combined with dynamic keyframe selection rather than fixed stride.

### 1.4 Stage 1 Screener Leakage
- Bitstream heuristics (DCT coefficients / motion vectors) cannot distinguish a re-encoded fake from a re-encoded real video because the bitstream syntax is produced by the platform's video compressor.
- **Mitigation:** Replace bitstream parsing with a distilled ultra-lightweight vision student (e.g. **MobileNetV4-Conv-Small** or **FastViT-T8**, <1.5 ms on GPU) distilled directly from the Stage 2 DINOv2 teacher.

### 1.5 Beauty Filter False Positives
- TikTok/Instagram beauty filters perform bilateral skin smoothing, pore removal, and teeth whitening that closely resemble GAN/diffusion skin statistics.
- **Mitigation:** Multi-task disentanglement: train on triplets $\{ \text{Real Native}, \text{Real Filtered}, \text{Deepfake} \}$ with an auxiliary beauty-filter prediction head.

---

## 2. Ingestion & Scale Engineering

- **Fleet Sizing:** At 100M uploads/day (~3,000 uploads/sec peak), keyframe decoding (NVDEC) and face detection/tracking are the primary bottlenecks, not ViT inference.
- **Hardware Decode & Tracking:** Pipeline keyframe decoding directly via hardware NVDEC with hardware-accelerated bounding box tracking across frames.
- **Virality-Gated Stage 3 Escalation:** Scale Stage 3 compute by gating escalation with platform velocity telemetry:
  $$\text{Escalation Priority} = P_{\text{uncertainty}}(\text{Stage 2}) \times \log_{10}(\text{Velocity}_{\text{impressions/sec}} + 1)$$
