# Deepfake Detection Model — Plan (no implementation yet)

Date: 2026-10-23. Based on findings in `research/papers.md`.
Target: image + video deepfake detection for a social-media platform —
must survive recompression/re-encoding, generalize to unseen generators,
and run near-real-time at scale.

## 1. Design principles (what the research says)

1. **Evaluate in-the-wild, not on FF++.** Deepfake-Eval-2024 and TrueFake show
   ~45–50% AUC collapse from lab to wild. FF++ c23/c40 and DFDC are
   regression-test material, not the target.
2. **Frozen foundation backbone + small head beats custom forensic CNNs.**
   DINOv2/v3 (and CLIP per UnivFD) feature spaces carry forensic signal and
   generalize across generators.
3. **Train through the distribution channel.** Platform compression emulation
   + aggressive codec/resize augmentation is non-negotiable; also prevents the
   model from exploiting compression "block effects" as a shortcut (PLADA).
4. **Kill dataset bias.** Semantically aligned real/fake pairs (B-Free) and
   format/dual alignment (DDA); VIB objective to suppress non-forensic features.
5. **Tiered inference.** One heavy model per upload is economically infeasible;
   funnel: provenance → light screener → core detector → offline escalation.
6. **Plan for continual updates.** Generators evolve; use the AI-GenBench-style
   incremental protocol (Pellegrini et al.) rather than full retrains.

## 2. System architecture (tiered funnel)

```
Upload
  │
  ▼
Stage 0 — Provenance & heuristics (CPU only)
  • C2PA / content-credentials verification when present
  • Perceptual hash (pHash/PDQ) dedup + known-fake/known-real lookup
  • Face detection (MediaPipe / RetinaFace); crop with ~1.3× box margin
    to capture blending boundaries; record face-track for video
  │
  ▼
Stage 1 — Front-door screener (target <5 ms/item GPU)
  • Small CNN head on compressed-domain features to fast-path
    obvious-authentic content; flag borderline/suspicious for Stage 2
  │
  ▼
Stage 2 — Core detector (~30–50 ms/item)
  • Image: frozen DINOv2/v3-base backbone + MLP head w/ VIB objective
  • Video: 3–5 keyframes/s → per-frame face-crop scores → top-k logit
    pooling (fakes often glitch for only a few frames)
  • Output: calibrated score + abstain band → human review
  │
  ▼
Stage 3 — Escalation (async, high-risk content only)
  • Heterogeneous ensemble à la HEDGE (DINOv3 + MetaCLIP2 routes,
    multi-resolution) — optionally distilled (FeatDistill)
  • Audio-visual sync check (AVFF/CAD-style) when speech audio present
  • Temporal localization (next-frame feature prediction) to find
    spliced segments (FakeParts-style partial fakes)
```

Open question to validate in prototyping: whether Stage 1 earns its keep vs.
running Stage 2 on everything — decide via throughput measurements.

## 3. Model design details

### Backbone & head (Stage 2)
- Frozen **DINOv2-base** initially (DINOv3 if license/weights permit); face crops
  as primary input for face content, full frame for non-face content.
- Light head: 2-layer MLP or linear probe. Compare vs. face-domain embeddings
  (face foundation model + metric learning, Mylonas & Papadopoulos 2025).
- Add **VIB-style information bottleneck** on the head to suppress
  authentication-irrelevant features (VIB-Net reported +23.6% acc on unseen
  generator families).

### Training data strategy
- **Sources (real):** modern camera pipelines incl. AI-enhanced phone images;
  ensure diversity of devices, demographics (age bias is documented), languages.
- **Sources (fake):** GAN family + diffusion (SDXL, FLUX-class, Midjourney/DALL·E
  outputs where obtainable) + video generators (AnimateDiff-class); face-swap /
  lip-sync / reenactment tools; partial manipulations (FakeParts-style).
- **Bias controls:**
  - Generate fakes *from* real samples (SD img2img regeneration, B-Free style)
    to keep real/fake semantically aligned.
  - Match file format/size distributions between real and fake (DDA).
- **Augmentation (the critical piece):**
  - Platform compression emulation — infer H.264/HEVC/resize params per target
    platform with <50 probe uploads (Montibeller et al. method), apply locally.
  - Random JPEG/WebP quality, downscale→upscale chains, double/triple
    recompression (cascades are the top real-world failure mode).
  - Beautification/smoothing filters as augmentation on *real* data to cut
    the filter false-positive mode.
  - Video: VFR simulation, frame drops, motion blur — prevents temporal
    detectors from flagging mobile artifacts.

### Video temporal handling
- Per-frame scores → top-k mean / learned temporal pooling.
- Optional temporal-consistency head (landmark/optical-flow displacement)
  only in Stage 3 — mobile artifacts make it noisy for front-door use.

### Continual learning
- Follow AI-GenBench incremental protocol: temporally order generators, evaluate
  update methods (replay buffers vs. distillation) so new-generator coverage
  doesn't regress old-generator accuracy.

## 4. Evaluation plan

| Level | Benchmark | Purpose |
|---|---|---|
| Lab sanity | FF++ (c23/c40), DFDC sample, Celeb-DF | Regression / comparison to literature |
| Cross-generator | AI-GenBench (temporal split), GenImage | Unseen-generator generalization |
| Compression | TrueFake (shared via FB/X/Telegram), ODDN/PLADA protocol, our emulated-compression test set | Social-media robustness |
| In-the-wild | Deepfake-Eval-2024 (image+video+audio), PDID | Deployment-realistic accuracy |
| Partial fakes | AV-Deepfake1M, FakeParts | Temporal/spatial localization |
| Bias audit | Age-Diverse Deepfake Dataset, per-demographic slices | Fairness false-positive check |

**Metrics:** AUC + TPR@fixed-FPR (moderation operates at fixed FPR budgets —
report TPR@1%FPR and TPR@0.1%FPR, not just accuracy). Calibration (ECE) for the
abstain band. Track per-source and per-compression-level breakdowns.

## 5. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Cascaded/multi-codec recompression destroys features | Multi-step augmentation, codec-diverse emulation, evaluate on doubly-compressed sets |
| Beauty filters → false positives | Filter augmentation on reals; fairness audit; abstain band |
| Temporal cues fire on VFR/dropped frames | Top-k pooling (don't require persistent signal); VFR augmentation |
| New generator released post-deployment | Continual-update pipeline + frozen-backbone head retrains are cheap |
| Adversarial evasion (resize-as-attack documented) | Include scaling in augmentation; monitor score drift |
| Ensemble cost at scale | Keep heavy ensemble in async Stage 3; distill into Stage 2 if it proves better |
| Watermark/provenance absent on most uploads | Treat C2PA as a bonus signal only — passive detection is the primary path |

## 6. Prototype milestones

1. **M1 — Data pipeline:** dataset assembly, platform-compression emulator,
   augmentation stack, aligned real/fake generation (B-Free-style).
2. **M2 — MVP image detector:** frozen DINOv2 + VIB-MLP head; evaluate on
   GenImage → Deepfake-Eval-2024 → emulated-compression split.
3. **M3 — Video extension:** face-track keyframe extraction + top-k pooling;
   evaluate on Deepfake-Eval-2024 video + AV-Deepfake1M.
4. **M4 — Escalation tier:** HEDGE-style 2-route ensemble + AV sync check;
   measure uplift vs. cost.
5. **M5 — Calibration & ops:** abstain band, per-slice audits, drift monitoring,
   continual-update protocol.
