# Implementation Architecture — Deepfake Detector ("DFD")

**Owner:** Devin (architecture + implementation)
**Status:** v0.2. S1, S2 and S3 (local subset) are done. The S4 video path and S6 funnel have scaffolding
(inference only, no video training yet). S5 Tinker route code is done and verified offline; it needs
`TINKER_API_KEY` to train. See §11 for the build log.
**Inputs:** `docs/architecture-spec.md`, `docs/model-plan.md`,
`research/papers.md`, `research/critiques/adversarial_review.md`

This document turns the research-level spec into an engineering design that can
be built incrementally. Where it deviates from `architecture-spec.md`, the
reason is stated (usually an adversarial-review mitigation or a tooling
constraint).

---

## 0. Decisions at a glance

| # | Decision | Why | Source |
|---|---|---|---|
| D1 | Core backbone = **frozen `facebook/dinov2-base`** (HF, ungated); DINOv3 is a config swap once gated access is granted | Best generalization-per-FLOP; linear probes on DINO beat forensic CNNs in the wild | papers.md §Foundation |
| D2 | **LoRA (r=8) on the Q/V projections of the last 3 ViT blocks** (`q_proj`/`v_proj` in transformers ≥5, `query`/`value` in older versions; both are matched) | Breaks off-the-shelf white-box transfer from the public backbone; cheap | critique §1.1 |
| D3 | **Multi-layer token features** (CLS + mean-patch from 4 intermediate layers) instead of last-layer CLS only | Low-level forensic traces live in earlier layers; linear probes on the final CLS discard them | Gemini critique, papers.md |
| D4 | **VIB head with annealed β and a floor/cap** | Suppresses dataset shortcuts without semantic collapse | VIB-Net; critique §1.2 |
| D5 | **Auxiliary beauty-filter head** (multi-task, triplets real / real-filtered / fake) | Cuts beauty-filter false positives | critique §1.5 |
| D6 | Video = per-frame features → **temporal adapter (depthwise 1-D conv, k=3,5)** → **top-k logit pooling** with **change-based keyframe selection** | Captures feature drift; robust to burst-only fakes and VFR | spec §2.2; critique §1.3 |
| D7 | Stage 1 = **distilled MobileNetV4-Conv-Small student** (timm), *not* a bitstream/DCT screener | Bitstream reflects the encoder, not the generator | critique §1.4 |
| D8 | Stage 3 VLM route trained on **Tinker** (Qwen3.5/3.6-VL LoRA SFT, answer ∈ {real, fake}, score = token log-prob) | Tinker trains VLMs only; adds backbone diversity (HEDGE-style) | Tinker docs |
| D9 | **Platform Transcoding Emulation (PTE)** applied to *both* classes with identical distributions | Prevents format/compression shortcuts (DDA, PLADA) | spec §3 |
| D10 | Metrics = **AUC, TPR@1%FPR, TPR@0.1%FPR, ECE**, per-slice (generator, codec level, demographic) | Moderation runs at fixed FPR budgets | model-plan §4 |
| D11 | Output = **calibrated probability + 3-way verdict** (real / abstain / fake) via temperature scaling + two thresholds | Abstain band feeds human review | model-plan §2 |

---

## 1. Repository layout

```
src/dfd/
  config.py              # dataclass configs (model, data, train), YAML load
  models/
    backbone.py          # DINOv2 wrapper: multi-layer features + LoRA injection
    lora.py              # minimal LoRA for nn.Linear (no peft dependency)
    heads.py             # VIBHead, aux beauty head
    temporal.py          # TemporalAdapter, top-k pooling, keyframe selection
    detector.py          # DeepfakeDetector (image + video forward)
    student.py           # Stage 1 MobileNetV4 student + distillation loss
  data/
    pte.py               # Platform Transcoding Emulation (JPEG/WebP chains, resize buckets, chroma, blur)
    beauty.py            # beautification filter simulator (bilateral / smoothing / sharpen)
    video_codec.py       # ffmpeg-based H.264/HEVC/AV1 re-encode chains (video)
    datasets.py          # HF dataset adapters (Tiny-GenImage, DeepFakeFace, ...), folder dataset, synthetic
    transforms.py        # train/eval transform pipelines
    faces.py             # face detection + 1.35x padded crop (optional dep)
  training/
    losses.py            # VIB loss, β schedule, aux loss, distillation loss
    trainer.py           # training loop (AMP, MPS/CUDA/CPU), checkpointing
  eval/
    metrics.py           # AUC, TPR@FPR, ECE, per-slice report
    calibration.py       # temperature scaling + abstain thresholds
  inference/
    pipeline.py          # tiered funnel: Stage0 hash → Stage1 student → Stage2 core → Stage3 escalate
    video.py             # frame sampling / decoding
  tinker_route/
    build_sft_data.py    # export PTE-augmented real/fake images as Tinker SFT chat examples
    train_vlm.py         # Tinker cookbook launcher (custom ClassifierDataset)
    score_vlm.py         # log-prob scoring for the ensemble
scripts/
  train.py  evaluate.py  distill.py  export_onnx.py
configs/
  mvp_image.yaml  smoke.yaml
tests/
```

---

## 2. Stage 2 core model (the MVP)

```
image (224×224, PTE-augmented)
   │
   ▼
DINOv2-base (frozen) ── LoRA(q,v) on blocks 9,10,11
   │  hidden_states from layers {4, 8, 10, 12}
   ▼
per layer: [CLS ; mean(patch tokens)]  → LayerNorm → concat   (4 × 2 × 768 = 6144-d)
   │
   ▼
VIB encoder: Linear(6144→512) → GELU → Linear(512→2·d_z)  → μ, logσ²   (d_z = 128)
   │  z = μ + σ·ε   (train)  |  z = μ   (eval)
   ├──► classifier  Linear(d_z→1)            → fake logit
   └──► beauty head Linear(d_z→1) (aux)      → "is beauty-filtered" logit
```

**Loss**

$$\mathcal{L} = \mathcal{L}_{BCE}(y,\hat y) + \beta_t\, D_{KL}(q(z|x)\,\|\,\mathcal N(0,I)) + \lambda_{b}\,\mathcal{L}_{BCE}(b,\hat b)$$

- β schedule: linear warm-up from 0 → `beta_max` (default 1e-3) over the first 30%
  of steps, clamped to `[beta_min, beta_max]` afterwards. KL is summed over latent
  dims and averaged over the batch. If validation TPR@1%FPR drops while KL keeps falling, β is
  too high (semantic collapse) → the trainer logs both for monitoring.
- λ_b = 0.2. The beauty label is 1 if the beauty-filter augmentation fired on the
  sample (applied to real *and* fake so it can't become a label proxy).

**Trainable parameters:** LoRA (~0.15M) + VIB head (~3.3M). The backbone stays
frozen, so head retrains for new generators are cheap (continual-update path).

**Video extension (M3)**

```
frames (sampled 3–5 fps) → keyframe selector (top-N by feature delta + uniform anchors)
  → per-frame multi-layer features [T, 6144]
  → TemporalAdapter: Linear→768, two depthwise Conv1d branches (k=3, k=5), residual
  → per-frame VIB → per-frame logits [T]
  → top-k mean (k = max(1, ⌈0.25·T⌉))  → video logit
```

This is a *post-backbone* temporal adapter. The spec's in-block StA (adapters
between ViT blocks) is deferred to after M3 benchmarks show the post-backbone
version leaves accuracy on the table, since in-block adapters break feature
caching and add latency.

---

## 3. Data pipeline

### 3.1 Sources (verified on HF Hub, 2026-10-24)

| Dataset | HF id | Access | Use |
|---|---|---|---|
| Tiny-GenImage | `TheKernel01/Tiny-GenImage` | open, 35k imgs, 8 generators + `generator` column | MVP train/val; **leave-one-generator-out** (LOGO) split |
| GenImage (full) | `nebula/GenImage-arrow` | open | scale-up |
| DeepFakeFace | `OpenRL/DeepFakeFace` | open (zips) | face-domain diffusion fakes |
| FF++ c23 | `bitmind/FaceForensicsC23` | open | lab regression only |
| NTIRE 2026 robust train | `deepfakesMSU/NTIRE-RobustAIGenDetection-train` | open (zips) | in-the-wild distortions |
| AV-Deepfake1M | `ControlNet/AV-Deepfake1M` | gated | video / temporal localization (M3) |
| Deepfake-Eval-2024, TrueFake, SIDA, Omni-Fake | external | request access | held-out evaluation only, never training |

### 3.2 Platform Transcoding Emulation (PTE)

The same stochastic chain is applied to both classes, with 1–3 hops sampled per image:

1. **Resize bucket:** short side ∈ {360, 480, 720, 1080} with random interpolation
   (bilinear/bicubic/lanczos/area), and optional down→up rescale (the resize-as-attack defence).
2. **Codec:** JPEG q∈[30,95] (4:2:0 or 4:4:4) | WebP q∈[30,95] | none.
3. **Mild degradations:** gaussian blur σ∈[0,1.2], gaussian noise, gamma/contrast jitter.
4. **Beauty filter (p=0.25):** bilateral smoothing + optional unsharp mask, which sets the aux label.
5. Final: random-resized-crop 224 (scale 0.6–1.0) for train, center-crop for eval.
   Crop rather than resize preserves pixel-level traces (Pellegrini et al.).

Video (M3): `ffmpeg` chains of libx264 crf 23–32 → optional libx265/libaom-av1,
fps resampling (VFR simulation), frame drops. This requires `ffmpeg` on PATH.

### 3.3 Bias controls
- **Format alignment (DDA):** all training images are decoded to RGB and pushed
  through PTE, so the original file format/quality cannot leak the label.
- **Aligned pairs (B-Free):** M1.5 adds an SD img2img regeneration script that
  creates fakes *from* reals (needs a GPU; uses HF `diffusers`).
- **Size jitter** prevents image dimensions from being used as a shortcut.

---

## 4. Stage 1 student (M4)

- `timm/mobilenetv4_conv_small.e2400_r224_in1k`, 1-logit head.
- Distillation: `L = BCE(y, s) + α · KL(σ(t/T) || σ(s/T))·T²` with the Stage 2
  teacher's calibrated logits, on PTE-augmented data.
- Operating point: only pass media as "confident real" when student
  p(fake) < τ₁, where τ₁ is chosen so that **student FNR vs. teacher ≤ 0.5%**
  on the validation set. Everything else goes to Stage 2.
- The kill-switch criterion is in model-plan §2: if Stage 1 cannot pass ≥60% of
  traffic at that FNR, drop it and run Stage 2 on everything.

---

## 5. Stage 3 escalation (M4)

| Route | Model | Where trained |
|---|---|---|
| A | Stage 2 core, but at 448px input (high-res, fine-grained cues) | local / GPU |
| B | **Qwen3.5/3.6-VL + LoRA**, SFT on prompt "Is this image real or AI-generated/manipulated?" → `real`/`fake` | **Tinker** (`tinker_cookbook` vlm_classifier recipe with a custom `ClassifierDataset`) |
| C (opt.) | DINOv3-L / MetaCLIP2 linear probe | local / GPU |

Score for B = `log p("fake") − log p("real")` from the sampler at temperature 0.
Fusion: logit-space weighted average, with weights fit on the validation set
(logistic regression on route logits). A dual-gate drops a route whose logit
deviates by more than 3σ from the others (HEDGE-style).

Video escalation adds NSG / AV-sync once M3 lands.

---

## 6. Calibration & verdicts

1. Fit temperature T on the validation set (LBFGS on NLL).
2. Choose `τ_fake` at the target FPR (default 1%) and `τ_real` at the target
   FNR (default 2%). Scores in `[τ_real, τ_fake)` → **abstain** → human review
   queue, with escalation priority `p_uncertain · log10(velocity+1)` (critique §2).
3. Persist `{T, τ_real, τ_fake}` alongside the checkpoint (`calibration.json`).

Per the Gemini critique, attention/Grad-CAM maps are **not** shipped to
moderators as evidence.

---

## 7. Evaluation protocol

`scripts/evaluate.py` produces one JSON + markdown report per run with:

- Overall: AUC, AP, TPR@1%FPR, TPR@0.1%FPR, ECE, accuracy@0.5.
- Slices: per generator, per PTE severity (clean / 1-hop / 3-hop / q<50),
  beauty-filtered reals FPR, and (when labels exist) demographic slices.
- **LOGO generalization:** train on 7 Tiny-GenImage generators, test on the
  held-out one, and rotate. This is the main generalization number until
  in-the-wild sets are obtained.

---

## 8. Compute plan

| Phase | Where | Notes |
|---|---|---|
| Dev / smoke tests | local Mac (MPS) | `configs/smoke.yaml`, synthetic data, dinov2-small |
| MVP training (Tiny-GenImage) | local MPS feasible (frozen backbone, cached features) or 1× cloud GPU | head-only training on cached features (`--cache-features`) is ~minutes |
| LoRA + full PTE training | 1× A10/L4/A100 (cloud) | on-the-fly augmentation means features can't be cached |
| VLM route | Tinker | requires `TINKER_API_KEY` |
| Latency benchmarking | NVIDIA T4/L4 + TensorRT | export via `scripts/export_onnx.py` |

---

## 9. Milestones (implementation order)

| Step | Deliverable | Exit criterion |
|---|---|---|
| **S1** | Package scaffold, configs, PTE + beauty augs, synthetic dataset | unit tests pass |
| **S2** | DINOv2+LoRA+VIB detector, trainer, metrics, calibration | smoke run trains end-to-end on synthetic data |
| **S3** | HF Tiny-GenImage adapter + LOGO split; first real training run | report with AUC/TPR@FPR per generator |
| **S4** | Video path (frame sampling, temporal adapter, top-k) + ffmpeg codec chains | video smoke test |
| **S5** | Tinker VLM route (data export + training launcher + scorer) | Tinker run on a Tiny-GenImage subset |
| **S6** | Stage 1 distillation + tiered inference pipeline + ONNX export | funnel runs end-to-end on a folder of media |
| **S7** | Ensemble fusion, fairness slices, continual-update protocol | report |

---

## 10. Known risks specific to this implementation

- **Tiny-GenImage is GenImage-derived** and carries GenImage's known JPEG/size
  bias (real = ImageNet JPEG, fake = PNG). PTE + format alignment mitigates
  this, but LOGO numbers will still overstate in-the-wild performance. It must
  be validated on Deepfake-Eval-2024 / TrueFake before any claims are made.
- **Face content:** GenImage is mostly non-face. Face swaps / reenactment
  need DeepFakeFace + FF++ + face cropping (`data/faces.py`) in S3/S4.
- **LoRA vs. generalization:** LoRA can re-introduce overfitting. Ablate
  frozen-only against LoRA on the LOGO split.
- **MPS numerics:** some ops fall back to CPU. Final numbers should come from CUDA runs.

---

## 11. Build log & results

### 2026-10-24: S1–S3 and S5 (offline)

**Environment:** `.venv` (core; pinned via `requirements.lock`, built with `--exclude-newer` 7-day cutoff),
`.venv-tinker` (separate env, because `tinker-cookbook` pins transformers 5.5 / huggingface-hub 1.x).

**Smoke (synthetic, dinov2-small):** pipeline trains end-to-end on MPS. The severity sweep already shows the
expected collapse under heavy recompression (clean AUC 1.00 → heavy 0.73) on a toy "upsampling grid" trace.

**S3: Tiny-GenImage subset, LOGO with Midjourney held out** (`configs/tiny_genimage_local.yaml`;
3,714 train / 1,750 val; DINOv2-base + LoRA r=8 ×3 blocks + VIB; 2 epochs, ~4 min on M-series MPS).
All validation images pass through PTE.

> ⚠️ These numbers are **contaminated**: the held-out generator's fakes were included in the validation
> set used for best-checkpoint selection and calibration (found in the 2026-10-24 code review). Treat
> Midjourney results as an upper bound. A clean rerun after the fix reports the honest numbers below.

| Eval condition | AUC | TPR@1%FPR | Midjourney (unseen) AUC | Midjourney TPR@1%FPR |
|---|---|---|---|---|
| Training-PTE mix (model selection) | 0.925 | 0.472 | 0.820 | 0.136 |
| clean | 0.964 | 0.635 | 0.891 | 0.248 |
| light (1× JPEG/WebP q75–95) | 0.959 | 0.573 | 0.891 | 0.208 |
| heavy (3 hops, q30–60, rescale) | 0.897 | 0.368 | 0.786 | 0.112 |
| beauty-filtered | 0.934 | 0.518 | 0.827 | 0.232 |

Calibrated verdicts (held-out half of val): FPR 0.2%, FNR 2.5%, **abstain 55%**. The abstain rate is far too
high for production, which shows the model needs more data and epochs; it is not yet a threshold problem.

**Read-outs:**
1. The gap to the unseen generator is real: Midjourney TPR@1%FPR is roughly ⅓ of in-distribution generators.
   This is the number to move.
2. Heavy cascaded recompression costs about 7 AUC points overall and 10 on the unseen generator, which matches the literature.
3. These numbers come from 13% of Tiny-GenImage and say nothing about in-the-wild performance (§10).

**S3 clean rerun (after the review fix; `runs/tiny_genimage_logo_mj_v2`).** Validation is now split into
`val_select` (1,187 images: in-distribution fakes + half the reals), which drives checkpoint choice and
calibration, and `val_test` (563 images: all 125 Midjourney fakes + the other half of the reals), which
nothing touches during training.

| Split | AUC | TPR@1%FPR | Notes |
|---|---|---|---|
| val_select (in-distribution generators) | 0.957 | 0.539 | used for selection and calibration |
| **val_test, Midjourney unseen (honest LOGO)** | **0.833** | **0.080** | contaminated run reported 0.820 / 0.136 |
| val_test at epoch 0 (not selected) | 0.842 | 0.184 | in-distribution selection picked the worse epoch for the unseen generator |

On unseen Midjourney fakes and held-out reals, calibrated verdicts give FPR 2.1% (target 1%) and
**FNR 12% (target 2%)**, with 43% abstaining. Thresholds tuned on known generators do not carry over
to a new one.

**Read-outs from the clean run:**
1. Unseen-generator detection at a moderation operating point is weak (8% TPR@1%FPR). This is the
   central problem, as the literature predicted.
2. Choosing the checkpoint on in-distribution data actively hurts unseen-generator performance
   (epoch 0 beat epoch 1). We need a proxy for generalization during selection, e.g. a nested
   leave-one-generator-out validation generator.
3. Calibration does not transfer across generators, so the abstain band and escalation tier carry more
   weight than the core score suggests.

**Next experiments (priority order):**
- Nested LOGO selection: hold out a second generator (not the test one) and select on it.
- Full Tiny-GenImage (8 GB) on a CUDA GPU, 5 epochs; then rotate LOGO across all 7 generators.
- Ablations: LoRA on vs. off; β ∈ {0, 1e-4, 1e-3, 1e-2}; feature layers {12} vs. {4,8,10,12}.
- Tinker VLM route on the same split, then logit fusion with Stage 2 (S7).
- Add face-domain data (DeepFakeFace, FF++ c23) and face cropping (S4).

### 2026-10-24 (later): nested holdouts, multi-crop scoring, credentials, DINOv3

**Credentials (verified with read-only calls, `scripts/check_credentials.py`):**
- `HF_TOKEN` is valid. Gated **DINOv3 (`facebook/dinov3-vitb16-pretrain-lvd1689m`) is granted**; the config
  downloaded and a full detector forward pass ran. **AV-Deepfake1M is still pending** (`GatedRepoError`).
- `TINKER_API_KEY` is valid: `get_server_capabilities` lists 35 models, and the route default
  `Qwen/Qwen3.6-35B-A3B` is among them. No billable Tinker calls were made.
- Found and fixed: the shell exported an **empty** `TINKER_API_KEY`, which hid the `.env` value.
  `dfd.env.load_dotenv` now treats empty variables as unset. All scripts load `.env` automatically.
- Found and fixed: DINOv3 was **not** a config swap, because its blocks live at `model.layer`.
  `_transformer_blocks` now handles that; LoRA attaches to 6 projections and the 4 register tokens are skipped.

**Nested LOGO protocol** (`configs/tiny_genimage_nested.yaml`, `split_nested_holdouts`). Training excludes
GLIDE and Midjourney. Validation is split into three disjoint parts:
- `fit`: 5 known generators + ⅓ of reals, used for calibration;
- `select`: GLIDE fakes + ⅓ of reals, used to pick the checkpoint;
- `test`: Midjourney fakes + ⅓ of reals, scored **once**, after selection.

| Split | n (fakes) | AUC | TPR@1%FPR |
|---|---|---|---|
| select (unseen GLIDE), chosen epoch | 417 (125) | 0.873 | 0.400 |
| **test (unseen Midjourney, untouched)** | 417 (125) | **0.830** | **0.168** |

Verdicts on test: FPR 0.0%, FNR 10.4%, 61% abstain. **Caveats:** the test set has only 292 reals, so
1% FPR is about 3 images and TPR@0.1%FPR cannot be resolved. Re-scoring the same checkpoint with a
different augmentation seed gives TPR@1%FPR 0.096 vs. 0.168, so differences under ~±0.08 are noise.
This is not directly comparable to the earlier 0.080, because the training set (GLIDE removed) and
the real split differ. Treat the gain as unproven until LOGO is rotated across generators with confidence intervals.

**Multi-crop scoring** (`dfd.inference.crops`): up to a 3×3 native-resolution grid + centre, top-25% mean.
Paired bootstrap on the Midjourney test set (2,000 resamples):

| Condition | Centre AUC / TPR@1% | Multi AUC / TPR@1% | ΔAUC (95% CI) | ΔTPR@1% (95% CI) |
|---|---|---|---|---|
| clean | 0.899 / 0.360 | 0.950 / 0.352 | **+0.051 [+0.025, +0.081]** | −0.008 [−0.072, +0.144] |
| training PTE mix | 0.846 / 0.096 | 0.832 / 0.088 | −0.014 [−0.033, +0.006] | −0.008 [−0.144, +0.048] |

- On clean media, multi-crop raises AUC significantly, and the gain survives a crop-count-matched control
  (+0.044 on images with ≥9 crops). This matters because Midjourney fakes are all 1024 px while reals have
  a median short side of 375 px, so the crop count would otherwise confound the result.
- Under realistic platform compression it gives **no gain**. Pooled real logits rise more than fake logits
  (+0.71 vs. +0.17), because more tiles give more chances for a compressed real tile to score high.
- **Decision:** the default stays `crop_mode="center"`. Multi-crop is opt-in (`--crop-mode multi`) and
  uses its own `calibration_multicrop.json`, fit on `fit` only. If it falls back to centre-crop
  thresholds, the pipeline marks `calibration_mismatch` and never auto-flags `fake`.
- **Limit:** Tiny-GenImage fakes are fully synthetic, not localized edits, so this benchmark cannot show
  the off-centre benefit multi-crop is meant for. The next test is localized manipulation data, such as
  DeepFakeFace inpainting or SIDA.

### 2026-10-24 (later): first Tinker runs — shortcut found and fixed

**Setup:** `Qwen/Qwen3.6-35B-A3B` with LoRA r=32, **20 steps × 32 examples** (640 images) on the nested
training split, about 9 s/step, no mid-run validation pass. Measured spend:

| Run | Train tokens | Est. cost |
|---|---|---|
| native-aspect run | 114,787 | $0.135 |
| square-crop run | 154,240 | $0.182 |
| scoring (~870 images × 2 queries) | ~0.4M prefill | ~$0.18 |
| **Total** | | **≈ $0.50** of $150 |

The billing CLI lags by a few hours; confirm with `tinker billing usage <start> <end>`.

**Shortcut found.** The first (native-aspect) checkpoint scored AUC **0.983** on unseen GLIDE. But on that
split, **image size alone gives AUC 1.000**, and "is the image square?" gives 0.992: every GenImage fake
is square, while 98% of reals are not. A VLM sees the whole frame, so size and aspect ratio leak through
its image-token grid. Centre-cropping the test images to squares dropped it to AUC 0.912 / TPR@1%FPR
0.112 (from 0.600).
**Fix:** `DEFAULT_SQUARE_SIZE = 448`. Both classes get the same square crop (random position in training,
centre at scoring), so every example is now exactly 241 tokens. This is on by default in training and
the scorer. The core detector is mostly immune because it scores fixed 224² native crops, but small
images (BigGAN 128 px) are still upsampled. This bias must be removed in the data (size- and
aspect-matched reals) before any in-the-wild claim.

**Corrected comparison, on the same 250 unseen-GLIDE images** (125 fakes, 125 reals, identical PTE
realizations):

| Route | AUC | TPR@1%FPR |
|---|---|---|
| core DINOv2 + LoRA + VIB (nested best.pt) | 0.870 | 0.424 |
| VLM route, square crops, 20 steps | 0.871 | 0.240 |
| equal-weight z-score fusion | **0.901** | **0.504** |

Fusion minus core: ΔAUC **+0.031, 95% CI [+0.008, +0.055]**; score correlation r = 0.75.

**Caveats:**
- The core checkpoint was *selected* on this split, so the core route is slightly favoured.
- Fusion z-scores use this split's statistics.
- With 125 reals, TPR@1%FPR resolves only about 1 image.
- The VLM saw only 640 examples.

This is evidence that the routes are **complementary**, not a final number. Next: train the VLM on the
full nested split, fit fusion weights on `fit` only, then score Midjourney `test` once.

**Tinker checkpoints:** `5e073c74-…` (square crop, valid) is kept; its 2 checkpoints total 8.3 GB.
`b440c6ad-…` (native aspect, shortcut-contaminated) was **deleted 2026-10-24**, with user approval.

### 2026-10-24 (later): full VLM route + fusion, nested LOGO (Midjourney scored once)

**Training:** `Qwen/Qwen3.6-35B-A3B` with LoRA r=32 and square-crop input, on the full nested training
split (3,428 images × 2 epochs = 216 steps, ~6.4 s/step), 1.65M train tokens ≈ **$1.94**. Checkpoints
were saved every 54 steps and **chosen on unseen GLIDE** with the same rule as the core detector
(AUC + TPR@1%FPR):

| ckpt | 000054 | 000108 | **000162** | final |
|---|---|---|---|---|
| GLIDE AUC / TPR@1% | 0.857 / 0.200 | 0.879 / 0.240 | **0.903 / 0.304** | 0.875 / 0.312 |

**Fusion** (`scripts/fuse_routes.py`, `dfd.eval.fusion`): standardized logistic regression on
[core, VLM] logits, **fit on the `fit` split only** (5 known generators, 916 images). Weights: core 1.60,
VLM 1.86. Calibration for every route is also fit on `fit`. All comparisons are paired on identical
degraded images (`dfd.eval.paired`). The core logits on `select` reproduce training-time validation
exactly (0.873 / 0.400), which checks the pairing.

| Split | Route | AUC | TPR@1%FPR | Verdicts FPR / FNR / abstain |
|---|---|---|---|---|
| select (GLIDE; used for selection, biased) | core | 0.873 | 0.400 | 0.3% / 8.0% / 48% |
| | VLM | 0.903 | 0.304 | 0.0% / 4.0% / 50% |
| | fused | 0.907 | 0.440 | 0.0% / 11.2% / 32% |
| **test (Midjourney, untouched)** | core | 0.844 | 0.096 | 0.3% / 6.4% / 59% |
| | VLM | 0.826 | 0.256 | 0.3% / 11.2% / 51% |
| | fused | **0.854** | 0.216 | 0.7% / **16.8%** / 38% |

Paired bootstrap on test (2,000 resamples):
- fused − core: ΔAUC **+0.009 [−0.012, +0.030]**, ΔTPR@1% +0.120 [−0.048, +0.304]. **Not significant.**
- VLM − core: ΔAUC −0.019 [−0.053, +0.016], ΔTPR@1% **+0.160 [0.000, +0.368]**. Borderline.
- Score correlation r = 0.78.

**Read-outs:**
1. The fusion gain on GLIDE (+0.034 AUC, significant) **did not replicate** on the untouched Midjourney
   test. GLIDE was used to select both checkpoints, so its numbers are optimistic. This is exactly what
   the nested protocol is meant to catch.
2. The VLM route has a different error profile: lower AUC but a higher low-FPR detection rate (25.6% vs
   9.6% at 1% FPR, CI lower bound 0). It is a plausible *escalation* signal, not a replacement.
3. Calibration fit on known generators does not transfer. Fusion cut abstentions from 59% to 38%, but
   missed fakes rose from 6.4% to 16.8% against a 2% target. **Do not deploy fused thresholds** without
   generator-diverse calibration data.
4. Single holdout, 292 test reals: every conclusion here needs the rotated-LOGO repeat before it informs
   a design decision.

**Spend this step:** ≈ $1.94 training + ≤ $0.66 scoring (3,001 images). **Cumulative Tinker spend ≈ $3.2 of $150.**
**Checkpoints (run `ba5f8840-…`):** `000054`, `000108` and `000162` expire 7 days after creation;
`final` never expires. The selected checkpoint is `000162`.

### 2026-10-24 (later): Modal launcher + 7-rotation nested LOGO, DINOv2 vs DINOv3

**Infrastructure:** `scripts/modal_logo.py` runs one Modal job per rotation (selection generator, test
generator). Each job uses an L4 + 8 cores + 32 GiB, a 2 h timeout and at most 7 concurrent jobs. One job
warms the cache first; the rest run in parallel. Only `HF_TOKEN` is sent, as an ephemeral secret. Volumes:
`dfd-hf-cache` and `dfd-runs`. Results are copied to `runs/modal/<tag>/` and summarized with
`scripts/summarize_logo.py`. Each job took **1.1–1.8 min wall-clock (~$0.04 list)**, versus ~8 min on
the Mac. **14 jobs ≈ $0.53** from the $30 included credit (estimate from wall time; confirm in the Modal
dashboard). Rotation rule: each of the 7 generators is the test generator once, and the selection
generator is the next one in the cycle. Same data subset and nested protocol as before
(`configs/tiny_genimage_nested.yaml`).

| Test generator | DINOv2 AUC / TPR@1% | DINOv3 AUC / TPR@1% |
|---|---|---|
| ADM | 0.783 / 0.104 | 0.729 / 0.120 |
| BigGAN | 0.855 / 0.160 | 0.743 / 0.048 |
| GLIDE | 0.861 / 0.288 | 0.868 / 0.304 |
| Midjourney | 0.780 / 0.088 | 0.818 / 0.088 |
| SD15 | 0.793 / 0.200 | 0.781 / 0.232 |
| VQDM | 0.884 / 0.232 | 0.823 / 0.056 |
| Wukong | 0.959 / 0.608 | 0.970 / 0.624 |
| **Mean (95% CI over rotations)** | **0.845 [0.805, 0.894] / 0.240 [0.142, 0.379]** | 0.819 [0.768, 0.880] / 0.210 [0.093, 0.370] |

**Paired DINOv3 − DINOv2** (same rotation, same images and degradations; exact sign-flip test):
ΔAUC −0.026 (3/7 wins, p = 0.22), ΔTPR@1% −0.030 (4/7 wins, p = 0.50). **No evidence that DINOv3 helps**
in this setup, with the same LoRA/VIB recipe and hyperparameters tuned for DINOv2. The core stays on
**DINOv2-base**; DINOv3 stays a config option.

**Read-outs:**
1. This is the first multi-generator generalization estimate. On an unseen generator the core averages
   **AUC 0.845** and catches **24% of fakes at 1% FPR**, ranging from 9% (Midjourney) to 61% (Wukong). The
   earlier single-holdout Midjourney numbers sit near the bottom of that range.
2. The result depends on **which generator is held out for checkpoint selection**. Midjourney as the test
   generator scored AUC 0.780 with SD15 as selector vs. 0.827–0.844 with GLIDE as selector. Single-rotation
   comparisons are not decision-grade; use this 7-rotation summary as the yardstick for every future change.
3. The rotations reuse the same 292 test reals, so they are not fully independent. The CIs over rotations
   capture generator-to-generator variation, not real-image sampling noise.
4. Verdict-level missed-fake rates on unseen generators reach 16–18% (ADM, Midjourney) against a 2% target,
   confirming again that calibration does not transfer across generators.

**Next on Modal (cheap now that a run costs ~$0.04):** full Tiny-GenImage (14 train / 4 val shards) to cut
test noise; ablations on the 7-rotation yardstick (LoRA off, β ∈ {0, 1e-4, 1e-2}, last-layer-only features);
a DINOv3 learning-rate sweep before rejecting it outright.
