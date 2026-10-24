# Deepfake Detection — Paper Findings

Compiled 2026-10-23. Sources: arXiv (searched via arxiv MCP), CVPR/NeurIPS open
access, and web. Focus: image + video deepfake detection that survives social-media
distribution (recompression, re-encoding, unseen generators, scale/latency).

## TL;DR

- Academic benchmarks (FF++, DFDC) are saturated and **not predictive** of
  real-world performance. In-the-wild AUC drops ~45–50% across modalities
  (Deepfake-Eval-2024, PDID, TrueFake).
- The strongest recent direction is **frozen vision foundation model backbones**
  (DINOv2/v3, CLIP) with light heads — they beat purpose-built forensic CNNs
  in-the-wild while being cheap to adapt.
- **Heterogeneous ensembles** (multiple backbones/resolutions, logit fusion) are
  the current in-the-wild SOTA (NTIRE 2026 challenge) but are heavy for
  production — better as an offline/escalation tier.
- For social media specifically: compression robustness must be trained in via
  augmentation/emulation (ODDN, PLADA, compression-emulation fine-tuning).
  Frequency-domain artifact detectors (NPR, DIRE, LGrad) largely **collapse**
  under platform re-encoding — do not rely on them alone.
- Video: sparse frame scoring + temporal aggregation works; audio-visual fusion
  (AVFF, CAD) helps only when speech audio is present.

---

## In-the-wild benchmarks & reality checks

### Deepfake-Eval-2024: A Multi-Modal In-the-Wild Benchmark
- **Authors:** N. A. Chandra, H. Lee, ..., A. Kim, O. Etzioni (TrueMedia / AI2)
- **Venue/date:** arXiv 2503.02857, v5 2025
- **Link:** https://arxiv.org/abs/2503.02857
- **Key idea:** Benchmark built from media actually circulated in 2024 —
  user-flagged content from TrueMedia.org and social platforms: 45 h video,
  56.5 h audio, 1,975 images, 88 source websites, 52 languages.
- **Metrics:** Open-source SOTA AUC drops **-50% video, -48% audio, -45% image**
  vs academic benchmarks. Commercial detectors better but still below human
  forensic analysts.
- **Relevance:** The most realistic evaluation target for this project. If we
  evaluate anywhere, it should be here + TrueFake-style shared-media splits.

### Fit for Purpose? Deepfake Detection in the Real World (PDID)
- **Venue/date:** arXiv 2510.16556, 2025
- **Link:** https://arxiv.org/abs/2510.16556
- **Key idea:** Systematic eval of academic/government/industry detectors on the
  Political Deepfakes Incident Database — real political deepfakes shared on
  social media since 2018.
- **Metrics:** Max AUC only **74.78% image / 73.67% video** across all detectors;
  paid tools beat free ones; all vulnerable to simple manipulations, esp. video.
- **Relevance:** Second confirmation that lab numbers don't transfer. Spectral
  characteristics of wild fakes differ from lab-generated fakes.

### TrueFake: 600k images shared via real social networks
- **Venue/date:** arXiv 2504.20658, 2025
- **Link:** https://arxiv.org/abs/2504.20658
- **Key idea:** 600k images from modern GAN/diffusion generators actually uploaded
  to Facebook, X, Telegram; measures detector TPR/TNR after platform processing.
- **Metrics:** NPR (low-level upsampling traces) severely degraded by platform
  pipelines; CLIP-based R50-ND degrades less. X leaves some SD images
  unmodified — platform-specific quirks matter.
- **Relevance:** Directly models our deployment condition. Any model must be
  trained/evaluated through platform-style transcoding, not on clean data.

### NTIRE 2026 Challenge: Robust AI-Generated Image Detection in the Wild
- **Authors:** A. Gushchin, R. Timofte, D. Vatolin, et al. (large team)
- **Venue/date:** arXiv 2604.11487, CVPR 2026 NTIRE workshop
- **Link:** https://arxiv.org/abs/2604.11487
- **Key idea:** Challenge specifically targeting detection robustness under
  real-world distortions; source of current SOTA recipes (see HEDGE, FeatDistill).
- **Relevance:** Winning solutions are ensemble-heavy — good reference for our
  "escalation tier" but not the front-door model.

---

## Foundation-model-based detectors (most promising family)

### UnivFD: Towards Universal Fake Image Detectors
- **Authors:** U. Ojha, Y. Li, Y. J. Lee (UW–Madison)
- **Venue/date:** CVPR 2023
- **Link:** https://openaccess.thecvf.com/content/CVPR2023/html/Ojha_Towards_Universal_Fake_Image_Detectors_That_Generalize_Across_Generative_Models_CVPR_2023_paper.html
- **Key idea:** No learned classifier — nearest-neighbor / linear probe on frozen
  CLIP feature space. Fake-vs-real without training avoids the "real as sink
  class" failure.
- **Metrics:** +15.07 mAP, +25.9% acc over prior SOTA on unseen diffusion and
  autoregressive generators.
- **Relevance:** Proof that VLM feature spaces generalize; baseline for our MVP.

### FatFormer: Forgery-aware Adaptive Transformer
- **Authors:** Liu et al.
- **Venue/date:** CVPR 2024
- **Link:** https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Forgery-aware_Adaptive_Transformer_for_Generalizable_Synthetic_Image_Detection_CVPR_2024_paper.html
- **Key idea:** CLIP ViT + forgery-aware adapters injecting local image +
  frequency-domain traces, plus language-guided contrastive alignment with
  real/fake text prompts.
- **Metrics:** ~98% acc on unseen GANs, ~95% on unseen diffusion models (trained
  only on 4-class ProGAN).
- **Relevance:** Strong numbers but Gemini critique flags the frequency branch as
  fragile under DCT re-quantization — treat accuracy as a lab ceiling.

### VFM forensic priors (DINOv3 line: REM, MIRROR)
- **Venue/date:** cited within HEDGE (arXiv 2604.03555), 2026
- **Key idea:** Self-supervised VFMs (DINOv3) already encode forensic signal;
  a linear probe on frozen DINOv3 reportedly beats purpose-built detectors on
  in-the-wild benchmarks. REM estimates a "real distribution envelope"; MIRROR
  projects patch features onto a learned real-image manifold.
- **Relevance:** Anchors our backbone choice — start with frozen DINOv2/v3.

### Face foundation models + metric learning
- **Authors:** S. Mylonas, S. Papadopoulos
- **Venue/date:** arXiv 2508.19730, 2025
- **Link:** https://arxiv.org/abs/2508.19730
- **Key idea:** Face-recognition foundation model embeddings + metric learning
  to improve cross-dataset generalization of deepfake detection.
- **Relevance:** Cheap trick worth testing — face-domain pretraining may beat
  generic ImageNet/CLIP features for face crops.

---

## Ensembles & in-the-wild SOTA

### HEDGE: Heterogeneous Ensemble for Detection of AI-Generated Images
- **Authors:** F. Wu, D. Lu, M. Yao, X. Xu, F. Guo
- **Venue/date:** arXiv 2604.03555, 2026 (4th place NTIRE 2026 challenge)
- **Link:** https://arxiv.org/abs/2604.03555
- **Key idea:** Three complementary routes: (A) DINOv3-Huge with staged data
  expansion + augmentation escalation, (B) same backbone at higher input
  resolution for fine-grained cues, (C) MetaCLIP2-Giant for backbone diversity.
  Logit-space weighted fusion + dual-gating to suppress branch outliers.
- **Metrics:** SOTA on multiple in-the-wild benchmarks.
- **Relevance:** Template for our heavy/offline tier. Structured heterogeneity
  (data × resolution × backbone) > single strong model.

### FeatDistill: Multi-Expert Ensemble via Feature Distillation
- **Authors:** Z. Tu et al.
- **Venue/date:** arXiv 2603.21939, 2026
- **Link:** https://arxiv.org/abs/2603.21939
- **Key idea:** Distill an ensemble of detection experts into a deployable model.
- **Relevance:** If ensemble wins offline, distillation is the path to production
  latency.

---

## Bias-free training & generalization methodology

### Generalized Design Choices for Deepfake Detectors
- **Authors:** L. Pellegrini, S. Pandolfini, D. Maltoni, et al.
- **Venue/date:** arXiv 2511.21507, v2 Nov 2025
- **Link:** https://arxiv.org/abs/2511.21507
- **Key idea:** Model-agnostic study on AI-GenBench (generators temporally
  ordered to simulate new releases). Isolates which design choices — augmentation
  pipeline, training duration, crop-vs-resize preprocessing, multiclass labeling,
  incremental updates — actually drive future-generator generalization across
  ResNet50-CLIP, ViT-L-CLIP, DINOv2, EfficientNet-B0.
- **Relevance:** Read before designing our training pipeline; provides the
  augmentation/preprocessing recipe and a continual-learning protocol for new
  generators.

### B-Free: A Bias-Free Training Paradigm
- **Venue/date:** CVPR 2025
- **Link:** https://openaccess.thecvf.com/content/CVPR2025/html/Guillaro_A_Bias-Free_Training_Paradigm_for_More_General_AI-generated_Image_Detection_CVPR_2025_paper.html
- **Key idea:** Generate fakes from the same real images (stable-diffusion
  regeneration) so real/fake pairs are semantically aligned; content-based
  augmentation kills shortcut features.
- **Relevance:** Fixes the "detector learns dataset bias, not forgery" problem
  that tanks wild performance.

### DDA: Dual Data Alignment (NeurIPS 2025)
- **Link:** https://proceedings.neurips.cc/paper_files/paper/2025/file/991c9f1799cfeebb4217baaacc462d86-Paper-Conference.pdf
- **Key idea:** Align both content and format biases between real/fake data;
  shows detectors exploit image format/size as non-causal features.
- **Relevance:** Confirms we must match JPEG/resize distributions between real
  and fake training samples or the model cheats.

### VIB-Net: Variational Information Bottleneck (CVPR 2025)
- **Link:** https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_Towards_Universal_AI-Generated_Image_Detection_by_Variational_Information_Bottleneck_Network_CVPR_2025_paper.html
- **Key idea:** VIB objective on pretrained features to suppress
  authentication-irrelevant features that carry superficial biases.
- **Metrics:** +12.48% mAP / +23.59% acc vs SOTA on unseen generator families
  (17 generators tested).
- **Relevance:** Lightweight head-level method composable with DINOv2 backbone.

### Present and Future Generalization of Synthetic Image Detectors
- **Venue/date:** arXiv 2409.14128, 2024
- **Link:** https://arxiv.org/abs/2409.14128
- **Key idea:** Systematic generalization study: no detector is universal;
  rescaling alone is a major attack vector; private generators (DALL·E,
  Midjourney) much harder than open ones.
- **Relevance:** Guidelines for training robust detectors; expect partial
  failure and design for it (calibrated abstention, human escalation).

---

## Social-media compression robustness (critical for our use case)

### Bridging the Gap: Social Network Compression Emulation (video)
- **Authors:** A. Montibeller, D. Shullani, D. Baracchi, A. Piva, G. Boato
- **Venue/date:** arXiv 2508.08765, v2 Aug 2025
- **Link:** https://arxiv.org/abs/2508.08765
- **Key idea:** Infer a platform's proprietary compression/resize parameters from
  <50 uploaded probe videos, then build a **local emulator** reproducing
  Facebook/YouTube/Bluesky processing on any dataset — no API needed. Detectors
  fine-tuned on emulated video ≈ trained on really-shared video.
- **Relevance:** Directly actionable — we can generate platform-realistic
  training data at scale. Do this for augmentation.

### ODDN: Open-World Deepfake Detection on Online Social Networks
- **Venue/date:** AAAI 2025 — https://doi.org/10.1609/aaai.v39i1.32063
- **Key idea:** Open-world data aggregation (paired + unpaired raw/compressed
  data) + compression-discard gradient correction so the model is insensitive to
  compression variants.
- **Metrics:** 71.4% mean acc across 17 datasets vs 69.2% next-best (QAD).
- **Relevance:** Shows how to train when paired raw/compressed data is scarce.

### PLADA: Pay Less Attention to Deceptive Artifacts
- **Venue/date:** arXiv 2506.20548, 2025
- **Link:** https://arxiv.org/abs/2506.20548
- **Key idea:** Block Effect Eraser (dual-stage attention over JPEG block
  artifacts) + open data aggregation. Identifies compression "block effects" as
  deceptive cues detectors wrongly rely on.
- **Metrics:** ≥7% margin over SOTA across 26 datasets under OSN compression.
- **Relevance:** Compression artifacts themselves are a shortcut feature — our
  augmentation must cover multiple codecs/bitrates so the model can't exploit
  them.

---

## Video & multimodal detection

### AVFF: Audio-Visual Feature Fusion
- **Authors:** T. Oorloff et al. (Reality Defender)
- **Venue/date:** arXiv 2406.02951, 2024
- **Link:** https://arxiv.org/abs/2406.02951
- **Key idea:** Self-supervised audio-visual feature fusion — learns real
  audio-visual correspondence, flags dissonance without needing paired fake
  training data.
- **Relevance:** Good auxiliary signal for talking-head video; useless on silent
  or music-backed clips (common on social platforms).

### CAD: Cross-Modal Alignment and Distillation
- **Authors:** Y. Du et al.
- **Venue/date:** arXiv 2505.15233, 2025
- **Link:** https://arxiv.org/abs/2505.15233
- **Key idea:** Multimodal framework arguing modality-specific artifact cues are
  unreliable alone; aligns and distills cross-modal features for video deepfakes.
- **Relevance:** Escalation-tier candidate when audio is present.

### Next-Frame Feature Prediction (multimodal)
- **Authors:** A. Anshul, S. Gopal, D. Rajan, E. S. Chng
- **Venue/date:** arXiv 2511.10212, 2025
- **Link:** https://arxiv.org/abs/2511.10212
- **Key idea:** Predict next-frame features — real videos are predictable, fakes
  deviate; supports temporal localization of spliced fake segments.
- **Relevance:** Self-supervised temporal signal; good for partially-manipulated
  videos (see FakeParts).

### Continual video deepfake detection
- **Venue/date:** arXiv 2609.03446, Sep 2026
- **Link:** https://arxiv.org/abs/2609.03446
- **Key idea:** Preserve knowledge across space and time as new video forgery
  patterns emerge — continual learning for video detectors.
- **Relevance:** Matches our need to update the detector as generators evolve.

### FakeParts: localized partial manipulations
- **Venue/date:** arXiv 2508.21052, 2025
- **Link:** https://arxiv.org/abs/2508.21052
- **Key idea:** New deepfake family: subtle localized edits (expression changes,
  short fake segments) in otherwise real video. Full-video classifiers miss them.
- **Relevance:** Motivates spatial/temporal localization, not just binary labels.

### MAVOS-DD: multilingual open-set AV benchmark
- **Venue/date:** arXiv 2505.11109, 2025
- **Link:** https://arxiv.org/abs/2505.11109
- **Key idea:** ~350 h of real/fake video, 8 languages, open-set evaluation of
  audio-video detectors.
- **Relevance:** Evaluation target if we use audio-visual cues.

### Interpretable video detection via explicit forensic features
- **Venue/date:** arXiv 2610.03380, Oct 2026
- **Link:** https://arxiv.org/abs/2610.03380
- **Key idea:** Explicit spatial + temporal forensic features (not learned
  end-to-end) with temporal modeling for interpretability.
- **Relevance:** Alternative to black-box video models; moderator-facing
  explanations.

---

## Adjacent / supporting

- **Diffusion Deepfake** (arXiv 2404.01579): diffusion-generated fakes break
  GAN-era detectors; motivation for diffusion-inclusive training data.
- **1M-Deepfakes Detection Challenge / AV-Deepfake1M** (arXiv 2409.06991):
  detection + temporal localization when small fake segments are mixed into
  real videos.
- **DeepFake-O-Meter v2.0** (arXiv 2404.13146): open platform integrating many
  SOTA detectors — useful for baseline comparisons.
- **Inclusion 2024 Global Multimedia Deepfake Detection Challenge**
  (arXiv 2412.20833): multi-dimensional face forgery detection (edit, synthesis,
  generation, Photoshop).
- **AI-generated Image Detection: Passive or Watermark?** (arXiv 2411.13553):
  passive vs watermark/provenance approaches — argues for combining both;
  relevant to pairing our detector with C2PA provenance checks.
- **Explainable Deepfake Detection Challenge** (arXiv 2607.21007): field moving
  toward explaining evidence, not just binary scores.
- **Open-set facial deepfake detection** (arXiv 2503.08055): open-set paradigm
  for unknown forgery methods.
- **Media Intelligence Platform** (arXiv 2610.27974, Oct 2026): multimodal
  platform for synthetic/manipulated content — closest "product" reference.
- **SocialDF** (arXiv 2506.05538): social-media deepfake benchmark + LipFD
  baseline for lip-sync detection.
- **Age-Diverse Deepfake Dataset** (arXiv 2508.06552): demographic/age bias in
  deepfake datasets — fairness check for our model.

---

## 2025–2026 Frontier: Generative Video, Social Benchmarks, & Physics Priors

### Omni-Fake: 1M+ Scale Multi-Modal Social Media Benchmark
- **Venue/date:** CVPR 2026
- **Link:** CVPR 2026 Open Access
- **Key idea:** 1,000,000+ sample omni-domain benchmark specifically targeting
  social-media distribution across images, audio, full video, and audio-video
  talking heads. Focuses on out-of-distribution (OOD) generalization across
  unseen modern generators (Sora, Kling, Wan2.1, SD3, Flux).
- **Relevance:** Essential replacement for saturated academic benchmarks;
  evaluates our entire multimodal pipeline under actual platform noise distributions.

### SIDA: Social Media Image Deepfake Detection, Localization, and Explanation
- **Venue/date:** CVPR 2025
- **Link:** CVPR 2025 Open Access
- **Key idea:** Dataset and framework targeting real-world social-media platform
  propagation and misinformation risks. Addresses the gap where clean lab images
  fail to reflect compression artifacts, re-encoding cascades, and selective
  tampering on platforms.
- **Relevance:** Direct validation dataset for spatial manipulation localization
  and social-media distribution survival.

### Normalized Spatiotemporal Gradient (NSG): Physics-Driven Diffusion Video Forensics
- **Authors:** Highlighted at NeurIPS
- **Venue/date:** NeurIPS 2025 / 2026
- **Key idea:** Rather than memorizing pixel artifacts, formulates video temporal
  evolution through probability flow conservation principles. Tracks the ratio
  of spatial probability gradients to temporal density changes. Exposes how
  diffusion video models (Sora, Wan, Kling) violate optical flow and temporal
  continuity physics.
- **Relevance:** Ideal feature extractor for our Stage 3 video escalation tier;
  captures physical discrepancies that persist even through heavy video compression.

### Spatiotemporal Adapters (StA) for Foundation Vision Backbones
- **Venue/date:** CVPR 2025
- **Key idea:** Keeps heavyweight vision foundation models (DINOv2 / CLIP) frozen
  and introduces lightweight, two-stream 3D convolutional adapters with varying
  kernel sizes. Decouples spatial learning from temporal dynamics and explicitly
  targets "Facial Feature Drift" (FFD) across consecutive frames.
- **Relevance:** High-efficiency architecture for our Stage 2 video core detector;
  avoids fine-tuning massive ViT weights while unlocking temporal coherence detection.

### DAID: Fair Deepfake Detectors Can Generalize Under Compression
- **Venue/date:** NeurIPS 2025
- **Key idea:** Investigates demographic fairness and cross-domain generalization
  under aggressive lossy compression. Models causal confounders between platform
  downsampling/re-encoding and facial demographic attributes to prevent false
  positive spikes on underrepresented skin tones.
- **Relevance:** Prevents severe demographic bias and false positive spikes in
  social media moderation systems.

---

## Resource-Constrained, OOD Calibration & Trustworthy Forensics (2025–2026)

### Split-Conformal Prediction for Out-of-Distribution Deepfake Detection
- **Venue/date:** arXiv / ICML Trustworthy AI 2024–2025
- **Key idea:** Standard post-hoc temperature scaling collapses when presented
  with unseen generative models (OOD). Conformal prediction applies distribution-free
  calibration to output prediction sets rather than overconfident point probabilities.
  Binds the platform False Positive Rate rigorously at $\le \alpha$ (e.g. 1%) by computing
  empirical quantile non-conformity thresholds on held-out authentic validation data.
- **Relevance:** Directly addresses the 16.8% FNR calibration drift observed on
  unseen Midjourney data; provides mathematically backed abstain thresholds with zero
  retraining cost.

### Residual-Guided Temporal Pooling (RGTP) for Video Deepfake Detection
- **Venue/date:** 2026
- **Key idea:** Traditional video detectors apply uniform temporal pooling, diluting
  sparse or localized artifacts. RGTP introduces adaptive residual-driven aggregation,
  weighting high-discrepancy frames heavily while retaining an ultra-lightweight compute
  footprint without requiring heavy 3D-CNN or multi-frame video transformer pretraining.
- **Relevance:** Foundation for our Stage 4 video extension; validates top-$k$ logit
  pooling over sparse keyframes (3–5 fps).

### Light-wDEF: Lightweight and Reproducible Video Forensics
- **Venue/date:** 2025
- **Key idea:** Standardized, resource-constrained benchmarking framework designed
  for lightweight accelerators (e.g., single T4 / local MPS GPUs). Demonstrates that
  parameter-efficient tuning of 2D foundation backbones with minimal temporal heads
  rivals massive end-to-end spatiotemporal models.
- **Relevance:** Directly informs our hackathon development constraints, proving that
  lightweight temporal adapters on frozen DINOv2 remain competitive with multi-node clusters.



---

## Gemini critique of candidate approach families

(Solicited via gemini-cli MCP, skeptical-reviewer persona; full transcript
paraphrased.)

**Ranking for our use case (highest → lowest viability):**

1. **Frozen VFM backbones (DINOv2/v3, CLIP) + light head** — best
   generalization-per-FLOP. Weakness: linear probes discard local high-frequency
   boundary traces; per-frame ViT cost on video.
2. **Bias-free / data-alignment training (B-Free, DDA, VIB-Net)** — raises the
   floor on unseen data. Weakness: can discard weak-but-real forensic signal
   along with shortcuts.
3. **Compression-robust OSN methods (ODDN, PLADA, emulation)** — necessary for
   deployment. Weakness: brittle to codec shifts (WebP/AV1, cascaded
   recompression); model can learn the compression mask instead of invariance.
4. **Adapted transformers (FatFormer)** — strong lab numbers but frequency
   adapters degrade under DCT re-quantization.
5. **Audio-visual fusion (AVFF, CAD)** — only when speech audio exists; desync
   and silent clips cause false positives.
6. **Heterogeneous ensembles (HEDGE)** — in-the-wild SOTA but P99 latency and
   cost too high for front-door; use as escalation tier.
7. **Artifact/frequency CNNs (NPR, LGrad, DIRE)** — collapse under first social
   re-encode; DIRE also ~100× too slow. Don't rely on them.
8. **MLLM explainable detectors** — hallucinated rationales, huge latency;
   unsuitable for ingest. (Also: Grad-CAM/attention maps highlight benign
   regions regardless of label — don't ship them to moderators.)

**Top-3 real-world failure modes flagged:**

1. **Cascaded recompression** (WhatsApp → CapCut → platform): multi-step lossy
   chains destroy boundary features; academia only tests single-step compression.
2. **Beautification-filter false positives**: TikTok/WeChat/camera smoothing
   mimics GAN skin statistics → authentic flagged as fake.
3. **Temporal-detectors vs. mobile artifacts**: VFR, dropped frames, motion blur
   read as temporal incoherence → false positives on real video.

**Over-claims to distrust:**

- "99%+ on FaceForensics++" — solved benchmark, memorization of pairs/compression
  shortcuts.
- "Frequency generalization" — only holds when resolution/compression
  distributions are uniform in eval.
- "Zero-shot diffusion detection" — breaks under LoRA/ControlNet/inpainting
  (partial generation violates full-frame assumptions).
- Attention-map explainability — shows where model looked, not why.

See `docs/model-plan.md` for the resulting architecture decision.
