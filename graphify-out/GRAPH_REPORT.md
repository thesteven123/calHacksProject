# Graph Report - calHacksProject  (2026-10-24)

## Corpus Check
- Corpus is ~26,202 words - fits in a single context window. You may not need a graph.

## Summary
- 695 nodes · 1327 edges · 57 communities (30 shown, 27 thin omitted)
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 161 edges (avg confidence: 0.89)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Compression & Beauty Augmentation
- CLI Scripts & Fusion Entry Points
- Tinker VLM Training Route
- Credentials & Script Plumbing
- Architecture Spec & Benchmarks
- Calibration & Metrics Math
- Calibrated Inference Pipeline
- Nested LOGO Config
- Local Tiny-GenImage Config
- Adversarial & Systems Roles
- Smoke Test Config
- MVP Image Config
- In-the-Wild Failure Modes
- Hackathon Funnel & Distillation
- Training Loop & Beta Schedule
- DINOv2/v3 Backbone Wrapper
- Video & Adversarial Mitigations
- Research Workflow & Datasets
- Review Findings & Credentials
- Detector & VIB Head Modules
- Compute Budget & Route Results
- Crop Policy & Video Sampling
- Core Design Decisions D2-D7
- LoRA Implementation
- DeepfakeDetector Class
- Loss Functions
- VLM Fusion & Shortcut Fix
- Model Unit Tests
- LOGO Leakage & Protocol
- Foundation-Model Detectors
- Temporal Adapter
- Forensic Researcher Role
- Platform Transcoding Emulation
- Aspect-Ratio Shortcut & Bias-Free Training
- Partial & AV Fakes
- Isolated: Age-Diverse Deepfake Dataset (arXiv 25
- Isolated: AVFF: Audio-Visual Feature Fusion (arX
- Isolated: ODDN: Open-World Deepfake Detection on
- Isolated: dfd
- Isolated: Continual Video Deepfake Detection (ar
- Isolated: DeepFake-O-Meter v2.0 (arXiv 2404.1314
- Isolated: Diffusion Deepfake (arXiv 2404.01579)
- Isolated: Explainable Deepfake Detection Challen
- Isolated: Face Foundation Models + Metric Learni
- Isolated: Generalized Design Choices for Deepfak
- Isolated: Inclusion 2024 Global Multimedia Deepf
- Isolated: Interpretable Video Detection via Expl
- Isolated: MAVOS-DD: Multilingual Open-Set AV Ben
- Isolated: Media Intelligence Platform (arXiv 261
- Isolated: Open-set Facial Deepfake Detection (ar
- Isolated: AI-generated Image Detection: Passive 
- Isolated: SocialDF + LipFD (arXiv 2506.05538)

## God Nodes (most connected - your core abstractions)
1. `DeepfakeDetector` - 26 edges
2. `PlatformTranscodingEmulator` - 21 edges
3. `PTEConfig` - 20 edges
4. `DetectionPipeline` - 19 edges
5. `DetectorTransform` - 17 edges
6. `main()` - 16 edges
7. `main()` - 15 edges
8. `Config` - 15 edges
9. `main()` - 14 edges
10. `build_base_datasets()` - 14 edges

## Surprising Connections (you probably didn't know these)
- `has_provenance_manifest (reads at most 4 MiB)` --semantically_similar_to--> `C2PA / Content Credentials Connector (Stage 0)`  [INFERRED] [semantically similar]
  CODE_REVIEW.md → docs/tooling-and-connectors.md
- `main()` --uses--> `DetectorTransform`  [INFERRED]
  scripts/evaluate.py → src/dfd/data/transforms.py
- `main()` --uses--> `DeepfakeDetector`  [INFERRED]
  scripts/evaluate.py → src/dfd/models/detector.py
- `main()` --calls--> `predict()`  [INFERRED]
  scripts/evaluate.py → src/dfd/training/trainer.py
- `core_logits()` --uses--> `PTEConfig`  [INFERRED]
  scripts/fuse_routes.py → src/dfd/config.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Shortcut/bias controls across PTE, DDA, B-Free/SBI and beautification augmentation** — docs_architecture_spec_platform_transcoding_emulation, docs_architecture_spec_format_normalization_dda, docs_architecture_spec_aligned_pairs_bfree_sbi, docs_architecture_spec_beautification_jitter [INFERRED 0.85]
- **Held-out in-the-wild evaluation benchmarks** — docs_architecture_spec_omni_fake, docs_architecture_spec_sida, docs_architecture_spec_deepfake_eval_2024, docs_architecture_spec_truefake [EXTRACTED 1.00]
- **Adversarial review vulnerabilities and mitigations for the tiered funnel** — research_critiques_adversarial_review_tiered_funnel, research_critiques_adversarial_review_stage1_screener_leakage, research_critiques_adversarial_review_distilled_vision_student, research_critiques_adversarial_review_white_box_uap_transfer, research_critiques_adversarial_review_lora_backbone_perturbation, research_critiques_adversarial_review_beauty_filter_false_positives, research_critiques_adversarial_review_virality_gated_escalation [EXTRACTED 1.00]
- **Stage 2 Core Detector Design Decisions (D1-D5)** — docs_implementation_architecture_dinov2_backbone, docs_implementation_architecture_lora_qv, docs_implementation_architecture_multi_layer_features, docs_implementation_architecture_vib_head, docs_implementation_architecture_beauty_filter_head, docs_implementation_architecture_stage2_core_model [EXTRACTED 1.00]
- **Tiered Moderation Funnel: Provenance -> Student -> Core -> VLM Escalation -> Human Review** — docs_tooling_and_connectors_c2pa_connector, docs_implementation_architecture_stage1_student, docs_implementation_architecture_stage2_core_model, docs_implementation_architecture_stage3_vlm_route, docs_implementation_architecture_calibration_verdicts, docs_tooling_and_connectors_moderation_escalation_connector [INFERRED 0.85]
- **Evaluation Leakage & Shortcut Controls** — code_review_logo_holdout_leak, code_review_split_val_holdout, docs_implementation_architecture_nested_logo_protocol, docs_implementation_architecture_square_crop_shortcut_fix, docs_implementation_architecture_bias_controls, code_review_missing_generator_metadata [INFERRED 0.75]
- **Hackathon Deliverables: calibration, video, student, explanations, demo** — research_evidence_and_next_steps_split_conformal_prediction, research_evidence_and_next_steps_temporal_adapter, research_evidence_and_next_steps_mobilenetv4_student_distillation, research_evidence_and_next_steps_vlm_forensic_rationale, research_evidence_and_next_steps_hackathon_demo [EXTRACTED 1.00]
- **Social-Media Compression Robustness Methods** — research_papers_social_network_compression_emulation, research_papers_oddn, research_papers_plada, research_papers_truefake [EXTRACTED 1.00]
- **Bias-Free / Data-Alignment Training Family** — research_papers_b_free, research_papers_dda, research_papers_vib_net, research_papers_generalized_design_choices [EXTRACTED 1.00]

## Communities (57 total, 27 thin omitted)

### Community 0 - "Compression & Beauty Augmentation"
Cohesion: 0.06
Nodes (32): PTEConfig, apply_beauty_filter(), encode(), PlatformTranscodingEmulator, PTEMeta, resize_short_side(), DetectorTransform, to_tensor() (+24 more)

### Community 1 - "CLI Scripts & Fusion Entry Points"
Cohesion: 0.05
Nodes (25): main(), main(), verdict_rates(), main(), main(), Config, DataConfig, build_base_datasets() (+17 more)

### Community 2 - "Tinker VLM Training Route"
Cohesion: 0.06
Nodes (10): main(), VLMRouteConfig, square_crop(), VLMScorer, build_renderer(), DeepfakeSFTBuilder, DeepfakeSFTDataset, prompt_messages() (+2 more)

### Community 3 - "Credentials & Script Plumbing"
Cohesion: 0.08
Nodes (6): core_logits(), main(), TrainConfig, paired_pte_images(), test_load_dotenv_fills_unset_and_empty_but_keeps_real_values(), test_load_dotenv_missing_file_is_noop()

### Community 4 - "Architecture Spec & Benchmarks"
Cohesion: 0.06
Nodes (33): Aligned Pairs (B-Free / SBI), Social Media Deepfake Detection Architecture Specification, Audio-Visual Sync Check (AVFF/CAD), Beautification Jitter, Benchmark & Validation Protocol, DAID Demographic Fairness Audit, Deepfake-Eval-2024, Facial Feature Drift (FFD) (+25 more)

### Community 5 - "Calibration & Metrics Math"
Cohesion: 0.09
Nodes (16): calibrate(), fit_temperature(), fit_fusion(), Fusion, paired_bootstrap(), binary_metrics(), expected_calibration_error(), report_to_markdown() (+8 more)

### Community 6 - "Calibrated Inference Pipeline"
Cohesion: 0.18
Nodes (5): Calibration, DetectionPipeline, dhash(), has_provenance_manifest(), Result

### Community 7 - "Nested LOGO Config"
Cohesion: 0.09
Nodes (21): data, generator_column, hf_data_files, hf_train_split, hf_val_split, holdout_generators, num_workers, selection_generators (+13 more)

### Community 8 - "Local Tiny-GenImage Config"
Cohesion: 0.10
Nodes (20): data, generator_column, hf_data_files, hf_train_split, hf_val_split, holdout_generators, num_workers, source (+12 more)

### Community 9 - "Adversarial & Systems Roles"
Cohesion: 0.14
Nodes (14): Agent Role: Adversarial Reviewer & Red Teamer, Agent Workspace Overview & Role Allocation, Agent Role: Social Media Production Systems Engineer, Devin Agent (Architecture Design & Implementation), Beauty-filter / computational-photography false positives (triplet disentanglement), Distilled lightweight vision student (MobileNetV4-Conv-Small / FastViT-T8), Adversarial ML & Systems Architecture Critique, Ingestion bottlenecks (decode, tracking, PCIe over ViT FLOPs) (+6 more)

### Community 10 - "Smoke Test Config"
Cohesion: 0.10
Nodes (19): data, source, synthetic_image_size, synthetic_train_size, synthetic_val_size, model, backbone, feature_layers (+11 more)

### Community 11 - "MVP Image Config"
Cohesion: 0.11
Nodes (17): data, generator_column, hf_dataset, hf_train_split, hf_val_split, holdout_generators, num_workers, source (+9 more)

### Community 12 - "In-the-Wild Failure Modes"
Cohesion: 0.21
Nodes (14): VLM Forensic Rationale (2-sentence explanations), Failure Mode: Cascaded Recompression, Deepfake-Eval-2024 (arXiv 2503.02857), FatFormer: Forgery-aware Adaptive Transformer (CVPR 2024), Deepfake Detection — Paper Findings, Artifact/Frequency CNNs (NPR, LGrad, DIRE), Gemini Critique of Candidate Approach Families, MLLM Explainable Detectors (+6 more)

### Community 13 - "Hackathon Funnel & Distillation"
Cohesion: 0.14
Nodes (11): Abstain / Escalate Decision Band, scripts/distill.py, BCE + KL Distillation Loss, Hackathon Demo & Presentation Architecture (Gradio/Streamlit on Modal), Hackathon Execution Funnel (Phases A–D), Platform Transcoding Emulation (PTE), SocialMediaDegradationEmulator, Tiered Funnel (Stage 0 → Stage 1 → Stage 2 → escalation) (+3 more)

### Community 14 - "Training Loop & Beta Schedule"
Cohesion: 0.18
Nodes (3): BetaSchedule, predict(), Trainer

### Community 15 - "DINOv2/v3 Backbone Wrapper"
Cohesion: 0.22
Nodes (4): ModelConfig, FoundationBackbone, _load_hf_backbone(), _transformer_blocks()

### Community 16 - "Video & Adversarial Mitigations"
Cohesion: 0.19
Nodes (11): LoRA on top-3 block Wq/Wv projections (manifold perturbation), Spatiotemporal Adapter temporal evasion (top-k pooling, dynamic keyframes), White-box foundation transfer & universal adversarial perturbations, Leave-One-Generator-Out (LOGO) Sweep, RunPod ($50 credit, batch GPU training), TemporalAdapter (src/dfd/models/temporal.py), Quick Video Datasets (Celeb-DF-v2 subset, DeepFakeFace, OpenForensics/SIDA), Facial Feature Drift (FFD) (+3 more)

### Community 17 - "Research Workflow & Datasets"
Cohesion: 0.21
Nodes (12): Antigravity (ML Literature Review Agent), Research Focus Areas (generalization, compression, benchmarks), Literature Review Workflow (arxiv / gemini-cli MCP), Tiny-GenImage Shard Interleaving (balanced subsets), Bias Controls (DDA format alignment, B-Free aligned pairs, size jitter), Held-Out In-the-Wild Eval Sets (Deepfake-Eval-2024, TrueFake, SIDA, Omni-Fake), Tiny-GenImage Dataset (TheKernel01/Tiny-GenImage), Tooling, MCPs, Skills & Connectors Recommendation (+4 more)

### Community 18 - "Review Findings & Credentials"
Cohesion: 0.17
Nodes (11): scripts/check_credentials.py (read-only credential check), CODE_REVIEW.md — Code Review & Audit, Finding (High): DeepfakeDetector.load Unsafe torch.load Pickle, Implementation Architecture — Deepfake Detector (DFD), Compute Plan (MPS dev, cloud GPU, Tinker, TensorRT), D1: Frozen DINOv2-base Backbone, DINOv3 Backbone (gated, model.layer blocks), Milestones S1-S7 (+3 more)

### Community 20 - "Compute Budget & Route Results"
Cohesion: 0.20
Nodes (9): Core DINOv2 Detector (DINOv2 + LoRA + VIB), Fused Route (DINOv2 + VLM), Local Mac MPS (smoke tests), Midjourney Holdout Test Set, Modal ($30 credit, serverless testing & demo host), VLM Route (Qwen3.6-35B-VL LoRA), Empirical Evidence & Further Implementation Research Roadmap, Post-hoc Temperature Scaling (+1 more)

### Community 21 - "Crop Policy & Video Sampling"
Cohesion: 0.20
Nodes (9): Follow-up: Native-Resolution 224 Center Crop Misses Off-Center Edits, Finding: Multi-Crop Verdicts Without Matching Calibration, Finding: Video Multi-Crop Ignored for Temporal Checkpoints, Finding: Video Inference Samples Only Start of Long Videos, D11: Temperature Scaling + 3-Way Verdict (real/abstain/fake), Multi-Crop Scoring (3x3 grid + centre, top-25% mean), D6: Temporal Adapter + Top-k Logit Pooling + Keyframe Selection, Moderation Case Management & Escalation Connector (+1 more)

### Community 22 - "Core Design Decisions D2-D7"
Cohesion: 0.22
Nodes (10): has_provenance_manifest (reads at most 4 MiB), Finding: Provenance Check Reads Entire Media File, D2: LoRA r=8 on Q/V of Last 3 ViT Blocks, D3: Multi-Layer Token Features (CLS + mean-patch, layers 4/8/10/12), D7: Stage 1 Distilled MobileNetV4-Conv-Small Student, Stage 2 Core Model (DINOv2 + LoRA + VIB), Tiered Inference Funnel (Stage0 hash -> Stage1 -> Stage2 -> Stage3), D4: VIB Head with Annealed Beta (+2 more)

### Community 25 - "Loss Functions"
Cohesion: 0.31
Nodes (4): detector_loss(), distillation_loss(), kl_standard_normal(), test_loss_backward_reaches_lora()

### Community 26 - "VLM Fusion & Shortcut Fix"
Cohesion: 0.32
Nodes (6): AGENTS.md — Project Roles & Working Notes, Build Log & Results (§11), HEDGE-Style Dual-Gate (drop route deviating >3 sigma), Route Fusion (logistic regression on core + VLM logits), D8: Stage 3 VLM Route (Qwen3.6-VL LoRA SFT on Tinker), dfd-experiment Skill

### Community 27 - "Model Unit Tests"
Cohesion: 0.25
Nodes (6): test_beta_schedule_and_distillation(), test_image_forward_shapes_and_lora_only_trainable_in_backbone(), test_select_keyframes_includes_scene_change(), test_topk_pool_respects_mask(), test_transformer_blocks_found_for_dinov3_layout(), test_video_forward_with_temporal_adapter_and_mask()

### Community 28 - "LOGO Leakage & Protocol"
Cohesion: 0.38
Nodes (7): Finding: LOGO Holdouts Leak into Validation/Selection/Calibration, Finding: Generator Holdouts Silently Ignored Without Metadata Column, split_val_holdout() (val_select / val_test split), Leave-One-Generator-Out (LOGO) Generalization Split, D10: Metrics AUC, TPR@1%FPR, TPR@0.1%FPR, ECE per Slice, Nested LOGO Protocol (fit / select GLIDE / test Midjourney), deepfake-forensic-eval Skill

### Community 29 - "Foundation-Model Detectors"
Cohesion: 0.40
Nodes (5): FeatDistill: Multi-Expert Ensemble via Feature Distillation (arXiv 2603.21939), HEDGE: Heterogeneous Ensemble (arXiv 2604.03555), NTIRE 2026 Robust AI-Generated Image Detection Challenge (arXiv 2604.11487), UnivFD: Towards Universal Fake Image Detectors (CVPR 2023), VFM Forensic Priors (DINOv3 line: REM, MIRROR)

### Community 32 - "Platform Transcoding Emulation"
Cohesion: 0.67
Nodes (4): D5: Auxiliary Beauty-Filter Head, D9: Platform Transcoding Emulation (PTE), compression-emulator Skill, ffmpeg (video codec chains)

### Community 33 - "Aspect-Ratio Shortcut & Bias-Free Training"
Cohesion: 0.50
Nodes (3): DEFAULT_SQUARE_SIZE = 448, B-Free: A Bias-Free Training Paradigm (CVPR 2025), DDA: Dual Data Alignment (NeurIPS 2025)

### Community 34 - "Partial & AV Fakes"
Cohesion: 0.67
Nodes (3): 1M-Deepfakes Detection Challenge / AV-Deepfake1M (arXiv 2409.06991), FakeParts: Localized Partial Manipulations (arXiv 2508.21052), Next-Frame Feature Prediction (arXiv 2511.10212)

## Ambiguous Edges - Review These
- `Residual-Guided / Top-k Mean Logit Pooling` → `Failure Mode: Temporal Detectors vs Mobile Artifacts`  [AMBIGUOUS]
  research/papers.md · relation: conceptually_related_to
- `VLM Forensic Rationale (2-sentence explanations)` → `Omni-Fake: 1M+ Multi-Modal Social Media Benchmark (CVPR 2026)`  [AMBIGUOUS]
  research/evidence_and_next_steps.md · relation: cites

## Knowledge Gaps
- **70 isolated node(s):** `dfd`, `Stage 0: Metadata, Provenance & Deduplication (C2PA, PDQ/pHash, face detection 1.35x padding)`, `Stage 1: Front-Door Screener (MobileNetV4-Small / EfficientNet-B0 on DCT/residuals)`, `Frozen DINOv2-Base / DINOv3 (ViT-B/14) Backbone`, `Facial Feature Drift (FFD)` (+65 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 275 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **27 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Split-Conformal Prediction with Monotone Non-Conformity Scores` connect `Hackathon Funnel & Distillation` to `Credentials & Script Plumbing`, `Compute Budget & Route Results`?**
  _High betweenness centrality (0.246) - this node is a cross-community bridge._
- **Are the 16 inferred relationships involving `DeepfakeDetector` (e.g. with `main()` and `main()`) actually correct?**
  _`DeepfakeDetector` has 16 INFERRED edges - model-reasoned connections that need verification._
- **What connects `dfd`, `Stage 0: Metadata, Provenance & Deduplication (C2PA, PDQ/pHash, face detection 1.35x padding)`, `Stage 1: Front-Door Screener (MobileNetV4-Small / EfficientNet-B0 on DCT/residuals)` to the rest of the system?**
  _70 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Compression & Beauty Augmentation` be split into smaller, more focused modules?**
  _Cohesion score 0.06388888888888888 - nodes in this community are weakly interconnected._
- **Why does `Agent Role: Social Media Production Systems Engineer` connect `Adversarial & Systems Roles` to `Hackathon Funnel & Distillation`?**
  _High betweenness centrality (0.116) - this node is a cross-community bridge._

### Low-confidence Hints
_AMBIGUOUS edges — the extractor was unsure. Verify before acting on these._

- **What is the exact relationship between `Residual-Guided / Top-k Mean Logit Pooling` and `Failure Mode: Temporal Detectors vs Mobile Artifacts`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `VLM Forensic Rationale (2-sentence explanations)` and `Omni-Fake: 1M+ Multi-Modal Social Media Benchmark (CVPR 2026)`?**
  _Edge tagged AMBIGUOUS (relation: cites) - confidence is low._