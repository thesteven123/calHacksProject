# Deepfake Detection Research & Architecture Project

## Division of Responsibilities

- **Antigravity (ML Literature Review Agent)**:
  - **Scope**: Exclusively performs literature review and academic/industry research.
  - **Responsibilities**: Search, analyze, and synthesize recent papers on deepfake generation (diffusion, autoregressive, flow-matching, GANs) and forensic detection; evaluate benchmarks (Omni-Fake, SIDA, Deepfake-Eval, TrueFake); conduct adversarial critiques and identify failure modes under social-media compression and camera pipelines.
  - **Boundaries**: Strictly no implementation or prototype code. All findings, paper analyses, and adversarial reviews are documented under `research/` (e.g., `research/papers.md`, `research/critiques/`).
- **Devin Agent (Architecture Design & Implementation)**:
  - **Scope**: In charge of designing the model architecture and executing the complete code implementation.
  - **Responsibilities**: Implement model architectures, feature extractors, loss formulations, training pipelines, data augmentation / compression emulators, hardware optimization (TensorRT, INT8/FP8), and production deployment services under `src/` and system architecture design in `docs/`.

## Devin Agent — Working Notes (Implementation)

- **Role:** Architecture design + full implementation. Source of truth for the build is `docs/implementation-architecture.md` (derived from `docs/architecture-spec.md`, `docs/model-plan.md`, and `research/critiques/adversarial_review.md`). Update it when design decisions change.
- **Code lives in** `src/dfd/` (Python package), `scripts/` (CLIs), `configs/` (YAML), `tests/` (pytest). Do not write implementation code into `research/`.
- **Environment:** `uv` venv at `.venv/` (Python 3.12), pinned by `requirements.lock`. Setup: `uv venv --python 3.12 .venv && uv pip sync --python .venv/bin/python requirements.lock && uv pip install --python .venv/bin/python --no-deps -e .`. Regenerate the lock with `uv pip compile pyproject.toml --extra dev --exclude-newer <today-7d> --python-version 3.12 -o requirements.lock`.
- **Tinker env is separate:** `.venv-tinker/` (`uv pip install --python .venv-tinker/bin/python -e ".[tinker,dev]"`), because `tinker-cookbook` downgrades transformers and huggingface-hub. Run `scripts/train_vlm_tinker.py` and `dfd.tinker_route.*` only from `.venv-tinker`.
- **Verify before finishing a step:** `.venv/bin/pytest -q` (~5 s) and `.venv/bin/python scripts/train.py --config configs/smoke.yaml` (synthetic data, ~1 min on MPS). Severity sweep: `.venv/bin/python scripts/evaluate.py --checkpoint runs/<run>/best.pt --config <cfg>`.
- **Run outputs** go to `runs/<name>/` (`best.pt`, `calibration.json`, `report.md`, `eval_report.md`, `metrics.jsonl`); git-ignored. The build log and results table live in `docs/implementation-architecture.md` §11.
- **Datasets:** Tiny-GenImage shards are interleaved real/fake with rotating generators, so a subset of shards stays balanced (`configs/tiny_genimage_local.yaml`). SD14 is absent from both splits. License is CC-BY-NC-SA (research only).
- **Tooling constraints:** Tinker trains LoRA on LLM/VLMs only (e.g., Qwen3.5/3.6-VL), so it backs the Stage 3 VLM route (`src/dfd/tinker_route/`). The DINOv2 core detector is trained with local or cloud PyTorch. Hugging Face hosts backbones (`facebook/dinov2-*`, DINOv3 gated) and training data (`TheKernel01/Tiny-GenImage`, etc.).
- **Secrets:** `TINKER_API_KEY`, `HF_TOKEN`, `WANDB_API_KEY` come from the environment or the gitignored project `.env` (mode 600). Scripts load it via `dfd.env.load_dotenv()`, which never overrides non-empty shell values and treats empty exports as unset. Never commit or print them.
- **Credential check (read-only, no billable calls):** `.venv-tinker/bin/python scripts/check_credentials.py`. It verifies the HF token, gated access (DINOv3 granted; AV-Deepfake1M pending as of 2026-10-24) and Tinker capabilities.
- **Evaluation protocol:** use nested LOGO (`data.selection_generators` for checkpoint choice, `data.holdout_generators` scored once after training; see `configs/tiny_genimage_nested.yaml`). Calibrate only on the `fit` split. `scripts/evaluate.py --split test --crop-mode both --save-preds` gives paired centre/multi-crop results. With ~300 reals, TPR@1%FPR differences under ~±0.08 are noise, so report bootstrap CIs.
- **VLM route input:** always square-crop both classes (`DEFAULT_SQUARE_SIZE`). Native-aspect input lets the VLM read image size and shape, which is a perfect label shortcut on GenImage (all fakes square). Scoring: `.venv-tinker/bin/python scripts/score_vlm_tinker.py --sampler <tinker://...> --config <cfg> --split select` (refuses `test` without `--allow-test`).
- **Modal (core-detector GPU runs):** auth lives in `~/.modal.toml` (workspace set up 2026-10-24; never read or print it). Client is in `.venv` (`modal==1.6.2`). Run LOGO rotations with `.venv/bin/modal run scripts/modal_logo.py --rotations GLIDE:Midjourney` (one job) or `--rotations all` (7 jobs, parallel after one cache-warming job). Jobs use L4 + 8 cores + 32 GiB (~$1.43/h list), with a 2 h timeout and at most 7 concurrent. Only `HF_TOKEN` is sent, as an ephemeral secret. Volumes: `dfd-hf-cache` (HF downloads), `dfd-runs` (outputs). Local copies go to `runs/modal/<tag>/`.
- **Crop mode:** inference defaults to `center`. `multi` is opt-in and needs `calibration_multicrop.json` (`evaluate.py --split fit --crop-mode multi --severities train_mix --save-calibration`).

## Research Workflow (Literature Review)

- Use the `arxiv` MCP tools to search and download recent papers on deepfake generation/detection. Use built-in `web_search`/`webfetch` for papers outside arXiv (CVPR/ICCV/NeurIPS open access, IEEE TIFS, blogs, Semantic Scholar).
- Use the `gemini-cli` MCP tools for deep analysis of long documents/papers (Gemini's large context window) and for second-opinion critiques of designs.
- Write all literature findings, survey notes, and adversarial critiques into markdown files under `research/` (e.g. `research/papers.md`, `research/critiques/`).
- Every paper noted should include: title, authors, venue/date, link, key idea, reported metrics/datasets, and relevance to this project.

## Research Focus Areas

- Image/video deepfake detection (CNN/ViT foundation backbones, frequency-domain vs. semantic features, blending boundaries, spatiotemporal physics)
- Generalization across modern generators (unseen diffusion models like Sora, Kling, Wan2.1, Flux, SD3)
- Social-media constraints: cascaded compression/re-encoding, mobile beautification filters, latency, scale, high-throughput inference
- Datasets & benchmarks: Omni-Fake, SIDA, Deepfake-Eval-2024, TrueFake, PDID, DAID fairness benchmarks

## Git & GitHub Coordination Protocol (copilot-git-organizer)

- **Owner:** `copilot-git-organizer` owns ALL git/GitHub operations (repo setup, commits, branches, pushes, PRs, merges).
- **Rule:** Do not run `git commit`, `git push`, `git branch`, `git merge`, or `git rebase` directly. Request `copilot-git-organizer` to perform git actions.
- **Branch Naming by Role:**
  - `devin-impl`: `impl/<topic>`
  - `agy-research` (Antigravity): `research/<topic>`
  - `copilot-code-review`: `review/<topic>`
- **Workflow:** When a unit of work is ready, message `copilot-git-organizer` with the target branch name and summary of changes to commit, push, and open the PR.
