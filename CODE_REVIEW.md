# Code Review

**Scope:** Current implementation under `src/dfd/`, `scripts/`, `configs/`, and `tests/`, reviewed against `docs/implementation-architecture.md`. The project directory has no Git metadata, so this review covers the current source tree rather than a commit diff.

## Findings

### Medium — LOGO holdouts leak into validation, checkpoint selection, and calibration

**Files:** `src/dfd/data/datasets.py:157–160`, `src/dfd/training/trainer.py:122–128`, `scripts/train.py:52–60`  
**Confidence:** High

`holdout_generators` is applied to the HF training dataset but not its validation dataset. The validation data then determines the best checkpoint, and `scripts/train.py` calibrates on a split of that same validation set. Thus the held-out generator is not independent of model selection or calibration, undermining the LOGO generalization estimate. Apply the held-out-generator exclusion consistently to training/validation used for selection and calibration, and reserve held-out-generator data for final evaluation only.

### Medium — Video inference samples only the beginning of long videos

**File:** `src/dfd/inference/video.py:20–30`  
**Confidence:** High

The loop stops after `max_candidates` sampled frames. With the defaults (4 fps, 256 candidates), it examines roughly the first 64 seconds, regardless of the video's total duration. Manipulated content occurring later can be missed and the result can be falsely reassuring. Keep the candidate limit, but distribute samples across the full duration (or explicitly report incomplete temporal coverage).

### Medium — Provenance check reads the entire media file into memory

**File:** `src/dfd/inference/pipeline.py:36–38`  
**Confidence:** High

`Path.read_bytes()` loads the complete image or video before slicing to 4 MiB. Large media therefore causes unnecessary I/O and memory pressure on every inference request. Open the file and read at most 4 MiB.

## Validation

- `.venv/bin/pytest -q`: **16 passed**
- Smoke training could not start in either `.venv` or `.venv-tinker`: both environments report `ModuleNotFoundError: No module named 'numpy'` when running `scripts/train.py`.

---

## Author response (Devin, 2026-10-24)

| # | Finding | Status | Change |
|---|---|---|---|
| 1 | LOGO holdouts leak into selection and calibration | **Fixed, confirmed valid** | Added `split_val_holdout()` (`src/dfd/data/datasets.py`). Validation is split into a disjoint `val_select` (in-distribution fakes + 50% of reals), which drives `best.pt` and `calibration.json`, and `val_test` (holdout fakes + the other 50% of reals), which is reported but never used for any decision. `scripts/train.py`, `Trainer` and the Tinker SFT builder use it. Reals are split rather than shared, so FPR thresholds aren't scored again on the images they were fitted to. Added 2 unit tests. |
| 2 | Video samples only the start of long clips | **Fixed, confirmed valid** | `src/dfd/inference/video.py` now does two passes. The first pass keeps only 32×32 thumbnails and sets the stride from the frame count, so candidates span the whole clip. If the frame count is wrong, it halves the candidates and doubles the stride. The second pass decodes only the chosen keyframes. This also removes the memory spike from holding up to 256 full-resolution frames. Coverage (`duration_s`, `coverage_frac`, `sampled_fps`) is reported in `details`, and a test asserts `coverage_frac > 0.9` on a 30 s clip. |
| 3 | Provenance check reads the whole file | **Fixed, confirmed valid** | `has_provenance_manifest` reads at most 4 MiB. |
| — | "numpy missing in both venvs" | **Could not reproduce** | `.venv/bin/python -c "import numpy"` → 2.5.4 in both envs. The smoke run and two Tiny-GenImage runs completed after the review. The likely cause is running `python scripts/train.py` with a non-venv interpreter. Use `.venv/bin/python` (see `AGENTS.md`). |

**Impact of finding 1:** the earlier S3 Midjourney numbers were optimistic. After a clean rerun, the
unseen-generator TPR@1%FPR is 0.080 (previously reported 0.136), AUC 0.833. The rerun also showed
that in-distribution checkpoint selection picks the worse epoch for the unseen generator. Details
are in `docs/implementation-architecture.md` §11.

**Validation after fixes:** `.venv/bin/pytest -q` → 18 passed.

**Not changed, but noted for a follow-up:** eval/inference takes a native-resolution 224 center crop.
On large photos and 1080p frames, that scores a small central patch (by design, to keep pixel
traces, but it can miss off-center edits). Multi-crop or tiled scoring with top-k pooling would fix it.

---

## Current Code Review and Audit (2026-10-24)

**Scope:** Current source tree under `src/dfd/`, `scripts/`, `configs/`, and `tests/`. The project directory has no Git metadata, so this is a current-state review, not a branch diff. The three findings above were checked and their fixes are present.

### High — Loading an untrusted checkpoint can execute arbitrary code

**File:** `src/dfd/models/detector.py:52–54`  
**Confidence:** High

`DeepfakeDetector.load()` calls `torch.load(..., weights_only=False)`, which uses pickle deserialization. Loading a malicious or tampered `.pt` file can execute code in the process before the model is constructed. This affects the prediction and evaluation entry points that accept checkpoint paths. Use safe weights-only loading for the current tensor-and-primitive checkpoint format, or only load authenticated artifacts and make that trust requirement explicit.

### Medium — Multi-crop inference can return automatic verdicts without matching calibration

**File:** `src/dfd/inference/pipeline.py:72–79, 107–116`  
**Confidence:** High

When multi-crop calibration is absent, the pipeline falls back to center-crop calibration if available, but only converts a `"fake"` verdict to `"abstain"`; a `"real"` verdict can still be returned from thresholds fit on a different score distribution. If no calibration file exists, it silently uses default thresholds and does not mark a mismatch. Require a matching calibration for automatic multi-crop verdicts, or abstain and report the missing/mismatched calibration.

### Medium — Configured generator holdouts are silently ignored if metadata is missing

**File:** `src/dfd/data/datasets.py:73–86, 92–99`  
**Confidence:** High

If an HF dataset lacks the configured generator column, generator filtering is skipped and fake samples receive the generic name `"fake"`. A configuration with `holdout_generators` or `selection_generators` then trains on unfiltered data and may produce an empty final-generator test split without an error. Fail fast when generator-based splits are requested but the column is unavailable.

### Medium — Video multi-crop mode is ignored for temporal checkpoints

**File:** `src/dfd/inference/pipeline.py:136–143`  
**Confidence:** High

With `crop_mode="multi"` and a temporal model, video inference takes the `else` branch and scores center crops, although the result still reports multi-crop mode and uses its calibration. Apply multi-crop scoring to the temporal path or reject the unsupported combination rather than silently using a different scoring method.

**Security note:** `.env` is gitignored and has mode `0600`; secret values were not read during this review.

**Validation:** `.venv/bin/pytest -q` -> 34 passed. Smoke training completed using the project source on `PYTHONPATH`; the direct script launch initially failed to resolve NumPy, while the explicit package-path launch succeeded.

---

## PR #8 Review — Paired Face Data (2026-10-24)

**Scope:** `impl/paired-face-data` (`45168e735019a6b661b6e62a9e76f58b6b79d2be`), compared with its PR base. Review focused on the new paired dataset loaders, their nested split integration, and the provided config/tests.

### High — Nested validation splits separate paired identities across fit, selection, and test

**Files:** `src/dfd/data/datasets.py:242–274`; integration in `src/dfd/data/datasets.py:341–353`  
**Confidence:** High

`DeepFakeFaceDataset` preserves source groups when creating its train/validation split, but the subsequent `split_nested_holdouts()` partitions individual rows: fakes are assigned by generator while real samples are randomly divided into thirds. It never consults `group()`. As a result, a real photo and its generated counterparts can land in different `fit`, `select`, and `test` splits; identities can also appear in more than one of those splits through different generated images. This violates the paired-data isolation goal and can leak correlated examples into checkpoint selection/calibration and final evaluation. Make the nested split group-aware, assigning each source identity wholly to one split and evaluating only the intended generator examples within that group.

### Medium — Explicit DeepFakeFace roots can silently omit configured archives

**File:** `src/dfd/data/paired.py:137–153`  
**Confidence:** High

When a root contains at least one expected zip, the loader selects zip mode and records only the archives that happen to exist. In extracted-folder mode it similarly records only folders that exist. Missing configured generators or `wiki.zip` are therefore silently omitted, potentially allowing a run to proceed without the intended training data. Validate that every requested generator archive/folder and required real-image source exists, and fail with the missing paths.

**Validation:** The PR commit's test suite passes: `.venv/bin/pytest -q` → **62 passed**. The tests cover leakage between the initial DeepFakeFace train/validation partitions, but not group overlap between the later `fit`/`select`/`test` subsets.

---

## PR #9 Review — Cloud Datasets (2026-10-24)

**Scope:** `impl/cloud-datasets` (`0075e9465e59cf3e48f39b1bb2ca15f32785eabb`), against `main`. Reviewed cloud ingestion, external evaluation, format alignment, and reporting.

### Medium — External evaluation emits automatic verdicts with missing or mismatched calibration

**File:** `scripts/eval_external.py:76–78, 87–91`  
**Confidence:** High

The frame and SFHQ reporting paths call `pipe.cal.verdict()` through `_verdict_rates()` without checking `pipe.calibration_status`. Unlike `DetectionPipeline._finalize()`, this path does not abstain when calibration is missing or fitted for a different crop mode. Reports can therefore present automatic real/fake verdict rates based on invalid thresholds. Force abstention unless calibration is matched, and include the calibration status in the verdict computation.

### Medium — Clip-level conformal verdict metrics reuse frame thresholds

**File:** `scripts/eval_external.py:73–78`  
**Confidence:** High

Clip logits are formed by averaging frame logits, then passed to `_verdict_rates()` with the same thresholds calibrated for individual-image scores. The clip-level verdict/FPR/FNR rates therefore do not have the stated calibration guarantee. Either calibrate thresholds on clip-aggregated calibration data or report clip discrimination metrics without presenting those verdict rates as calibrated.

### Medium — Modal external-eval defaults include an unsupported dataset and masks failure

**File:** `scripts/modal_data.py:260–286`; CLI choices in `scripts/eval_external.py:97`  
**Confidence:** High

`action="eval"` defaults to `dataset="all"`, which expands to include `genimage`, but the external evaluator accepts only `wilddeepfake` and `sfhq_t2i`. The GenImage subprocess exits with an argparse error; the Modal wrapper returns its nonzero code, prints it, and continues, so a default run can appear to finish despite a failed dataset evaluation. Require an explicit supported dataset or filter the supported set, and fail the entrypoint on nonzero return codes.

### Medium — Same-size cached files can be mislabeled as the newly requested revision

**File:** `scripts/modal_data.py:182–203`  
**Confidence:** High

Ingestion skips any existing file whose size matches the remote file, without checking its revision or content. It then writes a manifest claiming the newly requested repo revision and selected files. Same-size content changes across revisions can thus leave stale data in the volume while the manifest reports it as current. Validate cached content against the requested revision (or force refresh when revision changes) before updating the manifest.

**Validation:** The latest PR head's full suite passes: `.venv/bin/pytest -q` → **76 passed**.

### PR #9 Re-review — Fixes at `003983d1` (2026-10-24)

**Target:** `impl/cloud-datasets`, head `003983d1cdf61f8a42c37bf30d5d89da7826a1b3`, base `main`.

**Result:** No findings on the four previously reported issues.

- External verdict rates now force abstention unless `calibration_status == "matched"` and report whether verdicts are valid.
- Clip-level discrimination metrics remain, while clip verdict rates are omitted because the conformal thresholds are calibrated on single-image scores.
- Modal external evaluation validates the dataset against supported evaluation datasets, requires a checkpoint, and exits with an error if any evaluation subprocess fails.
- Ingestion now calls `hf_hub_download` for each selected file at the pinned revision, relying on its local-dir metadata revalidation rather than accepting a same-size file without checking.

**Validation:** Full test suite at the reviewed head: `.venv/bin/pytest -q` → **80 passed**.

---

## PR #10 Review — OOD Gating (2026-10-24)

**Scope:** `impl/ood-gating` (`88f6ed6e922c8be2ae3eb256c08ca12ecc60040b`), against `main`. Reviewed the OOD gate implementation, evaluation script, Modal fan-out, tests, and architecture report.

### Low — OOD gate accepts invalid alpha values

**File:** `src/dfd/eval/ood.py:47–58`; input parsing in `scripts/ood_gate_eval.py:95–100`  
**Confidence:** High

`--alphas` accepts arbitrary floats, and `fit_ood_gate()` passes each value directly to `conformal_upper()` without checking the conformal error level. Values outside `(0, 1)` are not valid significance levels; negative values silently produce an infinite threshold (no gating), while values greater than one can select an unintended order statistic or raise an indexing error. Validate each alpha as finite and strictly between zero and one before fitting the gates, with a clear CLI error.

**Validation:** Full suite at the reviewed PR head: `.venv/bin/pytest -q` → **87 passed**.

---

## PR #14 Review — FFHQ and FairFace (2026-10-24)

**Scope:** `impl/face-ffhq` (`4b2e88d11a22b39d690479e8d7a1229225241ac5`), against `main`. Reviewed FFHQ ingestion and tar indexing, paired-face integration, FairFace demographic evaluation, Modal retry configuration, lazy data imports, docs, and tests.

### Medium — Modal retry does not retry a failed training subprocess

**File:** `scripts/modal_logo.py:85–99`  
**Confidence:** High

The new `modal.Retries(max_retries=1, ...)` retries failed Modal function invocations, but `train_rotation()` runs training with `subprocess.run()` and returns a normal result dict even when `proc.returncode != 0`. A training process that exits nonzero (including a child process terminated by a resource/cancellation event) is therefore treated as a successful Modal invocation and is not retried. Check the subprocess return code and raise an exception on failure, or implement an explicit retry around the training attempt, while preserving logs and avoiding unsafe duplicate output handling.

**Validation:** Full suite at the reviewed PR head: `.venv/bin/pytest -q` → **114 passed**.

### PR #14 Re-review — Modal Failure Handling (2026-10-24)

**Target:** `impl/face-ffhq`, head `adb6c74ef1d05bac7ef532b0b55539ff3da3c851d`, against `main`.

**Result:** No findings. The runner now distinguishes signal-terminated subprocesses (negative return code), which raise and trigger the Modal retry, from ordinary positive exit codes, which are recorded as deterministic failures without wasting a retry. In both cases it saves the subprocess log and commits the run volume; the local entrypoint records failures, continues collecting other rotations (including a failed warm-first job), and exits nonzero after reporting them. This addresses both the retry and masked-failure findings. The PR comment explains the intentional no-retry policy for deterministic positive exits.

**Validation:** Full suite at the reviewed PR head: `.venv/bin/pytest -q` → **117 passed**.

### PR #12 Re-review — Alpha validation follow-up (2026-10-24)

**Target:** `impl/ood-alpha-validation`, head `15bdbba2ff28402436d2e6d15d4e0a9994581c6f`, base `main`.

The follow-up fixes the original finding for numeric inputs: `fit_ood_gate()` rejects non-finite values and values outside `(0, 1)`, and the CLI reports those values through `argparse`. One low-severity input-validation gap remains:

#### Low — Nonnumeric alpha entries bypass CLI validation

**File:** `scripts/ood_gate_eval.py:88–90`  
**Confidence:** High

The list comprehension converts entries with `float(a)` before entering the guarded validation. A typo such as `--alphas 0.05,nope` therefore raises an uncaught `ValueError` traceback instead of the intended `argparse` error. Empty entries are also filtered out, so `0.05,,0.10` is silently accepted. Parse each token with an explicit error path and reject empty list elements.

**Validation:** Full suite at the reviewed PR head: `.venv/bin/pytest -q` → **99 passed**. Manually confirmed the nonnumeric case raises `ValueError: could not convert string to float: 'nope'`; the empty-token case proceeds to later config loading, confirming it is silently discarded.

### PR #13 Re-review — Alpha parsing follow-up (2026-10-24)

**Target:** `impl/ood-alpha-parse`, head `f69a110c17be41b8541c5d8e479b56d64b0983dd`, base `main`.

**Result:** No findings. Alpha parsing now rejects nonnumeric entries, empty tokens, and trailing commas through a consistent `argparse` error before loading configuration. Numeric bounds remain validated as finite and strictly within `(0, 1)`, with the same validation enforced in `fit_ood_gate()`.

**Validation:** Full test suite at the reviewed PR head: `.venv/bin/pytest -q` → **110 passed**. Manually confirmed malformed and invalid alpha strings (`0.05,nope`, `0.05,,0.10`, `,`, empty, `0,1`, `nan,0.1`) all exit through `argparse`.

---

## PR #11 Review — Face T2I Training (2026-10-24)

**Scope:** `impl/face-t2i-training` (`80dcf2284eaa3e9293347bcbc0bdfce0a5095a29`), against `main`. Reviewed SFHQ-T2I integration into paired-face training, generator exclusions, format alignment, md5 train/validation split, external evaluation split selection, configs, and tests.

**Result:** No actionable findings. Training correctly excludes both selection and holdout generators through the shared `holdout` tuple, leaving SDXL as the only SFHQ training generator in the provided configurations. The filename-based md5 split is deterministic and applied before per-generator sampling caps; train and validation rows remain disjoint. SFHQ alignment uses the same size and JPEG quality range as DeepFakeFace, and the evaluation CLI supports explicit `all`, `train`, or `val` selection.

**Validation:** Full suite at the reviewed PR head: `.venv/bin/pytest -q` → **87 passed**.

---

## PR #15 Review — FairFace Demographic Fairness (2026-10-24)

**Scope:** `impl/fairness-check` (`38c591d15c109545296ca5cf05eabb19a0f490af`), stacked on `impl/face-ffhq` (PR #14). Reviewed the fairness statistics and reporting, FairFace integration in `eval_external.py`, tests, and `AGENTS.md`.

**Result:** No actionable findings. The group-level FPR/review rates, Wilson intervals, exact one-sided binomial tests with Holm correction, and minimum-sample-size filtering are implemented consistently with the documented output. Unmatched calibration forces abstention and marks fairness verdicts invalid rather than presenting them as meaningful FPR estimates.

**Validation:** Full suite at the reviewed PR head: `.venv/bin/pytest -q` → **122 passed**.

### PR #15 Re-review — Empirical-Bayes and Age Intersections (2026-10-24)

**Target:** `impl/fairness-check`, head `e8dd38fcadb7b3dcc474022ca5bdf2f6c05d2c5e`, stacked on PR #14.

#### Medium — Sparse groups can defeat empirical-Bayes shrinkage for reported gaps

**File:** `src/dfd/eval/fairness.py:61–70, 107–139`  
**Confidence:** High

`disparity_summary()` computes the EB prior from every group in `table` (`eb_shrink(table)`), then filters to `n >= min_n` only when taking the reported gap. Consequently, a very small group with an extreme observed rate can inflate the method-of-moments between-group variance and nearly eliminate shrinkage among the powered groups, even though that sparse group is excluded from the gap. For example, adding a single `k=1, n=1` group to three `n=1000` groups with rates 0.1%, 1%, and 2% changes the powered-group EB gap from about 1.69% to the raw 1.9%. This is particularly relevant for the new three-way intersections, where sparse cells are expected. Estimate the EB prior with a noise-aware policy robust to sparse cells (and apply the same policy in bootstrap replicates), or otherwise prevent underpowered cells from driving the reported shrunk gap; add an imbalanced-group regression test.

**Other follow-up changes:** Derived age3 categories and the 3-way race|gender|age3 table are wired into fairness reporting. The bootstrap CI now targets the EB gap, and the ratio is suppressed for fewer than five minimum-group events.

**Validation:** Full suite at the reviewed PR head: `.venv/bin/pytest -q` → **124 passed**.

---

## Current Source Code Review — Modal Training Failure Reporting (2026-10-24)

**Scope:** Current source snapshot at `45168e7` (working-tree code diff is empty; this is a current-source review, not a PR diff). Focused on core model loading, calibration-aware inference, generator filtering, video handling, and the Modal training runner.

### Medium — Failed Modal training subprocesses do not fail the sweep command

**File:** `scripts/modal_logo.py:81–94, 123–141`  
**Confidence:** High

`train_rotation()` captures a nonzero `train.py` exit code in a result dictionary and returns normally. The local entrypoint later prints the code and log tail but continues and exits successfully, so automation can report a completed sweep even when one or more rotations failed and produced no valid metrics. Propagate child-process failure (or collect all failed rotations and exit nonzero after reporting them), while retaining the captured log output.

**Validation:** `.venv/bin/pytest -q` → **62 passed**.

---

### PR #15 Re-review — Powered-Group EB Prior and B3 Results (2026-10-24)

**Target:** `impl/fairness-check`, head `f91d80132c54dc958e6ca05a4db0a52ef901b5df`, stacked on PR #14.

**Result:** No findings. `disparity_summary()` now estimates the empirical-Bayes prior and its bootstrap interval from powered groups only, matching the groups used for the reported EB gap. The added regression test confirms that an underpowered 100%-rate cell does not affect that gap or its interval. The B3 results and limitations are documented in `docs/implementation-architecture.md` §11.

**Validation:** Full suite at the reviewed PR head: `.venv/bin/pytest -q` → **125 passed**.

---

## PR #16 Review — Video V1 (2026-10-25)

**Scope:** `impl/video-v1` (`46d4ab1c75309e9b485c4d55a5906c7fe43ff3fb`), against `main`. Reviewed TalkingHeadBench/CelebV-HQ ingestion, video frame extraction and tar indexing, clip-level training and splits, temporal model scoring, and WildDeepfake evaluation.

### Medium — Same-size extracted CelebV-HQ clips can remain stale after source revision changes

**File:** `scripts/modal_data.py:198–218`  
**Confidence:** High

`extract_tar_members()` skips an existing extracted clip whenever its size matches the archive member, without checking its content or the source archive revision. If `videos.tar` is updated and a selected clip changes while retaining its size, re-ingestion correctly downloads the new archive but keeps the old extracted clip. The subsequent `video_frames` extraction then trains on stale data while the new ingestion manifest reports the requested revision. Tie extracted clips to the archive revision/content (or re-extract selected members after archive revalidation) before accepting the cache.

**Validation:** Full suite at the reviewed PR head: `.venv/bin/pytest -q` → **143 passed**.

---

## PR #17 Review — Video V1 Results (2026-10-25)

**Scope:** `impl/video-v1-results` (`310574f3769a710d5469857a8f6aa3b992573908`), against `main`. Documentation-only change in `docs/implementation-architecture.md` §11; checked reported select/test and WildDeepfake metrics against the saved run reports.

### Low — WildDeepfake TPR is rounded to zero despite being nonzero

**File:** `docs/implementation-architecture.md` §11  
**Confidence:** High

The reported WildDeepfake clip report has `tpr@1%fpr = 0.002439...` (about **0.24%**), but the table displays `0.00`. This is ordinary rounding if rates are shown as fractions, but reads as a zero result and hides the small nonzero value. Show `0.0024` or `0.24%` consistently with the table’s unit convention.

The other checked values agree with the run reports: the selection AUC/TPR are 0.997/0.91; EmoPortrait test AUC/TPR are 0.990/0.79, with conformal FPR/FNR/abstain 0.6%/4.1%/13%; WildDeepfake has 806 clips, AUC 0.60, real-clip FPR 53.5%, fake FNR 2.7%, and abstention 34%. The section explicitly says **“Do not deploy or fuse this checkpoint.”**

**Validation:** Documentation review only; no tests run.

### PR #17 Re-review — WildDeepfake TPR Display (2026-10-25)

**Target:** `impl/video-v1-results`, head `fa2b81cb7f670010fcf4b925eb4966add28f1423`, against `main`.

**Result:** No findings on the previous issue. The WildDeepfake clip-level `TPR@1%FPR` cell now shows `0.0024`, consistent with the recorded `0.002439...` result. The follow-up changes only that table cell.

**Validation:** Documentation diff checked; no tests run.

### PR #16 Re-review — Parallel Dataset Downloads (2026-10-25)

**Target:** `impl/video-v1`, head `1ec0879a65f4bb6a310e2e6eebb64ba83acbb836`, against `main`.

**Result:** No findings. The download helper now uses an eight-worker thread pool while still calling the Hub download function for every file, preserving per-file revision/etag revalidation and size checks. Volume commits are issued on the coordinating thread every 200 newly downloaded files and after completion. The updated test asserts that every expected file is requested without relying on thread completion order. The earlier CelebV-HQ extraction hash fix remains present.

**Validation:** Full suite at the reviewed PR head: `.venv/bin/pytest -q` → **143 passed**.

### PR #16 Re-review — CelebV-HQ Extraction Cache Validation (2026-10-25)

**Target:** `impl/video-v1`, head `23c08c3dd6b72cf08c493fdf7e4d256b34208305`, against `main`.

**Result:** No findings on the previous issue. `extract_tar_members()` now hashes extracted content and rewrites a cached clip whenever its SHA-256 differs from the current tar member, including same-size stale content. The regression test exercises that case.

**Validation:** Full suite at the reviewed PR head: `.venv/bin/pytest -q` → **143 passed**.

---

## PR #18 Review — WildDeepfake Video V2 (2026-10-25)

**Target:** `impl/video-v2`, head `5aaff2e6c408527d4d2d98b6a749f6a4dab0c042`, against `main`.

**Scope:** WildDeepfake train ingestion and frame extraction, `video_wdf` train/validation split, named evaluation splits, temporal evaluation alignment, and opt-in large-dataset behavior.

### High — Train and test clips can collide and be mixed by clip ID

**Files:** `src/dfd/data/video_frames.py:74–76`; `src/dfd/data/video.py:26–29`  
**Confidence:** High

`wdf_clips()` constructs IDs only from kind, video number, and clip number, while the documented dataset layout says video numbering restarts in the official train and test splits. `VideoClipDataset` then groups every indexed frame by the clip directory in the tar member name, without including the tar/split. When the same `(kind, video, clip)` occurs in both official splits, frames from both tars are combined into one dataset item. This can put official test frames into training/validation clips and contaminate `wdf_test` metrics. I reproduced the collision with same-ID train/test tar members: requesting the test clip returned both frames. Include the split in the generated clip ID (and keep grouping consistent), or otherwise scope index grouping to the intended split; add a regression test with identical video/clip coordinates in train and test.

**Validation:** Full suite at the reviewed PR head: `.venv/bin/pytest -q` → **149 passed**. The suite currently does not cover a cross-split duplicate-ID collision.

---

## Current Codebase Review — Graphify Architecture Pass (2026-10-25)

**Scope:** Current working-tree implementation at `45168e7`, with Graphify refreshed from source (`1039` nodes, `1916` edges). Traced configuration → datasets/splits → training/calibration/evaluation, checkpoint loading → inference, and VLM/core score fusion. This is a source-tree review, not a branch diff.

### Medium — Modal sweep can exit successfully after failed training subprocesses

**File:** `scripts/modal_logo.py:127–143`  
**Confidence:** High

`train_rotation()` returns nonzero child-process status as data, but the local entrypoint only prints that status and the log tail; it never raises or returns a failure after processing the rotations. Automation can therefore treat a sweep with missing/failed runs as successful. Suggested fix: aggregate failed rotation names and exit nonzero after writing all result files and printing their diagnostics. This issue was previously recorded in this file and remains present in the current source.

### Medium — Fusion does not verify that VLM score files use the same checkpoint and preprocessing

**Files:** `scripts/fuse_routes.py:72–78`; metadata written by `scripts/score_vlm_tinker.py:65–68`  
**Confidence:** High

The scorer records `sampler` and `square` in each score file, but fusion checks only that the declared split is either absent or matches the requested fit/select/test name. It can silently fit fusion/calibration on one sampler or square-crop policy and report select/test scores from another, making the resulting metrics invalid while still pairing rows by index. Suggested fix: require non-empty compatible sampler and preprocessing metadata across all supplied score files; validate prediction index uniqueness/range and labels/generators against the configured split before computing core logits.

**Validation:** `.venv/bin/pytest -q` → **62 passed**. `.venv/bin/python scripts/train.py --config configs/smoke.yaml` completed three synthetic-data epochs and wrote its report. Graphify code map refreshed and saved under `graphify-out/`.

---

### PR #18 Re-review — WildDeepfake Split-Collision Fix (2026-10-25)

**Target:** `impl/video-v2`, head `ea86b65587c897983cca18a7c19d2ea721b0bfd4`, against `main`.

**Result:** No findings on the previous issue. `wdf_clips()` now includes the official split in both `clip_id` and `video_group`. `VideoClipDataset` keys frames by `(tar, clip directory)` and resolves each clip to its split-specific frame tar, preventing train/test frames from being merged even when legacy clip IDs match. Two regression tests cover split-restarted WildDeepfake IDs and duplicate legacy IDs across tars.

**Validation:** Full suite at the reviewed PR head: `.venv/bin/pytest -q` → **151 passed**.

---

## Post-Merge Review — PR #19 Serving Security and PR #20 Web Frontend (2026-10-25)

**Targets:** PR #19 `impl/demo-serving` at `d704343631fccda83e2444cb661c0436d33d78f9` (merged as `3910c4c`); PR #20 `impl/demo-web` at `1da8c679ddc7ff87672f5d1d59394c1400525c9f` (merged as `5cd70b2`). Reviewed after merge at the user's request. No Git actions taken.

### PR #19 — Security findings

#### Medium — Upload size limits are checked only after the entire upload is read into memory

**File:** `serve/asgi_apps.py:103–107, 142–146`  
**Confidence:** High

Both detect endpoints parse the multipart form and call `await f.read()` without a bound before `_detect_call()` applies the 15/50 MB size checks. An authenticated client (Bearer key or demo session) can submit an arbitrarily large file and force the ASGI worker to allocate the complete payload before receiving a 413, enabling memory exhaustion. Enforce a request-body limit before multipart parsing and use a bounded read (`limit + 1`) or validated upload size before materializing bytes.

#### Low — Media path containment uses a vulnerable string-prefix check

**File:** `serve/asgi_apps.py:174–177`  
**Confidence:** High

After resolving the requested path, `str(target).startswith(str(MEDIA_ROOT.resolve()))` accepts sibling paths whose names merely begin with `/demo/media`. For example, `../media-private/secret` resolves under `/demo/media-private/` and passes this prefix check if that sibling directory/file exists. This lets an authenticated caller read files outside the intended media directory. Use `target.is_relative_to(MEDIA_ROOT.resolve())` (or an equivalent path-component-aware containment check) and add traversal tests for sibling-prefix paths.

**Security review note:** The security-review subagent reported no vulnerabilities; direct verification of the post-merge source exposed the two issues above.

### PR #20 — Frontend finding

#### Low — Upload object URLs are never revoked

**Files:** `demo/web/src/pages/Playground.tsx:83–99`; `demo/web/src/pages/Instagram.tsx:248–255`  
**Confidence:** High

Both flows create object URLs for uploaded files and retain them in UI state without calling `URL.revokeObjectURL()`. Clearing the chat, failed scans, or repeated Instagram uploads leave the backing image/video blobs pinned for the lifetime of the page. Revoke URLs when their message/post is removed and on component teardown; also revoke immediately for failed scans once no preview uses the URL.

**Validation:** PR #20 frontend production build: `npm ci --no-audit --no-fund && npm run build` in the extracted PR snapshot → passed (`tsc --noEmit` and Vite build). Dependencies were installed only in `/tmp/calhacks-pr20`; no project dependency files changed.

**Follow-up:** At the user's direction, prepared a patch for the highest-severity PR #19 finding (bounded upload reads and regression tests). Patch: `.copilot/session-state/3f5b1aac-d842-4243-9f82-57407b505039/files/pr19-bounded-upload.patch`. The exact extracted PR #19 snapshot passes **153 tests** with the patch applied. The patch is not applied to a Git worktree and no Git actions were taken.

---

## PR #21 Review — Video V3 (2026-10-25)

**Target:** `impl/video-v3`, head `c9869f17858a66130e87b73fe9632607f9743d97`, against `main`. Reviewed Celeb-DF Drive-secret handling and fork-safe dataset handles. No Git actions taken.

**Result:** No actionable findings in the reviewed areas.

- The PR diff contains no added Google Drive URLs or Drive-ID query/file-path literals. The actual ID is loaded from the gitignored local secret file and passed to the Modal function as an ephemeral secret; it is not embedded in source/config or exposed as a function argument.
- The download and size-probe error paths suppress exception text and chaining, so urllib errors cannot echo a URL containing the ID. The entrypoint reports only the archive size and a redacted status; it does not print the secret.
- Video, FFHQ, WildDeepfake, SFHQ-T2I, and DeepFakeFace lazy file handles are cleared during pickling and reopened when the process ID changes. Regression tests cover worker handle reopening and DataLoader reads matching main-process reads.

**Validation:** Full suite at the reviewed head: **168 passed, 2 skipped**. The skips require the optional local YuNet model. The two fork-safety tests passed (with expected Python fork deprecation warnings).

---

## PR #22 Review — Demo Video (2026-10-25)

**Target:** `impl/demo-video`, head `da5c99204032c64aad9a077bb64d322324709d2e`, based on PR #21. Reviewed the new commit against PR #21 and the overall PR changes against the prior serving/frontend findings. No Git actions taken.

**Previously reported findings addressed:**

- Uploads now reject known-oversize files before reading and cap unknown-size reads at `limit + 1`; both upload endpoints use this bounded helper.
- Media route containment now uses resolved-path `is_relative_to`, preventing sibling-prefix traversal.
- Failed scans now revoke preview URLs. Successful uploads still lack cleanup when their message/post is removed or the component unmounts; see finding below.

### Low — Successful-upload object URLs still are not released

**Files:** `demo/web/src/pages/Playground.tsx:83–101, 118`; `demo/web/src/pages/Instagram.tsx:248–256`
**Confidence:** High

The new `URL.revokeObjectURL()` calls run only when scanning returns no detection. Successful uploads retain their object URLs indefinitely: Playground's “New chat” clears messages without revoking them, and Instagram has no release path for uploaded posts when the component unmounts. Repeated successful uploads therefore continue pinning their media blobs for the lifetime of the page. Track owned URLs and revoke them when the corresponding content is removed and on component teardown.

### Low — Video result reports calibration even when calibration is missing or mismatched

**File:** `src/dfd/inference/router.py:271–277`; displayed by `demo/web/src/components/ExplanationSheet.tsx:69–74`
**Confidence:** High

`VideoClipScorer` returns a specialist row with `calibration_status` of `missing` or `mismatched` and forces its verdict to `review`, but `detect_video()` sets `video.calibrated = True` for every non-`None` row. The explanation sheet consequently says “calibrated: yes” for an uncalibrated score. Derive this field from `vrow["calibration_status"] == "matched"` (and cover missing/mismatched cases in a regression test).

**Validation:** Full suite at the exact PR head: **175 passed, 2 skipped**. Frontend `npm run build` passed.

---

## PR #21 and PR #22 Re-review (2026-10-25)

**PR #21 target:** `impl/video-v3`, head `c9869f17858a66130e87b73fe9632607f9743d97`. No changes since the prior review; no new findings in the requested Celeb-DF secret-handling or fork-safety scope.

**PR #22 target:** `impl/demo-video`, head `af1fcd6e7c2f9a9ee592070478c73fa54a5e291a`, based on PR #21. Reviewed the follow-up commit against the prior PR #22 review. No Git actions taken.

### Medium — Forensic explanation sheet states unsupported findings and always labels media as suspected

**File:** `demo/web/src/components/ExplanationSheet.tsx:43–58`  
**Confidence:** High

The sheet no longer renders the API-provided verdict or `detection.reasons`; it hardcodes “AI or synthetic media suspected” and asserts facial boundary warping and lighting inconsistencies for every item, optionally attaching a face-model probability as though it were evidence of those artifacts. The API does not produce these specific visual findings, and items with `real` or `review` verdicts receive the same fake-positive headline. It also claims no C2PA manifest was found although this application does not check manifests. Render only supported API findings and use the actual verdict; otherwise label these signals as unavailable/not analyzed.

### Medium — Review verdicts can be labeled “Altered with AI”

**File:** `demo/web/src/components/FlagChip.tsx:8–13`  
**Confidence:** High

For any `review` verdict with confidence at least 0.5, `badgeFor()` returns “Altered with AI”. A review verdict means the detector could not decide, so this UI converts uncertainty into an affirmative manipulation label. Keep review/uncertain results neutral rather than applying the fake-score threshold.

### Low — Upload preview object URLs still are not released

**Files:** `demo/web/src/pages/Playground.tsx:83–101, 118`; `demo/web/src/pages/Instagram.tsx:248–256`  
**Confidence:** High

This commit removes the failed-scan `URL.revokeObjectURL()` calls added in the previous head. Successful and failed upload URLs are now retained after scans; clearing the Playground chat or leaving the Instagram view also does not revoke successful-upload URLs. Revoke each owned URL when its content is removed and on component teardown.

### Low — Video API marks uncalibrated scores as calibrated

**File:** `src/dfd/inference/router.py:271–277`  
**Confidence:** High

`detect_video()` still sets `video.calibrated = True` whenever a scorer returns a row, even when `calibration_status` is `missing` or `mismatched` and the specialist verdict is forced to `review`. The latest frontend no longer displays this field, but API clients receive misleading calibration metadata. Set it only when the video specialist calibration status is `matched`.

**Validation:** PR #22 full suite at `af1fcd6e7c2f9a9ee592070478c73fa54a5e291a`: **175 passed, 2 skipped**. Frontend `npm run build` passed.

---

## PR #21 and PR #22 Re-review — V3 Serving Update (2026-10-25)

**PR #21:** `impl/video-v3`, head `c9869f17858a66130e87b73fe9632607f9743d97`, unchanged; no new findings in the reviewed Celeb-DF secret-handling or fork-safety scope.

**PR #22:** `impl/demo-video`, head `8cf28ebb4365e2781ad19658db4e409b0ff4e155`, based on PR #21. Reviewed the added frontend follow-ups, V3 serving-path change, and architecture results entry. No Git actions taken.

**Previous PR #22 findings:** The explanation-sheet unsupported claims and fake-positive copy, and the “Altered with AI” label for `review` verdicts, are fixed. The remaining findings are:

### Medium — Multipart uploads are fully parsed before their size is checked

**File:** `serve/asgi_apps.py:132–140, 175–183`
**Confidence:** High

The routes call `await request.form()` before `_read_upload()` checks the upload size or performs its bounded `limit + 1` read. Starlette's multipart parser consumes the entire file part first, keeping small parts in memory and spooling larger parts to a temporary file; its default `max_part_size` limit applies to non-file fields, not uploaded files. An authenticated client can therefore still send an arbitrarily large file and exhaust temporary storage/network resources before the 413 is returned. Enforce a request-body/file-size cap while parsing or at the server/proxy boundary, not only after form parsing.

### Medium — The V3 specialist can be silently disabled while the API advertises it as active

**File:** `serve/modal_app.py:56, 91–97`; metadata in `serve/asgi_apps.py:36–48`
**Confidence:** High

The V3 scorer is instantiated only if `/runs/video_v3/config/best.pt` exists; otherwise the app starts with `video_scorer=None` and silently uses the uncalibrated keyframe heuristic. `/api/models` still unconditionally advertises `video_v3` and its metrics. Fail startup when the configured V3 checkpoint is unavailable, or expose its unavailable status and avoid claiming the model is active.

### Low — Video calibration metadata remains incorrect

**File:** `src/dfd/inference/router.py:271–277`
**Confidence:** High

`detect_video()` still sets `video.calibrated = True` whenever the scorer returns a row, even when its `calibration_status` is `missing` or `mismatched` and its verdict is forced to `review`. The restored explanation sheet displays this field, so users can see “calibrated: yes” for an uncalibrated result. Set it only when the specialist status is `matched`.

### Low — API model notes still describe the V3 video result as only a heuristic

**File:** `serve/asgi_apps.py:44–48`
**Confidence:** High

The global model metadata says “Video verdicts are a per-keyframe heuristic, not clip-calibrated,” although the served V3 route combines a clip-level model with keyframe results and reports a clip-model method when scoring succeeds. Update the note to describe both components and clarify the calibration status.

### Low — Uploaded preview object URLs are not revoked

**Files:** `demo/web/src/pages/Playground.tsx:83–101, 117`; `demo/web/src/pages/Instagram.tsx:248–256`
**Confidence:** High

No `URL.revokeObjectURL()` calls remain. Successful and failed uploads retain their blob URLs after scanning; clearing the Playground chat and leaving the Instagram view do not release URLs owned by those components. Revoke each URL when its content is removed and on component teardown.

**Validation:** Exact PR #22 head: Python suite **175 passed, 2 skipped**; frontend Vitest **12 passed**; `npm run build` passed. GitHub reported no PR checks.

---

## PR #23 Review — Attention-MIL Pooling Ablation (2026-10-25)

**Target:** `impl/image-ablation`, head `c895957be3f81b50437c8d632104fc4daa110c34`, based on PR #21 (`impl/video-v3`). Reviewed model pooling/checkpoint compatibility and the paired ablation analysis. No Git actions taken.

**Checkpoint compatibility:** The default `patch_pool="mean"` follows the existing patch-token mean path without adding model parameters or changing saved state keys. Old configs omit the new field and `from_dict()` supplies the `"mean"` default. I synthesized a legacy-format checkpoint with the field and any patch-pool tensors omitted; `DeepfakeDetector.load()` loaded it and reproduced logits bit-for-bit (max absolute difference 0). Attention-MIL is opt-in, starts with uniform attention due to zero-initialized output weights, and its trainable parameters are included in optimizer groups and saved checkpoints.

### Medium — Missing FPR measurements can satisfy the pre-registered success criterion

**File:** `scripts/modal_ablation.py:145–154`
**Confidence:** High

`decide()` excludes `None` FPR values from the `over` set, then treats an empty set as `c3=True`. If every rotation's FPR is missing, the function can return `SUCCESS` when the AUC criteria pass. Require a finite FPR for each expected rotation; missing or invalid measurements should produce an invalid/inconclusive decision, not satisfy the gate.

### Low — Paired bootstrap verifies labels but not paired sample identity

**File:** `scripts/modal_ablation.py:96–113`
**Confidence:** High

The paired bootstrap only checks that baseline and ablation label arrays match. It ignores the saved `generators` arrays and has no sample IDs, so reordered or different examples with the same label sequence are silently treated as paired observations. That invalidates the paired confidence intervals and can affect the registered decision. Persist stable test-example identifiers (and compare them, plus labels/generators) before resampling.

**Validation:** Full suite at the reviewed head: **172 passed, 2 skipped**. Legacy checkpoint compatibility and exact default-mean output parity were verified directly. A targeted edge-case check confirmed `decide()` currently returns `SUCCESS` with missing FPR values, and `paired_bootstrap()` accepts same-label arrays with different generator assignments.

---

## Gradio/Vercel Demo Review — commits `da10af5` and `d876e34` (2026-10-25)

**Target:** `demo/gradio` only, `8cf28eb..d876e34` in `calHacksProject-demo2`. Security and UX review focused on credential handling, hosting limits, unset-backend behavior, and verdict honesty. No Git actions taken.

**Security result:** No High or Medium findings. `DFD_API_KEY` is read and used only by the Python backend to construct the outbound authorization header; there is no browser-side key reference or logging. `demo/gradio/.env.local` and `.vercel` are ignored and untracked, and neither is present in the reviewed diff. With `DFD_API_URL` unset, the UI explicitly says “Demo mode: mock detector (no model configured)”; a local check verified mock calibration labels and the research disclaimer. The UI renders the detector's label, verdict, reasons, specialist scores, and disclaimer without inventing explanations.

Vercel's approximately 4.5 MB body limit is clearly disclosed in the UI and README, which also directs users needing larger uploads to another host. The configured 120-second function limit accommodates the documented ~24-second Modal cold start under normal inference latency.

### Low — Remote timeout equals the Vercel function limit

**Files:** `demo/gradio/backend.py:74`; `demo/gradio/vercel.json:3`
**Confidence:** Medium

The outbound HTTP timeout is 120 seconds, the same as the Vercel function's maximum duration. If a cold or slow inference reaches that boundary, Vercel may terminate the function before the client can convert the timeout into the UI's handled “Scan failed” response. Give the HTTP request a shorter deadline than the function budget, or increase the function budget with appropriate deployment-plan support.

**Validation:** Python syntax compilation passed; a local unset-URL check confirmed the explicit mock banner, valid verdict, mock specialist status, and research disclaimer. Secret files were not read or printed.
