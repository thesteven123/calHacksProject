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
