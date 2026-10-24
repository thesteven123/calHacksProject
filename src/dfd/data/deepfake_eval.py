"""Deepfake-Eval-2024 (TrueMedia.org, arXiv 2503.02857) — EVALUATION ONLY.

Gated on Hugging Face (`nuriachandra/Deepfake-Eval-2024`, CC-BY-SA-4.0 with research-use terms). It is in-the-wild
media flagged on social platforms in 2024: the closest thing we have to deployment distribution, so it must never be
used for training, threshold fitting, or checkpoint selection — only as a final, read-only measurement.
`build_base_datasets` refuses it as a training source.

Repo layout (from the file listing): image-data/, video-data/, audio-data/ plus
`{image,video,audio}-metadata-publish.csv`. Column names are detected at load time because the CSVs are only
readable after access is granted; unknown layouts fail with the columns found, rather than guessing.
"""

from __future__ import annotations

import csv
from pathlib import Path

REPO_ID = "nuriachandra/Deepfake-Eval-2024"
ACCESS_URL = f"https://huggingface.co/datasets/{REPO_ID}"
EVAL_ONLY = True

FILENAME_COLUMNS = ("Filename", "filename", "file_name", "file", "File", "path", "Path")
LABEL_COLUMNS = ("Ground Truth", "ground_truth", "GroundTruth", "label", "Label", "gt", "Ground truth")
FAKE_VALUES = {"fake", "deepfake", "manipulated", "ai-generated", "ai generated", "synthetic", "1"}
REAL_VALUES = {"real", "authentic", "genuine", "0"}


def _pick(columns: list[str], candidates: tuple[str, ...], what: str) -> str:
    for c in candidates:
        if c in columns:
            return c
    raise ValueError(f"Deepfake-Eval-2024 metadata: no {what} column among {candidates}; columns found: {columns}")


def parse_metadata(csv_path: str | Path, data_dir: str | Path) -> tuple[list[tuple[Path, int]], dict]:
    """Return [(media_path, label)] with label 1 = fake, plus stats on skipped rows (unknown labels, missing files)."""
    data_dir = Path(data_dir)
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        columns = list(reader.fieldnames or [])
        fcol, lcol = _pick(columns, FILENAME_COLUMNS, "filename"), _pick(columns, LABEL_COLUMNS, "label")
        items, stats = [], {"rows": 0, "unknown_label": 0, "missing_file": 0, "label_values": {}}
        for row in reader:
            stats["rows"] += 1
            raw = (row.get(lcol) or "").strip()
            stats["label_values"][raw] = stats["label_values"].get(raw, 0) + 1
            value = raw.lower()
            label = 1 if value in FAKE_VALUES else 0 if value in REAL_VALUES else None
            if label is None:
                stats["unknown_label"] += 1
                continue
            name = Path((row.get(fcol) or "").strip()).name
            path = data_dir / name
            if not name or not path.is_file():
                stats["missing_file"] += 1
                continue
            items.append((path, label))
    stats.update(columns=columns, filename_column=fcol, label_column=lcol, usable=len(items),
                 n_fake=sum(y for _, y in items), n_real=sum(1 - y for _, y in items))
    return items, stats


def download(modality: str = "image", token: str | None = None) -> tuple[Path, Path]:
    """Download one modality (read-only, HF cache). Raises with the access URL if the gate is not approved."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError

    if modality not in ("image", "video"):
        raise ValueError("modality must be 'image' or 'video'")
    try:
        root = snapshot_download(REPO_ID, repo_type="dataset", token=token,
                                 allow_patterns=[f"{modality}-metadata-publish.csv", f"{modality}-data/*"])
    except GatedRepoError as e:
        raise SystemExit(f"Deepfake-Eval-2024 access not granted yet. Request it at {ACCESS_URL} "
                         f"(evaluation-only use).") from e
    return Path(root) / f"{modality}-metadata-publish.csv", Path(root) / f"{modality}-data"
