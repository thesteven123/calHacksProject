"""Deepfake-Eval-2024 loader tests (offline; no gated download)."""

import pytest
from PIL import Image

from dfd.config import DataConfig
from dfd.data.datasets import build_base_datasets
from dfd.data.deepfake_eval import parse_metadata


def _write(tmp_path, header, rows):
    data = tmp_path / "image-data"
    data.mkdir()
    for r in rows:
        Image.new("RGB", (8, 8)).save(data / r[0])
    csv = tmp_path / "image-metadata-publish.csv"
    csv.write_text(header + "\n" + "\n".join(",".join(r) for r in rows) + "\n")
    return csv, data


def test_parse_detects_columns_maps_labels_and_skips_unknown(tmp_path):
    csv, data = _write(tmp_path, "Filename,Ground Truth,Notes",
                       [("a.jpg", "Fake", "x"), ("b.png", "Real", "y"), ("c.jpg", "Unknown", "z")])
    (data / "c.jpg").unlink()  # also exercise missing-file accounting irrelevant to unknown labels
    items, stats = parse_metadata(csv, data)
    assert [(p.name, y) for p, y in items] == [("a.jpg", 1), ("b.png", 0)]
    assert stats["unknown_label"] == 1 and stats["label_column"] == "Ground Truth" and stats["usable"] == 2


def test_parse_fails_loudly_on_unknown_layout(tmp_path):
    csv, data = _write(tmp_path, "name,truthiness", [("a.jpg", "Fake")])
    with pytest.raises(ValueError, match="columns found"):
        parse_metadata(csv, data)


def test_training_refuses_deepfake_eval_2024():
    with pytest.raises(ValueError, match="evaluation-only"):
        build_base_datasets(DataConfig(source="deepfake_eval_2024"))
    with pytest.raises(ValueError, match="evaluation-only"):
        build_base_datasets(DataConfig(source="hf", hf_dataset="nuriachandra/Deepfake-Eval-2024"))
