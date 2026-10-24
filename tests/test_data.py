import numpy as np
from PIL import Image

from dfd.config import Config, PTEConfig
from dfd.data.datasets import (
    SyntheticDeepfakeDataset, TransformedDataset, build_base_datasets,
    split_nested_holdouts, split_val_holdout,
)
from dfd.data.pte import SEVERITY_PRESETS, PlatformTranscodingEmulator
from dfd.data.transforms import DetectorTransform


def _img(size=(300, 200)):
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, (*size[::-1], 3), dtype=np.uint8))


def test_pte_heavy_applies_three_hops_and_marks_severity():
    img, meta = PlatformTranscodingEmulator(SEVERITY_PRESETS["heavy"])(_img(), np.random.default_rng(1))
    assert meta.hops == 3 and meta.severity == "heavy" and img.mode == "RGB"


def test_pte_disabled_is_identity():
    src = _img()
    img, meta = PlatformTranscodingEmulator(PTEConfig(enabled=False))(src, np.random.default_rng(0))
    assert meta.severity == "clean" and np.array_equal(np.asarray(img), np.asarray(src))


def test_beauty_preset_sets_flag():
    _, meta = PlatformTranscodingEmulator(SEVERITY_PRESETS["beauty"])(_img(), np.random.default_rng(0))
    assert meta.beauty


def test_transform_upsizes_small_images_and_outputs_unit_range():
    x, _ = DetectorTransform(PTEConfig(), 224, train=True)(_img((128, 128)), np.random.default_rng(0))
    assert x.shape == (3, 224, 224) and 0.0 <= x.min() and x.max() <= 1.0


def test_eval_transform_is_deterministic():
    ds = TransformedDataset(SyntheticDeepfakeDataset(4), DetectorTransform(PTEConfig(), 224, train=False))
    assert (ds[1]["pixels"] == ds[1]["pixels"]).all()


def test_split_val_holdout_is_disjoint_and_test_has_both_classes():
    base = SyntheticDeepfakeDataset(64)
    sel, test = split_val_holdout(base, ("synth_gan",))
    sel_meta = [sel.meta(i) for i in range(len(sel))]
    test_meta = [test.meta(i) for i in range(len(test))]
    assert not set(sel.indices) & set(test.indices)
    assert len(sel) + len(test) == len(base)
    assert all(g != "synth_gan" for _, g in sel_meta)
    assert {y for y, _ in test_meta} == {0, 1} and all(g in ("synth_gan", "real") for _, g in test_meta)


def test_split_val_without_holdout_keeps_everything_for_selection():
    sel, test = split_val_holdout(SyntheticDeepfakeDataset(16), ())
    assert len(sel) == 16 and len(test) == 0


def test_nested_holdouts_keep_three_generators_and_reals_disjoint():
    from dfd.config import DataConfig

    cfg = DataConfig(
        source="synthetic", synthetic_train_size=64, synthetic_val_size=96,
        selection_generators=("synth_gan",), holdout_generators=("synth_diff",),
    )
    train, _ = build_base_datasets(cfg)
    assert all(train.meta(i)[1] == "real" for i in range(len(train)))

    class ThreeFakes:
        def __len__(self):
            return 90

        def meta(self, i):
            return (0, "real") if i % 3 == 0 else (1, ("base", "selector", "final")[(i // 3) % 3])

    fit, select, test = split_nested_holdouts(ThreeFakes(), ("selector",), ("final",), seed=7)
    assert set(fit.indices) | set(select.indices) | set(test.indices) == set(range(90))
    assert not set(fit.indices) & set(select.indices) and not set(fit.indices) & set(test.indices)
    assert not set(select.indices) & set(test.indices)
    for ds, allowed_fake in ((fit, "base"), (select, "selector"), (test, "final")):
        labels = [ds.meta(i) for i in range(len(ds))]
        assert {label for label, _ in labels} == {0, 1}
        assert {g for label, g in labels if label == 1} == {allowed_fake}


def test_nested_holdouts_reject_overlap_and_missing_generator():
    from pytest import raises

    data = SyntheticDeepfakeDataset(64)
    with raises(ValueError, match="distinct"):
        split_nested_holdouts(data, ("synth_gan",), ("synth_gan",))
    with raises(ValueError, match="fake samples"):
        split_nested_holdouts(data, ("missing",), ("synth_gan",))


def test_config_yaml_roundtrip(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("model:\n  feature_layers: [2, 4]\ndata:\n  pte:\n    max_hops: 2\n")
    cfg = Config.from_yaml(p)
    assert cfg.model.feature_layers == (2, 4) and cfg.data.pte.max_hops == 2


def test_holdout_matching_no_validation_fakes_fails_fast():
    from pytest import raises

    with raises(ValueError, match="no fakes"):
        split_val_holdout(SyntheticDeepfakeDataset(32), ("generator_that_does_not_exist",))


def test_nested_holdout_error_reports_empty_split():
    from pytest import raises

    with raises(ValueError, match="select=0"):
        split_nested_holdouts(SyntheticDeepfakeDataset(64), ("missing",), ("synth_gan",))


def test_hf_dataset_without_generator_column_fails_fast_when_filtering(monkeypatch):
    """CODE_REVIEW 2026-10-24: generator holdouts must not be silently skipped when metadata is absent."""
    import datasets
    from pytest import raises

    from dfd.config import DataConfig
    from dfd.data.datasets import HFImageDataset

    imgs, labels = [_img((32, 32))] * 4, [0, 1, 0, 1]
    no_gen = datasets.Dataset.from_dict({"image": imgs, "label": labels})
    with_gen = datasets.Dataset.from_dict({"image": imgs, "label": labels, "generator": ["Real", "ADM", "Real", "SD15"]})
    cfg = DataConfig(source="hf", generator_column="generator", holdout_generators=("Midjourney",))

    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: no_gen)
    with raises(ValueError, match="column 'generator' is missing"):
        HFImageDataset(cfg, "train", exclude_generators=("Midjourney",))
    with raises(ValueError, match="missing"):
        build_base_datasets(cfg)  # fails on the train split before training could leak the holdout
    assert len(HFImageDataset(cfg, "train")) == 4  # no filtering requested -> still usable

    # Train has generator metadata but validation does not: the holdout test split cannot be formed.
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, split=None, **k: with_gen if split == "train" else no_gen)
    with raises(ValueError, match="need column"):
        build_base_datasets(cfg)
