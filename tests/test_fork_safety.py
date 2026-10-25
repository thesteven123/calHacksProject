"""Lazy file handles must not be shared with forked DataLoader workers (shared fd seek positions race)."""

import multiprocessing as mp

import numpy as np
import pytest
from torch.utils.data import DataLoader

from dfd.data.external import FFHQDataset, SFHQT2IDataset, WildDeepfakeDataset
from dfd.data.paired import DeepFakeFaceDataset
from dfd.data.video import VideoClipDataset
from dfd.data.wilddeepfake_index import build_wilddeepfake_index, write_wilddeepfake_index
from test_external_data import _sfhq_zip, _wilddeepfake_root
from test_ffhq_fairface import _ffhq_root
from test_paired_data import _create_synthetic_dff
from test_video import _frame_root, _thb_clips

FORK = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else None
pytestmark = pytest.mark.skipif(FORK is None, reason="fork start method unavailable")


def _child(ds, getter, parent_id, q):
    q.put(id(getter(ds)) != parent_id)


def _child_gets_new_handle(ds, getter) -> bool:
    ds[0]  # open the handle in the parent, as a main-process read before DataLoader workers start would
    parent = getter(ds)
    q = FORK.Queue()
    p = FORK.Process(target=_child, args=(ds, getter, id(parent), q))
    p.start()
    p.join(30)
    return q.get(timeout=5)


def _datasets(tmp_path):
    clips = _thb_clips()
    video = VideoClipDataset(_frame_root(tmp_path / "frames", clips, n_frames=3), clips)
    wild_root = _wilddeepfake_root(tmp_path)
    write_wilddeepfake_index(wild_root, build_wilddeepfake_index(wild_root, split="test"), split="test")
    dff = DeepFakeFaceDataset(root=_create_synthetic_dff(tmp_path / "dff", num_identities=4))
    return {
        "video": (video, lambda d: d._handle(d.frames[0][0]["tar"])),
        "ffhq": (FFHQDataset(_ffhq_root(tmp_path / "ffhq")), lambda d: d._handle(d.items[0]["tar"])),
        "wild": (WildDeepfakeDataset(wild_root), lambda d: d._handle(d.items[0]["tar"])),
        "sfhq": (SFHQT2IDataset(_sfhq_zip(tmp_path)), lambda d: d._z()),
        "dff": (dff, lambda d: d._get_zip(d.items[0][0])),
    }


def test_forked_child_reopens_handles(tmp_path):
    for name, (ds, getter) in _datasets(tmp_path).items():
        assert _child_gets_new_handle(ds, getter), name


def _collate(batch):
    return [np.asarray(frames[0]) for frames, _, _ in batch]


def test_forked_dataloader_reads_match_main_process(tmp_path):
    clips = _thb_clips()
    ds = VideoClipDataset(_frame_root(tmp_path, clips, n_frames=4), clips)
    expected = [np.asarray(ds[i][0][0]) for i in range(len(ds))]  # main-process reads open the handles first
    loader = DataLoader(ds, batch_size=3, num_workers=3, multiprocessing_context=FORK, collate_fn=_collate)
    got = [a for batch in loader for a in batch]
    assert len(got) == len(expected) and all(np.array_equal(a, b) for a, b in zip(got, expected))
