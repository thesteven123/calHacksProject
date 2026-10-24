"""Tinker SupervisedDataset that streams PTE-augmented real/fake images as VLM chat examples.

Images go through the same Platform Transcoding Emulation as the core detector, so the VLM route
is also trained "through the distribution channel".
"""

from __future__ import annotations

import math
import random

import chz
import numpy as np
import tinker
from tinker_cookbook.image_processing_utils import get_image_processor, resize_image
from tinker_cookbook.renderers import ImagePart, Message, TextPart, TrainOnWhat, get_renderer
from tinker_cookbook.supervised.common import datum_from_model_input_weights
from tinker_cookbook.supervised.types import SupervisedDataset, SupervisedDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer

from dfd.config import Config
from dfd.data.datasets import build_base_datasets, split_nested_holdouts, split_val_holdout
from dfd.data.pte import PlatformTranscodingEmulator
from dfd.tinker_route import ANSWERS, DEFAULT_SQUARE_SIZE, PROMPT, square_crop


def build_renderer(model_name: str, renderer_name: str):
    return get_renderer(
        name=renderer_name, tokenizer=get_tokenizer(model_name), image_processor=get_image_processor(model_name)
    )


def prompt_messages(image) -> list[Message]:
    return [Message(role="user", content=[ImagePart(type="image", image=image), TextPart(type="text", text=PROMPT)])]


class DeepfakeSFTDataset(SupervisedDataset):
    def __init__(self, base, renderer, pte: PlatformTranscodingEmulator, batch_size: int,
                 max_length: int, max_image_size: int, augment: bool, seed: int = 0,
                 square_size: int = DEFAULT_SQUARE_SIZE):
        self.base, self.renderer, self.pte = base, renderer, pte
        self.batch_size, self.max_length, self.max_image_size = batch_size, max_length, max_image_size
        self.square_size = square_size
        self.augment, self.seed, self.epoch = augment, seed, 0
        self.order = list(range(len(base)))
        self.set_epoch(0)

    def set_epoch(self, seed: int = 0) -> None:
        self.epoch = seed
        self.order = list(range(len(self.base)))
        random.Random(self.seed + seed).shuffle(self.order)

    def __len__(self) -> int:
        return math.ceil(len(self.order) / self.batch_size)

    def _datum(self, idx: int) -> tinker.Datum:
        img, label, _ = self.base[idx]
        rng = np.random.default_rng([self.seed, self.epoch, idx])
        if self.augment:
            img, _ = self.pte(img, rng)
        if self.square_size:
            # Random square window in training (centre at eval/scoring) so neither aspect ratio nor native
            # size reaches the VLM; see DEFAULT_SQUARE_SIZE.
            w, h = img.size
            s = min(w, h)
            if self.augment and w != h:
                left, top = int(rng.integers(0, w - s + 1)), int(rng.integers(0, h - s + 1))
                img = img.crop((left, top, left + s, top + s))
            img = square_crop(img, self.square_size)
        else:
            img = resize_image(image=img, max_size=self.max_image_size)
        messages = prompt_messages(img) + [Message(role="assistant", content=[TextPart(type="text", text=ANSWERS[label])])]
        model_input, weights = self.renderer.build_supervised_example(
            messages=messages, train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE
        )
        return datum_from_model_input_weights(model_input, weights, max_length=self.max_length)

    def get_batch(self, index: int) -> list[tinker.Datum]:
        return [self._datum(i) for i in self.order[index * self.batch_size : (index + 1) * self.batch_size]]


@chz.chz
class DeepfakeSFTBuilder(SupervisedDatasetBuilder):
    dfd_config: str
    model_name_for_tokenizer: str
    renderer_name: str
    batch_size: int = 32
    max_length: int = 8192
    max_image_size: int = 480
    square_size: int = DEFAULT_SQUARE_SIZE
    run_nll_evaluator: bool = True

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        cfg = Config.from_yaml(self.dfd_config)
        train_base, val_full = build_base_datasets(cfg.data, cfg.train.seed)
        if cfg.data.selection_generators:
            val_base, _, _ = split_nested_holdouts(
                val_full, tuple(cfg.data.selection_generators), tuple(cfg.data.holdout_generators), cfg.train.seed
            )
        else:
            val_base, _ = split_val_holdout(val_full, tuple(cfg.data.holdout_generators), seed=cfg.train.seed)
        renderer = build_renderer(self.model_name_for_tokenizer, self.renderer_name)
        pte = PlatformTranscodingEmulator(cfg.data.pte)
        common = dict(renderer=renderer, pte=pte, batch_size=self.batch_size, max_length=self.max_length,
                      max_image_size=self.max_image_size, square_size=self.square_size)
        train = DeepfakeSFTDataset(train_base, augment=True, seed=cfg.train.seed, **common)
        val = DeepfakeSFTDataset(val_base, augment=True, seed=cfg.train.seed + 1, **common) if self.run_nll_evaluator else None
        return train, val
