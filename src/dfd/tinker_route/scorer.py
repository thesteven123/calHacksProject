"""Score images with a Tinker-hosted VLM: logit = log p("fake") - log p("real") at the answer position."""

from __future__ import annotations

from concurrent.futures import Future

import tinker
from PIL import Image
from tinker_cookbook.image_processing_utils import resize_image

from dfd.tinker_route import ANSWERS, DEFAULT_MODEL, DEFAULT_RENDERER, DEFAULT_SQUARE_SIZE, square_crop
from dfd.tinker_route.sft_data import build_renderer, prompt_messages


class VLMScorer:
    """`square_size` must match how the checkpoint was trained (0 for native-aspect checkpoints)."""

    def __init__(self, model_path: str | None = None, model_name: str = DEFAULT_MODEL,
                 renderer_name: str = DEFAULT_RENDERER, max_image_size: int = 480,
                 square_size: int = DEFAULT_SQUARE_SIZE):
        service = tinker.ServiceClient()
        self.client = (
            service.create_sampling_client(model_path=model_path)
            if model_path else service.create_sampling_client(base_model=model_name)
        )
        self.renderer = build_renderer(model_name, renderer_name)
        tok = self.renderer.tokenizer
        self.answer_tokens = {a: tok.encode(a, add_special_tokens=False) for a in ANSWERS}
        self.max_image_size, self.square_size = max_image_size, square_size

    def _submit(self, img: Image.Image) -> dict[str, Future]:
        img = img.convert("RGB")
        img = square_crop(img, self.square_size) if self.square_size else resize_image(image=img, max_size=self.max_image_size)
        prompt = self.renderer.build_generation_prompt(prompt_messages(img))
        return {
            a: self.client.compute_logprobs(
                tinker.ModelInput(chunks=[*prompt.chunks, tinker.types.EncodedTextChunk(tokens=toks)])
            )
            for a, toks in self.answer_tokens.items()
        }

    def score_many(self, images: list[Image.Image]) -> list[float]:
        """Submits all requests before awaiting any, so Tinker can batch them server-side."""
        pending = [self._submit(img) for img in images]
        scores = []
        for futures in pending:
            lp = {a: sum(f.result()[-len(self.answer_tokens[a]):]) for a, f in futures.items()}
            scores.append(lp["fake"] - lp["real"])
        return scores

    def score(self, img: Image.Image) -> float:
        return self.score_many([img])[0]
