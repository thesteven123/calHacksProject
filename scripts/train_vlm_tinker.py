"""Fine-tune a Qwen-VL model on Tinker as the Stage 3 real/fake VLM route.

Requires: `uv pip install -e ".[tinker]"` and TINKER_API_KEY in the environment.

    python scripts/train_vlm_tinker.py dfd_config=configs/tiny_genimage_local.yaml log_path=runs/vlm_tinker
    python scripts/train_vlm_tinker.py dfd_config=... model_name=Qwen/Qwen3.5-9B renderer_name=qwen3_5_disable_thinking
"""

from __future__ import annotations

import asyncio
import os

import chz
from tinker_cookbook import cli_utils
from tinker_cookbook.supervised import train

from dfd.env import load_dotenv
from dfd.tinker_route import DEFAULT_MODEL, DEFAULT_RENDERER, DEFAULT_SQUARE_SIZE
from dfd.tinker_route.sft_data import DeepfakeSFTBuilder

load_dotenv()


@chz.chz
class VLMRouteConfig:
    dfd_config: str
    log_path: str = "runs/vlm_tinker"
    model_name: str = DEFAULT_MODEL
    renderer_name: str = DEFAULT_RENDERER
    learning_rate: float = 5e-4
    lr_schedule: str = "cosine"
    num_epochs: int = 1
    lora_rank: int = 32
    batch_size: int = 32
    max_image_size: int = 480
    square_size: int = DEFAULT_SQUARE_SIZE  # 0 = native aspect (leaks size/aspect shortcut on GenImage)
    save_every: int = 20
    eval_every: int = 20
    max_steps: int | None = None
    wandb_project: str | None = None


def main(cfg: VLMRouteConfig) -> None:
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("TINKER_API_KEY is not set")
    cli_utils.check_log_dir(cfg.log_path, behavior_if_exists="ask")
    builder = DeepfakeSFTBuilder(
        dfd_config=cfg.dfd_config,
        model_name_for_tokenizer=cfg.model_name,
        renderer_name=cfg.renderer_name,
        batch_size=cfg.batch_size,
        max_image_size=cfg.max_image_size,
        square_size=cfg.square_size,
    )
    config = train.Config(
        log_path=cfg.log_path,
        model_name=cfg.model_name,
        recipe_name="dfd_vlm_route",
        renderer_name=cfg.renderer_name,
        dataset_builder=builder,
        learning_rate=cfg.learning_rate,
        lr_schedule=cfg.lr_schedule,
        num_epochs=cfg.num_epochs,
        lora_rank=cfg.lora_rank,
        save_every=cfg.save_every,
        eval_every=cfg.eval_every,
        max_steps=cfg.max_steps,
        wandb_project=cfg.wandb_project,
    )
    asyncio.run(train.main(config))


if __name__ == "__main__":
    main(chz.entrypoint(VLMRouteConfig))
