"""Score a Tinker-trained VLM route on a held-out split and report AUC / TPR@FPR (billable, but cheap).

    .venv-tinker/bin/python scripts/score_vlm_tinker.py --sampler tinker://.../sampler_weights/final \
        --config configs/tiny_genimage_nested.yaml --split fit --per-class 60

Images get the config's deterministic eval-time PTE, matching the core detector's validation, so the two
routes see identically degraded inputs. The `test` split is refused unless --allow-test is passed, to keep
the final generator untouched until a checkpoint is chosen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dfd.config import Config
from dfd.data.datasets import build_base_datasets, eval_splits
from dfd.env import load_dotenv
from dfd.eval.paired import balanced_indices, paired_pte_images
from dfd.eval.metrics import sliced_report
from dfd.tinker_route import DEFAULT_MODEL, DEFAULT_RENDERER, DEFAULT_SQUARE_SIZE


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sampler", required=True, help="tinker://.../sampler_weights/<name>")
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", choices=["fit", "select", "test"], default="fit")
    ap.add_argument("--per-class", type=int, default=60, help="Balanced sample size per label (0 = whole split)")
    ap.add_argument("--model-name", default=DEFAULT_MODEL)
    ap.add_argument("--renderer-name", default=DEFAULT_RENDERER)
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--square", type=int, default=DEFAULT_SQUARE_SIZE,
                    help="Centre-crop to square and resize to N px (0 = native aspect, for checkpoints trained "
                         "before the aspect-ratio fix). Must match training.")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    if args.split == "test" and not args.allow_test:
        raise SystemExit("Refusing to score the final test split without --allow-test")

    load_dotenv()
    from dfd.tinker_route.scorer import VLMScorer

    cfg = Config.from_yaml(args.config)
    _, base_val = build_base_datasets(cfg.data, cfg.train.seed)
    base = eval_splits(cfg.data, base_val, cfg.train.seed)[args.split]
    chosen = balanced_indices(base, args.per_class, cfg.train.seed)
    images, labels, gens, sevs = [], [], [], []
    for _, img, y, g, sev in paired_pte_images(cfg, base, chosen):
        images.append(img); labels.append(y); gens.append(g); sevs.append(sev)

    scorer = VLMScorer(model_path=args.sampler, model_name=args.model_name, renderer_name=args.renderer_name,
                       square_size=args.square)
    logits = scorer.score_many(images)
    report = sliced_report(labels, logits, gens, sevs)
    ov = report["overall"]
    print(f"[vlm|{args.split}] n={ov['n']} fakes={ov['n_fake']} auc={ov['auc']} tpr@1%fpr={ov['tpr@1%fpr']} "
          f"acc@0.5={ov['acc@0.5']}")
    for g, r in report["by_generator"].items():
        print(f"   {g:12s} auc={r['auc']}")
    out = Path(args.output or Path("runs") / f"vlm_score_{args.split}.json")
    out.write_text(json.dumps({"sampler": args.sampler, "report": report,
                               "split": args.split, "square": args.square,
                               "preds": [{"idx": i, "label": y, "generator": g, "severity": s, "logit": lg}
                                         for i, y, g, s, lg in zip(chosen, labels, gens, sevs, logits)]}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
