"""Training loop for the Stage 2 detector (CUDA / MPS / CPU)."""

from __future__ import annotations

import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from dfd.config import Config
from dfd.eval.metrics import sliced_report
from dfd.models.detector import DeepfakeDetector
from dfd.training.losses import BetaSchedule, detector_loss


def resolve_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict(model: DeepfakeDetector, loader: DataLoader, device: torch.device) -> dict[str, list]:
    model.eval()
    out: dict[str, list] = {"logits": [], "labels": [], "generators": [], "severities": []}
    for batch in loader:
        mask = batch.get("mask")
        logits = model(batch["pixels"].to(device), None if mask is None else mask.to(device))["logit"].float().cpu()
        out["logits"] += logits.tolist()
        out["labels"] += batch["label"].tolist()
        out["generators"] += list(batch["generator"])
        out["severities"] += list(batch["severity"])
    return out


class Trainer:
    def __init__(self, model: DeepfakeDetector, cfg: Config, train_loader: DataLoader, val_loader: DataLoader,
                 test_loader: DataLoader | None = None, fit_loader: DataLoader | None = None):
        self.cfg, self.tc = cfg, cfg.train
        self.device = resolve_device(self.tc.device)
        self.model = model.to(self.device)
        self.train_loader, self.val_loader, self.test_loader = train_loader, val_loader, test_loader
        self.fit_loader = fit_loader or val_loader
        self.out_dir = Path(self.tc.output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        groups = model.param_groups()
        param_groups = [{"params": groups["head"], "lr": self.tc.lr}]
        if groups["lora"]:
            param_groups.append({"params": groups["lora"], "lr": self.tc.lora_lr})
        self.optimizer = torch.optim.AdamW(param_groups, weight_decay=self.tc.weight_decay)

        steps_per_epoch = len(train_loader)
        self.total_steps = self.tc.max_steps or self.tc.epochs * steps_per_epoch
        warmup = int(self.tc.warmup_frac * self.total_steps)

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return (step + 1) / max(1, warmup)
            progress = (step - warmup) / max(1, self.total_steps - warmup)
            return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        self.beta = BetaSchedule(self.tc.beta_max, self.tc.beta_min, int(self.tc.beta_warmup_frac * self.total_steps))
        use_amp = self.tc.amp and self.device.type == "cuda"
        self.autocast = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if use_amp else nullcontext

        n_trainable = sum(p.numel() for g in groups.values() for p in g)
        print(f"[trainer] device={self.device} trainable_params={n_trainable:,} lora_layers={model.backbone.lora_count}")

    def _log(self, record: dict[str, Any]) -> None:
        with open(self.out_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")

    def evaluate(self) -> dict[str, Any]:
        preds = predict(self.model, self.val_loader, self.device)
        report = sliced_report(preds["labels"], preds["logits"], preds["generators"], preds["severities"])
        return {"report": report, "preds": preds}

    def train(self) -> dict[str, Any]:
        step, best_score, best = 0, -float("inf"), None
        epochs = self.tc.epochs if self.tc.max_steps is None else math.ceil(self.tc.max_steps / len(self.train_loader))
        for epoch in range(epochs):
            if hasattr(self.train_loader.dataset, "set_epoch"):
                self.train_loader.dataset.set_epoch(epoch)
            self.model.train()
            t0 = time.time()
            for batch in self.train_loader:
                pixels = batch["pixels"].to(self.device, non_blocking=True)
                labels = batch["label"].to(self.device)
                beauty = batch["beauty"].to(self.device)
                mask = batch.get("mask")
                with self.autocast():
                    out = self.model(pixels, None if mask is None else mask.to(self.device))
                out = {k: v.float() for k, v in out.items()}
                loss, logs = detector_loss(out, labels, self.beta(step), beauty, self.tc.beauty_weight)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for g in self.optimizer.param_groups for p in g["params"]], self.tc.grad_clip)
                self.optimizer.step()
                self.scheduler.step()
                step += 1
                if step % self.tc.log_every == 0:
                    logs.update(step=step, epoch=epoch, lr=self.scheduler.get_last_lr()[0])
                    self._log(logs)
                    print(f"[train] ep={epoch} step={step} " + " ".join(f"{k}={v:.4g}" for k, v in logs.items() if isinstance(v, float)))
                if step >= self.total_steps:
                    break
            result = self.evaluate()
            ov = result["report"]["overall"]
            score = ov["tpr@1%fpr"] if ov["tpr@1%fpr"] is not None else -1.0
            score += ov["auc"] or 0.0
            self._log({"epoch": epoch, "step": step, "val": ov, "epoch_time_s": time.time() - t0})
            print(f"[val] ep={epoch} auc={ov['auc']} tpr@1%fpr={ov['tpr@1%fpr']} ece={ov['ece']:.4f}")
            if score > best_score:
                best_score, best = score, result
                self.model.save(self.out_dir / "best.pt", extra={"epoch": epoch, "val": ov})
            if step >= self.total_steps:
                break
        self.model.save(self.out_dir / "last.pt", extra={"step": step})
        # The final generator is inspected only once, after checkpoint selection.
        if self.test_loader is not None:
            self.model, _ = DeepfakeDetector.load(self.out_dir / "best.pt")
            self.model.to(self.device).eval()
            test_preds = predict(self.model, self.test_loader, self.device)
            best["test_preds"] = test_preds
            best["test_report"] = sliced_report(
                test_preds["labels"], test_preds["logits"], test_preds["generators"], test_preds["severities"]
            )
            t = best["test_report"]["overall"]
            print(f"[test] auc={t['auc']} tpr@1%fpr={t['tpr@1%fpr']} (final generator, once)")
        if self.fit_loader is not self.val_loader:
            best["fit_preds"] = predict(self.model, self.fit_loader, self.device)
        return best
