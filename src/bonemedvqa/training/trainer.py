"""Main training loop."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from torch.amp import GradScaler, autocast
except ImportError:  # pragma: no cover
    from torch.cuda.amp import GradScaler, autocast

from bonemedvqa.evaluation.closed_metrics import compute_closed_metrics
from bonemedvqa.losses.combined_loss import CombinedLoss
from bonemedvqa.training.callbacks import EarlyStopping
from bonemedvqa.training.checkpoint import load_checkpoint, save_checkpoint
from bonemedvqa.training.scheduler import build_scheduler
from bonemedvqa.utils.io import ensure_dir, save_json
from bonemedvqa.utils.logger import get_logger


class Trainer:
    """Train / validate BoneMedVQA models."""

    def __init__(
        self,
        model: torch.nn.Module,
        cfg: dict[str, Any],
        device: torch.device,
        id_to_label: dict[int, str] | None = None,
    ):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.id_to_label = id_to_label or {}
        self.logger = get_logger("bonemedvqa.trainer")
        train_cfg = cfg.get("train", {})
        self.epochs = int(train_cfg.get("epochs", 5))
        self.grad_clip = float(train_cfg.get("grad_clip", 1.0))
        self.accumulate = max(1, int(train_cfg.get("accumulate_grad_batches", 1)))
        self.amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
        try:
            self.scaler = GradScaler("cuda", enabled=self.amp)
        except TypeError:  # older torch
            self.scaler = GradScaler(enabled=self.amp)

        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            params,
            lr=float(train_cfg.get("lr", 1e-4)),
            weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        )
        self.scheduler = build_scheduler(self.optimizer, train_cfg)
        self.answer_embedding = torch.nn.Embedding(
            num_embeddings=getattr(model.closed_head, "num_classes", 2),
            embedding_dim=model.hidden_dim,
        ).to(device)
        self.criterion = CombinedLoss(cfg, answer_embedding=self.answer_embedding)
        self.early_stopping = EarlyStopping(
            patience=int(train_cfg.get("early_stopping_patience", 5)),
            mode="max",
        )
        out_cfg = cfg.get("output", {})
        self.ckpt_dir = ensure_dir(out_cfg.get("checkpoint_dir", "outputs/checkpoints"))
        self.log_dir = ensure_dir(out_cfg.get("log_dir", "outputs/logs"))
        self.best_metric = -1.0
        self.save_metric_name = str(out_cfg.get("save_best_metric", "macro_f1"))

        resume = train_cfg.get("resume")
        self.start_epoch = 0
        if resume:
            payload = load_checkpoint(resume, self.model, self.optimizer, map_location=device)
            self.start_epoch = int(payload.get("epoch", 0)) + 1
            self.logger.info("Resumed from %s at epoch %s", resume, self.start_epoch)

    def _move_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        out = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                out[k] = v.to(self.device)
            else:
                out[k] = v
        return out

    def train_one_epoch(self, loader: DataLoader, epoch: int) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        n = 0
        self.optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(loader, desc=f"train epoch {epoch}", leave=False)
        for step, batch in enumerate(pbar, start=1):
            batch = self._move_batch(batch)
            try:
                amp_ctx = autocast("cuda", enabled=self.amp)
            except TypeError:
                amp_ctx = autocast(enabled=self.amp)
            with amp_ctx:
                outputs = self.model(batch)
                losses = self.criterion(outputs, batch)
                loss = losses["total"] / self.accumulate
            if self.amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % self.accumulate == 0:
                if self.amp:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                if self.amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            total_loss += float(losses["total"].detach().cpu())
            n += 1
            pbar.set_postfix(loss=total_loss / max(n, 1))
        if self.scheduler is not None:
            self.scheduler.step()
        return {"loss": total_loss / max(n, 1)}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict[str, Any]:
        self.model.eval()
        all_preds, all_labels, all_probs = [], [], []
        total_loss = 0.0
        n = 0
        for batch in loader:
            batch = self._move_batch(batch)
            outputs = self.model(batch)
            losses = self.criterion(outputs, batch)
            total_loss += float(losses["total"].detach().cpu())
            n += 1
            preds = outputs["pred"].detach().cpu().tolist()
            labels = batch["labels"].detach().cpu().tolist()
            probs = outputs["probs"].detach().cpu()
            for p, y, pr in zip(preds, labels, probs):
                if y < 0:
                    continue
                all_preds.append(p)
                all_labels.append(y)
                all_probs.append(pr.tolist())
        metrics = compute_closed_metrics(all_labels, all_preds, all_probs)
        metrics["loss"] = total_loss / max(n, 1)
        return metrics

    def fit(self, train_loader: DataLoader, val_loader: DataLoader | None = None) -> dict[str, Any]:
        history = []
        for epoch in range(self.start_epoch, self.epochs):
            train_stats = self.train_one_epoch(train_loader, epoch)
            val_stats = self.evaluate(val_loader) if val_loader is not None else {}
            row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_stats.items()}, **{f"val_{k}": v for k, v in val_stats.items()}}
            history.append(row)
            self.logger.info("Epoch %s | %s", epoch, row)
            save_json(Path(self.log_dir) / "history.json", history)

            metric = float(val_stats.get(self.save_metric_name, val_stats.get("accuracy", -1)))
            if metric > self.best_metric:
                self.best_metric = metric
                save_checkpoint(
                    Path(self.ckpt_dir) / "best.pt",
                    self.model,
                    self.optimizer,
                    epoch=epoch,
                    metrics=val_stats,
                    extra={
                        "id_to_label": self.id_to_label,
                        "cfg": self.cfg,
                        "trainable_params": self.model.count_trainable_parameters(),
                    },
                )
                self.logger.info("Saved best checkpoint (%.4f)", metric)
            save_checkpoint(
                Path(self.ckpt_dir) / "last.pt",
                self.model,
                self.optimizer,
                epoch=epoch,
                metrics=val_stats,
                extra={"id_to_label": self.id_to_label, "cfg": self.cfg},
            )
            if val_loader is not None and self.early_stopping.step(metric):
                self.logger.info("Early stopping at epoch %s", epoch)
                break
        return {"history": history, "best_metric": self.best_metric}
