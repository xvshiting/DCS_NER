"""
Training script for LDG-GLiNER.

Usage:
    python train.py [options]

Key options:
    --data_dir          Directory containing description JSONL files
    --backbone          Path or name of backbone model
    --model_arch        Architecture name (see MODEL_REGISTRY below)
    --output_dir        Directory for logs and config (default: checkpoints)
    --model_save_dir    Where to save best/last checkpoints (default: output_dir)
    --epochs            Number of training epochs
    --batch_size        Per-GPU batch size
    --grad_accum        Gradient accumulation steps
    --lr                Peak learning rate

Only two checkpoints are kept:
    <model_save_dir>/best  – highest val F1 seen so far
    <model_save_dir>/last  – end of training

Example:
    python train.py \\
        --data_dir dataset/ \\
        --backbone /data/model_hub/mdeberta-v3-base \\
        --model_arch deberta_span \\
        --output_dir runs/run1 \\
        --model_save_dir models/run1 \\
        --epochs 5 \\
        --batch_size 2 \\
        --grad_accum 8 \\
        --lr 2e-5
"""

import argparse
import json
import logging
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Type

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from ldggliner.data_processor import NERSpanDataset, build_collate_fn
from ldggliner.model import DebertaSchemaSpanModel, SpanBatch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# Add new architectures here. Each value is the model class; its __init__
# must accept the keyword arguments produced by build_model().

MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "deberta_span": DebertaSchemaSpanModel,
}


def build_model(cfg: "TrainConfig") -> nn.Module:
    """Instantiate a model from the registry using the current config."""
    if cfg.model_arch not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_arch '{cfg.model_arch}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    cls = MODEL_REGISTRY[cfg.model_arch]
    return cls(
        backbone_name=cfg.backbone,
        share_encoders=cfg.share_encoders,
        use_width_embedding=cfg.use_width_embedding,
        max_span_width=cfg.max_span_width,
        num_heads=cfg.num_heads,
        dropout=cfg.dropout,
        label_chunk_size=cfg.label_chunk_size,
    )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # Data
    data_dir: str = "dataset/"
    val_ratio: float = 0.02
    max_samples: Optional[int] = None      # limit samples (debug)
    max_text_len: int = 256
    max_label_len: int = 128
    max_span_width: int = 12
    max_total_labels: int = 25             # ALL pos labels + neg labels to fill budget
    label_chunk_size: int = 16            # labels encoded per chunk to bound GPU memory

    # Model
    model_arch: str = "deberta_span"       # key in MODEL_REGISTRY
    backbone: str = "/data/model_hub/mdeberta-v3-base"
    share_encoders: bool = True
    use_width_embedding: bool = True
    num_heads: int = 8
    dropout: float = 0.1

    # Paths
    output_dir: str = "checkpoints"        # logs, config
    model_save_dir: Optional[str] = None   # best/last model; defaults to output_dir

    # Training
    epochs: int = 5
    batch_size: int = 4
    grad_accum: int = 4
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    max_grad_norm: float = 1.0
    seed: int = 42

    # Eval & logging
    eval_steps: int = 500
    log_steps: int = 50
    threshold: float = 0.5

    # Mixed precision
    use_amp: bool = True


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train LDG-GLiNER")
    cfg = TrainConfig()

    # Data
    parser.add_argument("--data_dir", default=cfg.data_dir)
    parser.add_argument("--val_ratio", type=float, default=cfg.val_ratio)
    parser.add_argument("--max_samples", type=int, default=cfg.max_samples)
    parser.add_argument("--max_text_len", type=int, default=cfg.max_text_len)
    parser.add_argument("--max_label_len", type=int, default=cfg.max_label_len)
    parser.add_argument("--max_span_width", type=int, default=cfg.max_span_width)
    parser.add_argument("--max_total_labels", type=int, default=cfg.max_total_labels)
    parser.add_argument("--label_chunk_size", type=int, default=cfg.label_chunk_size)

    # Model
    parser.add_argument("--model_arch", default=cfg.model_arch,
                        choices=list(MODEL_REGISTRY.keys()),
                        help="Model architecture to train")
    parser.add_argument("--backbone", default=cfg.backbone)
    parser.add_argument("--share_encoders", action="store_true", default=cfg.share_encoders)
    parser.add_argument("--no_share_encoders", dest="share_encoders", action="store_false")
    parser.add_argument("--use_width_embedding", action="store_true", default=cfg.use_width_embedding)
    parser.add_argument("--no_width_embedding", dest="use_width_embedding", action="store_false")
    parser.add_argument("--num_heads", type=int, default=cfg.num_heads)
    parser.add_argument("--dropout", type=float, default=cfg.dropout)

    # Paths
    parser.add_argument("--output_dir", default=cfg.output_dir,
                        help="Directory for logs and config")
    parser.add_argument("--model_save_dir", default=cfg.model_save_dir,
                        help="Where to save best/last checkpoints (default: output_dir)")

    # Training
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--batch_size", type=int, default=cfg.batch_size)
    parser.add_argument("--grad_accum", type=int, default=cfg.grad_accum)
    parser.add_argument("--lr", type=float, default=cfg.lr)
    parser.add_argument("--weight_decay", type=float, default=cfg.weight_decay)
    parser.add_argument("--warmup_ratio", type=float, default=cfg.warmup_ratio)
    parser.add_argument("--max_grad_norm", type=float, default=cfg.max_grad_norm)
    parser.add_argument("--seed", type=int, default=cfg.seed)

    # Eval & logging
    parser.add_argument("--eval_steps", type=int, default=cfg.eval_steps)
    parser.add_argument("--log_steps", type=int, default=cfg.log_steps)
    parser.add_argument("--threshold", type=float, default=cfg.threshold)

    # Mixed precision
    parser.add_argument("--use_amp", action="store_true", default=cfg.use_amp)
    parser.add_argument("--no_amp", dest="use_amp", action="store_false")

    args = parser.parse_args()
    return TrainConfig(**vars(args))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@dataclass
class SpanMetrics:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def precision(self) -> float:
        return self.tp / max(self.tp + self.fp, 1)

    def recall(self) -> float:
        return self.tp / max(self.tp + self.fn, 1)

    def f1(self) -> float:
        p, r = self.precision(), self.recall()
        return 2 * p * r / max(p + r, 1e-8)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float,
    amp_dtype: Optional[torch.dtype] = None,
) -> Dict[str, float]:
    """Micro-averaged span-level precision / recall / F1."""
    model.eval()
    metrics = SpanMetrics()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = batch_to_device(batch, device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=amp_dtype is not None):
                out = model(batch)

            if "loss" in out:
                total_loss += out["loss"].item()
                n_batches += 1

            logits = out["logits"]       # (B, S, M)
            span_mask = out["span_mask"] # (B, S)
            targets = batch.span_targets # (B, S, M)

            preds = (torch.sigmoid(logits) > threshold) & span_mask.unsqueeze(-1)

            if targets is not None:
                gold = targets.bool()
                valid = span_mask.unsqueeze(-1).expand_as(gold)
                preds_v = preds & valid
                gold_v = gold & valid
                metrics.tp += int((preds_v & gold_v).sum())
                metrics.fp += int((preds_v & ~gold_v).sum())
                metrics.fn += int((~preds_v & gold_v).sum())

    model.train()
    return {
        "val_loss": total_loss / max(n_batches, 1),
        "precision": metrics.precision(),
        "recall": metrics.recall(),
        "f1": metrics.f1(),
    }


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def batch_to_device(batch: SpanBatch, device: torch.device) -> SpanBatch:
    return SpanBatch(
        text_input_ids=batch.text_input_ids.to(device),
        text_attention_mask=batch.text_attention_mask.to(device),
        label_input_ids=batch.label_input_ids.to(device),
        label_attention_mask=batch.label_attention_mask.to(device),
        num_labels=batch.num_labels,
        span_targets=(batch.span_targets.to(device)
                      if batch.span_targets is not None else None),
    )


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    cfg: TrainConfig,
    step: int,
    metrics: Dict[str, float],
    tag: str,               # "best" or "last"
    model_save_dir: str,
):
    """Save model + training state to <model_save_dir>/<tag>/."""
    path = os.path.join(model_save_dir, tag)
    os.makedirs(path, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "step": step,
            "metrics": metrics,
            "config": vars(cfg),
        },
        os.path.join(path, "checkpoint.pt"),
    )
    logger.info(f"[{tag}] Saved checkpoint to {path}  metrics={metrics}")


def load_checkpoint(path: str, model: nn.Module, optimizer=None, scheduler=None):
    ckpt = torch.load(os.path.join(path, "checkpoint.pt"), map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt["step"], ckpt.get("metrics", {})


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg: TrainConfig):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )
    set_seed(cfg.seed)

    # Resolve paths
    output_dir = cfg.output_dir
    model_save_dir = cfg.model_save_dir or output_dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)

    # Persist config
    with open(os.path.join(output_dir, "train_config.json"), "w") as f:
        json.dump(vars(cfg), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}  |  arch: {cfg.model_arch}  |  backbone: {cfg.backbone}")

    # ---- Tokenizer & Dataset ----
    tokenizer = AutoTokenizer.from_pretrained(cfg.backbone)

    logger.info(f"Loading dataset from {cfg.data_dir}")
    full_dataset = NERSpanDataset(
        data_dir=cfg.data_dir,
        tokenizer=tokenizer,
        max_text_len=cfg.max_text_len,
        max_label_len=cfg.max_label_len,
        max_span_width=cfg.max_span_width,
        max_total_labels=cfg.max_total_labels,
        max_samples=cfg.max_samples,
    )
    logger.info(f"Total samples: {len(full_dataset)}")

    n_val = max(1, int(len(full_dataset) * cfg.val_ratio))
    n_train = len(full_dataset) - n_val
    train_dataset, val_dataset = random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    logger.info(f"Train: {n_train}  Val: {n_val}")

    collate = build_collate_fn(tokenizer=tokenizer, max_span_width=cfg.max_span_width)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate,
    )

    # ---- Model ----
    logger.info(f"Building model: {cfg.model_arch}")
    model = build_model(cfg)

    # ---- Dtype & device ----
    # Unify all parameters to a single dtype to avoid mismatches between the
    # fp16 backbone checkpoint and fp32-initialized task-specific layers.
    # bfloat16: same exponent range as fp32, no GradScaler needed, ~half memory.
    # fp32 fallback: if GPU has no bf16 support.
    amp_dtype: Optional[torch.dtype] = None
    scaler = None
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            model.to(device=device, dtype=torch.bfloat16)
            amp_dtype = torch.bfloat16
            logger.info("Model dtype: bfloat16 | AMP: bfloat16")
        else:
            model.to(device=device, dtype=torch.float32)
            amp_dtype = torch.float16
            scaler = torch.cuda.amp.GradScaler()
            logger.info("Model dtype: float32 | AMP: float16 + GradScaler")
    else:
        model.to(device=device, dtype=torch.float32)
        logger.info("Model dtype: float32 | AMP: disabled (CPU)")

    if not cfg.use_amp:
        amp_dtype = None
        scaler = None
        logger.info("AMP disabled by --no_amp")

    # ---- Optimizer ----
    # Backbone gets the base lr; task-specific heads get a higher lr to train faster.
    backbone_params = list(model.text_encoder.parameters())
    if not cfg.share_encoders:
        backbone_params += list(model.label_encoder.parameters())
    backbone_ids = {id(p) for p in backbone_params}
    head_params = [p for p in model.parameters() if id(p) not in backbone_ids]

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": cfg.lr},
            {"params": head_params,     "lr": cfg.lr * 5},
        ],
        weight_decay=cfg.weight_decay,
    )

    steps_per_epoch = math.ceil(len(train_loader) / cfg.grad_accum)
    total_steps = steps_per_epoch * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    logger.info(
        f"Steps: {total_steps} total / {warmup_steps} warmup | "
        f"Effective batch: {cfg.batch_size * cfg.grad_accum}"
    )
    logger.info(f"Checkpoints → best/last saved to: {model_save_dir}")

    # ---- Training loop ----
    best_f1 = 0.0
    global_step = 0
    optimizer.zero_grad()

    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        accum_count = 0

        for micro_step, batch in enumerate(train_loader):
            batch = batch_to_device(batch, device)

            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=amp_dtype is not None):
                out = model(batch)

            if "loss" not in out:
                logger.warning("No loss in output — check that span_targets are provided.")
                continue

            loss = out["loss"] / cfg.grad_accum
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            running_loss += out["loss"].item()
            accum_count += 1

            if (micro_step + 1) % cfg.grad_accum == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % cfg.log_steps == 0:
                    avg_loss = running_loss / max(accum_count, 1)
                    logger.info(
                        f"Epoch {epoch+1} | Step {global_step}/{total_steps} | "
                        f"Loss {avg_loss:.4f} | LR {scheduler.get_last_lr()[0]:.2e}"
                    )
                    running_loss = 0.0
                    accum_count = 0

                # Mid-epoch evaluation → update best
                if global_step % cfg.eval_steps == 0:
                    metrics = evaluate(model, val_loader, device, cfg.threshold, amp_dtype)
                    logger.info(
                        f"[Eval] Step {global_step} | "
                        f"Loss {metrics['val_loss']:.4f} | "
                        f"P {metrics['precision']:.4f} | "
                        f"R {metrics['recall']:.4f} | "
                        f"F1 {metrics['f1']:.4f}"
                    )
                    if metrics["f1"] > best_f1:
                        best_f1 = metrics["f1"]
                        save_checkpoint(model, optimizer, scheduler, cfg,
                                        global_step, metrics, "best", model_save_dir)

        # End-of-epoch evaluation → update best
        metrics = evaluate(model, val_loader, device, cfg.threshold, amp_dtype)
        logger.info(
            f"[Epoch {epoch+1} End] "
            f"Loss {metrics['val_loss']:.4f} | "
            f"P {metrics['precision']:.4f} | "
            f"R {metrics['recall']:.4f} | "
            f"F1 {metrics['f1']:.4f}"
        )
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            save_checkpoint(model, optimizer, scheduler, cfg,
                            global_step, metrics, "best", model_save_dir)

    # Save last model (overwrite previous last)
    save_checkpoint(model, optimizer, scheduler, cfg,
                    global_step, metrics, "last", model_save_dir)

    logger.info(f"Training complete. Best val F1: {best_f1:.4f}")
    logger.info(f"best → {os.path.join(model_save_dir, 'best')}")
    logger.info(f"last → {os.path.join(model_save_dir, 'last')}")


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
