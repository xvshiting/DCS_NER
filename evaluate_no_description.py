"""
evaluate_no_description.py
==========================
Evaluate LDG-GLiNER using label names only — no descriptions, no LLM.

This is the label-only baseline: the model receives just the entity type
string (e.g. "person") without any contextual description.  Use this to
measure how much the description augmentation contributes.

Shares all model loading / decoding logic with evaluate.py via imports;
only the description step is removed.

Usage:
    python evaluate_no_description.py \\
        --gliner_ckpt checkpoints/run9/best/checkpoint.pt \\
        --backbone    /data/model_hub/mdeberta-v3-base \\
        --test_data   /data/dataset/ner/instruct_uie_ner \\
        --output_dir  eval_results/no_desc

    # MultiCoNER2022 (per-language results):
    python evaluate_no_description.py \\
        --gliner_ckpt checkpoints/run9/best/checkpoint.pt \\
        --backbone    /data/model_hub/mdeberta-v3-base \\
        --test_data   /data/dataset/ner/multiconer2022_arrow \\
        --output_dir  eval_results/multiconer_no_desc
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from evaluate import (
    SubsetMetrics,
    load_gliner,
    encode_sample,
    collate_for_eval,
    decode_predictions,
    normalize_gold_spans,
    EvalConfig,
)
from ldggliner.model import SpanBatch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config  (description-related fields intentionally absent)
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # GLiNER model
    gliner_ckpt: str = "checkpoints/best/checkpoint.pt"
    backbone: str = "/data/model_hub/mdeberta-v3-base"
    model_arch: str = "deberta_span_v1"
    max_span_width: int = 12
    label_chunk_size: int = 16
    threshold: float = 0.5
    flat_ner: bool = True
    multi_label: bool = False
    max_text_len: int = 256
    max_label_len: int = 128

    # Test data
    test_data: str = "/data/dataset/ner/instruct_uie_ner"
    subsets: Optional[List[str]] = None
    max_samples_per_subset: Optional[int] = None
    batch_size: int = 8

    # Output
    output_dir: str = "eval_results/no_desc"
    save_predictions: bool = False


def parse_args() -> Config:
    p = argparse.ArgumentParser(
        description="Evaluate LDG-GLiNER with label names only (no descriptions)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    d = Config()
    p.add_argument("--gliner_ckpt",  default=d.gliner_ckpt)
    p.add_argument("--backbone",     default=d.backbone)
    p.add_argument("--model_arch",   default=d.model_arch,
                   choices=["deberta_span_v1", "deberta_span_v2", "deberta_span_v3"])
    p.add_argument("--max_span_width",  type=int,   default=d.max_span_width)
    p.add_argument("--label_chunk_size",type=int,   default=d.label_chunk_size)
    p.add_argument("--threshold",       type=float, default=d.threshold)
    p.add_argument("--flat_ner",   action=argparse.BooleanOptionalAction, default=d.flat_ner,
                   help="Greedy NMS: suppress overlapping spans")
    p.add_argument("--multi_label", action=argparse.BooleanOptionalAction, default=d.multi_label,
                   help="Global NMS across labels (vs per-label)")
    p.add_argument("--max_text_len",  type=int, default=d.max_text_len)
    p.add_argument("--max_label_len", type=int, default=d.max_label_len)
    p.add_argument("--test_data",     default=d.test_data)
    p.add_argument("--subsets", nargs="+", default=None, metavar="SUBSET")
    p.add_argument("--max_samples_per_subset", type=int, default=d.max_samples_per_subset)
    p.add_argument("--batch_size",    type=int, default=d.batch_size)
    p.add_argument("--output_dir",    default=d.output_dir)
    p.add_argument("--save_predictions", action="store_true", default=d.save_predictions)
    args = p.parse_args()
    return Config(**vars(args))


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(cfg: Config):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )
    os.makedirs(cfg.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ---- Load test data ----
    from datasets import load_from_disk
    logger.info(f"Loading test data: {cfg.test_data}")
    test_split = load_from_disk(cfg.test_data)["test"]
    logger.info(f"Test samples: {len(test_split)}")

    # Group by subset
    subset_indices: Dict[str, List[int]] = defaultdict(list)
    for i, s in enumerate(test_split):
        subset_indices[s["dataset"]].append(i)

    subsets = sorted(subset_indices.keys())
    logger.info(f"Subsets ({len(subsets)}): {subsets}")

    if cfg.subsets:
        unknown = set(cfg.subsets) - set(subsets)
        if unknown:
            logger.warning(f"Unknown subsets (ignored): {sorted(unknown)}")
        subsets = [s for s in subsets if s in set(cfg.subsets)]
        logger.info(f"Evaluating ({len(subsets)}): {subsets}")

    if cfg.max_samples_per_subset:
        for k in subset_indices:
            subset_indices[k] = subset_indices[k][: cfg.max_samples_per_subset]

    # ---- Load tokenizer & model ----
    logger.info(f"Loading tokenizer: {cfg.backbone}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.backbone)

    # Build an EvalConfig just for load_gliner (desc fields are unused here)
    eval_cfg = EvalConfig(
        gliner_ckpt=cfg.gliner_ckpt,
        backbone=cfg.backbone,
        model_arch=cfg.model_arch,
        max_span_width=cfg.max_span_width,
        label_chunk_size=cfg.label_chunk_size,
        threshold=cfg.threshold,
        flat_ner=cfg.flat_ner,
        multi_label=cfg.multi_label,
        max_text_len=cfg.max_text_len,
        max_label_len=cfg.max_label_len,
    )
    logger.info(f"Loading GLiNER: {cfg.gliner_ckpt}")
    model = load_gliner(eval_cfg, device)
    logger.info("Model ready  [label-only mode — no descriptions]")

    amp_dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else None
    )

    # ---- Evaluate ----
    subset_metrics: Dict[str, SubsetMetrics] = {s: SubsetMetrics() for s in subsets}
    overall = SubsetMetrics()

    pred_file = None
    if cfg.save_predictions:
        pred_path = os.path.join(cfg.output_dir, "predictions.jsonl")
        pred_file = open(pred_path, "w", encoding="utf-8")
        logger.info(f"Saving predictions → {pred_path}")

    for subset_name in subsets:
        indices = subset_indices[subset_name]
        sm = subset_metrics[subset_name]
        n_batches = (len(indices) + cfg.batch_size - 1) // cfg.batch_size

        pbar = tqdm(
            range(0, len(indices), cfg.batch_size),
            total=n_batches,
            desc=subset_name,
            unit="batch",
            dynamic_ncols=True,
        )

        for batch_start in pbar:
            batch_indices = indices[batch_start: batch_start + cfg.batch_size]
            raw_samples   = [test_split[i] for i in batch_indices]

            encoded_batch: List[Dict] = []
            label_lists:   List[List[str]] = []
            gold_sets:     List[Set[Tuple[int, int, str]]] = []

            for raw in raw_samples:
                text = raw["sentence"]

                raw_gold: List[Tuple[int, int, str]] = [
                    (e["pos"][0], e["pos"][1], e["type"].lower())
                    for e in raw.get("entities", [])
                    if len(e.get("pos", [])) >= 2
                ]
                types = list(dict.fromkeys(t for _, _, t in raw_gold))
                if not types:
                    continue

                # Label-only: empty description for every type
                label_descs = {t: "" for t in types}

                enc = encode_sample(text, label_descs, tokenizer,
                                    cfg.max_text_len, cfg.max_label_len)
                if enc is None:
                    continue

                gold = normalize_gold_spans(raw_gold, enc["offset_mapping"],
                                            cfg.max_span_width)
                encoded_batch.append(enc)
                label_lists.append(enc["labels"])
                gold_sets.append(gold)

            if not encoded_batch:
                continue

            batch, offset_mappings = collate_for_eval(encoded_batch, tokenizer)
            batch = SpanBatch(
                text_input_ids=batch.text_input_ids.to(device),
                text_attention_mask=batch.text_attention_mask.to(device),
                label_input_ids=batch.label_input_ids.to(device),
                label_attention_mask=batch.label_attention_mask.to(device),
                num_labels=batch.num_labels,
                span_targets=None,
            )

            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                    enabled=amp_dtype is not None):
                    out = model(batch)

            pred_sets = decode_predictions(
                logits=out["logits"].cpu(),
                span_mask=out["span_mask"].cpu(),
                start_idx=out["start_idx"].cpu(),
                end_idx=out["end_idx"].cpu(),
                offset_mappings=offset_mappings,
                label_lists=label_lists,
                threshold=cfg.threshold,
                flat_ner=cfg.flat_ner,
                multi_label=cfg.multi_label,
            )

            for (s_idx, raw_s), preds, gold in zip(
                zip(batch_indices, raw_samples), pred_sets, gold_sets
            ):
                sm.tp += len(preds & gold)
                sm.fp += len(preds - gold)
                sm.fn += len(gold - preds)
                sm.n_samples += 1
                overall.tp += len(preds & gold)
                overall.fp += len(preds - gold)
                overall.fn += len(gold - preds)
                overall.n_samples += 1

                if pred_file is not None:
                    def _fmt(span_set):
                        return [{"char_start": cs, "char_end": ce, "type": t}
                                for cs, ce, t in sorted(span_set)]
                    pred_file.write(json.dumps({
                        "split_idx": s_idx,
                        "dataset":   subset_name,
                        "sentence":  raw_s.get("sentence", ""),
                        "gold": _fmt(gold),
                        "pred": _fmt(preds),
                        "tp":   _fmt(preds & gold),
                        "fp":   _fmt(preds - gold),
                        "fn":   _fmt(gold - preds),
                    }, ensure_ascii=False) + "\n")

            pbar.set_postfix(
                P=f"{sm.precision():.3f}",
                R=f"{sm.recall():.3f}",
                F1=f"{sm.f1():.3f}",
            )

        pbar.close()
        logger.info(f"  {subset_name}: {sm}")

    # ---- Print results ----
    print("\n" + "=" * 70)
    print(f"{'Subset':<35} {'P':>7} {'R':>7} {'F1':>7} {'N':>8}")
    print("-" * 70)
    rows = []
    for name in sorted(subset_metrics):
        sm = subset_metrics[name]
        if sm.n_samples == 0:
            continue
        print(f"{name:<35} {sm.precision():>7.4f} {sm.recall():>7.4f} "
              f"{sm.f1():>7.4f} {sm.n_samples:>8}")
        rows.append({
            "subset": name,
            "precision": sm.precision(), "recall": sm.recall(), "f1": sm.f1(),
            "n_samples": sm.n_samples,
            "tp": sm.tp, "fp": sm.fp, "fn": sm.fn,
        })
    print("-" * 70)
    print(f"{'OVERALL':<35} {overall.precision():>7.4f} {overall.recall():>7.4f} "
          f"{overall.f1():>7.4f} {overall.n_samples:>8}")

    # Macro F1
    f1s = [sm["f1"] for sm in rows if sm["n_samples"] > 0]
    macro = sum(f1s) / len(f1s) if f1s else 0.0
    print(f"{'MACRO F1':<35} {macro:>7.4f}")
    print("=" * 70)

    if pred_file is not None:
        pred_file.close()

    # ---- Save results ----
    out_path = os.path.join(cfg.output_dir, "results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": vars(cfg),
            "overall": {
                "precision": overall.precision(), "recall": overall.recall(),
                "f1": overall.f1(), "n_samples": overall.n_samples,
                "tp": overall.tp, "fp": overall.fp, "fn": overall.fn,
            },
            "macro_f1": macro,
            "subsets": rows,
        }, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved → {out_path}")


if __name__ == "__main__":
    cfg = parse_args()
    run_evaluation(cfg)
