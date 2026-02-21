"""
Evaluation script for LDG-GLiNER.

Pipeline per test sample:
  1. Load sample from instruct_uie_ner test split (HuggingFace Arrow format)
  2. Generate label descriptions via the fine-tuned LLM (closed mode)
     OR look them up from cached training JSONL descriptions
  3. Run the trained GLiNER model to predict entity spans
  4. Decode token-level spans → character-level spans
  5. Exact-match against gold entities → per-subset and overall F1

Usage:
    python evaluate.py \\
        --gliner_ckpt checkpoints/best/checkpoint.pt \\
        --backbone /data/model_hub/mdeberta-v3-base \\
        --test_data /data/dataset/ner/instruct_uie_ner \\
        --desc_mode llm \\
        --llm_base /data/model_hub/qwen/Qwen3-1.7B \\
        --llm_adapter /home/will/Projects/LLaMA-Factory/saves/.../lora/... \\
        --output_dir eval_results/run1

Description modes (--desc_mode):
    llm    Use the fine-tuned LLM to generate descriptions on-the-fly (closed mode)
    cache  Look up descriptions from training JSONL files (fast, no LLM needed)
    none   Use the label name only, no description
"""

import argparse
import json
import logging
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class EvalConfig:
    # GLiNER model
    gliner_ckpt: str = "checkpoints/best/checkpoint.pt"
    backbone: str = "/data/model_hub/mdeberta-v3-base"
    model_arch: str = "deberta_span_v1"   # deberta_span_v1 | deberta_span_v2 | deberta_span_v3
    max_span_width: int = 12
    label_chunk_size: int = 16
    threshold: float = 0.5

    # Test data
    test_data: str = "/data/dataset/ner/instruct_uie_ner"
    subsets: Optional[List[str]] = None            # None = run all subsets
    max_samples_per_subset: Optional[int] = None   # None = use all
    batch_size: int = 8

    # Description source
    desc_mode: str = "llm"                         # llm | cache | none | precomputed
    desc_cache_dir: str = "dataset/"               # JSONL dir for 'cache' mode
    precomputed_descs: str = "descriptions_cache.jsonl"  # for 'precomputed' mode
    llm_base: str = "/data/model_hub/qwen/Qwen3-1.7B"
    llm_adapter: Optional[str] = "/home/will/Projects/LLaMA-Factory/saves/Qwen3-1.7B-Thinking/lora/train_2026-02-09-12-46-24"
    llm_max_new_tokens: int = 512
    llm_desc_key: str = "D6"                       # which definition to use from cached descs

    # Text limits
    max_text_len: int = 256
    max_label_len: int = 128

    # Output
    output_dir: str = "eval_results"
    save_predictions: bool = False   # write per-sample predictions to predictions.jsonl


def parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser(description="Evaluate LDG-GLiNER on instruct_uie_ner test set")
    cfg = EvalConfig()

    parser.add_argument("--gliner_ckpt", default=cfg.gliner_ckpt)
    parser.add_argument("--backbone", default=cfg.backbone)
    parser.add_argument("--model_arch", default=cfg.model_arch,
                        choices=["deberta_span_v1", "deberta_span_v2", "deberta_span_v3"])
    parser.add_argument("--max_span_width", type=int, default=cfg.max_span_width)
    parser.add_argument("--label_chunk_size", type=int, default=cfg.label_chunk_size)
    parser.add_argument("--threshold", type=float, default=cfg.threshold)

    parser.add_argument("--test_data", default=cfg.test_data)
    parser.add_argument("--subsets", nargs="+", default=None,
                        metavar="SUBSET",
                        help="Subset name(s) to evaluate; default runs all subsets")
    parser.add_argument("--max_samples_per_subset", type=int, default=cfg.max_samples_per_subset)
    parser.add_argument("--batch_size", type=int, default=cfg.batch_size)

    parser.add_argument("--desc_mode", default=cfg.desc_mode,
                        choices=["llm", "cache", "none", "precomputed"])
    parser.add_argument("--desc_cache_dir", default=cfg.desc_cache_dir)
    parser.add_argument("--precomputed_descs", default=cfg.precomputed_descs)
    parser.add_argument("--llm_base", default=cfg.llm_base)
    parser.add_argument("--llm_adapter", default=cfg.llm_adapter)
    parser.add_argument("--llm_max_new_tokens", type=int, default=cfg.llm_max_new_tokens)
    parser.add_argument("--llm_desc_key", default=cfg.llm_desc_key)

    parser.add_argument("--max_text_len", type=int, default=cfg.max_text_len)
    parser.add_argument("--max_label_len", type=int, default=cfg.max_label_len)
    parser.add_argument("--output_dir", default=cfg.output_dir)
    parser.add_argument("--save_predictions", action="store_true", default=cfg.save_predictions)

    args = parser.parse_args()
    return EvalConfig(**vars(args))


# ---------------------------------------------------------------------------
# Description provider
# ---------------------------------------------------------------------------

CLOSED_TEMPLATE = """You are given a text and a set of target labels.
Generate descriptions ONLY for the labels listed below.
Do NOT generate any labels that are not listed.

<Text>
{TEXT}
</Text>

<LABEL_SET>
{LABEL_SET}
</LABEL_SET>

Output format (STRICT):
Each line must be:
LABEL: DESCRIPTION

Each LABEL must appear exactly once.
Do not add extra text or blank lines."""


def _parse_llm_output(text: str) -> Dict[str, str]:
    """Parse 'LABEL: DESC' lines from LLM output into {label: desc}."""
    result = {}
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^(.+?):\s+(.+)$", line)
        if m:
            result[m.group(1).strip().lower()] = m.group(2).strip()
    return result


class DescriptionProvider:
    """
    Provides label descriptions from one of four sources:
      - 'llm'         : fine-tuned LLM generates descriptions per sample (slow)
      - 'cache'       : pre-computed descriptions from training JSONL files
      - 'precomputed' : load from descriptions_cache.jsonl produced by precompute_descriptions.py
      - 'none'        : returns empty string (label name only)
    """

    def __init__(self, cfg: EvalConfig):
        self.mode = cfg.desc_mode
        self.cfg = cfg
        self._llm_model = None
        self._llm_tokenizer = None
        self._cache: Dict[str, str] = {}        # label -> description  (cache mode)
        self._precomputed: Dict[int, Dict[str, str]] = {}  # split_idx -> {label: desc}

        if self.mode == "llm":
            self._load_llm()
        elif self.mode == "cache":
            self._build_cache()
        elif self.mode == "precomputed":
            self._load_precomputed()

    def _load_llm(self):
        from transformers import AutoModelForCausalLM
        logger.info(f"Loading LLM from {self.cfg.llm_base}")
        self._llm_tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.llm_base, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.cfg.llm_base,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        if self.cfg.llm_adapter:
            from peft import PeftModel
            logger.info(f"Loading LoRA adapter from {self.cfg.llm_adapter}")
            model = PeftModel.from_pretrained(model, self.cfg.llm_adapter)
        model.eval()
        self._llm_model = model
        logger.info("LLM loaded.")

    def _build_cache(self):
        """Load descriptions from training JSONL files into {label: desc} dict."""
        import glob
        pattern = os.path.join(
            self.cfg.desc_cache_dir,
            "instruct_uie_ner_converted_description_*.jsonl"
        )
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(
                f"No description JSONL files found at {pattern}"
            )
        key = self.cfg.llm_desc_key
        logger.info(f"Building description cache from {len(files)} JSONL files (key={key})...")
        seen = 0
        for fpath in files:
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line.strip())
                    desc = obj.get("description", [])
                    if isinstance(desc, list):
                        for d in desc:
                            label = d.get("label", "").lower()
                            defs = d.get("definitions", {})
                            if label and key in defs:
                                self._cache.setdefault(label, defs[key])
                                seen += 1
                    elif isinstance(desc, dict):
                        for label, defs in desc.items():
                            if key in defs:
                                self._cache.setdefault(label.lower(), defs[key])
                                seen += 1
        logger.info(f"Cache ready: {len(self._cache)} unique labels.")

    def _load_precomputed(self):
        path = self.cfg.precomputed_descs
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Precomputed descriptions file not found: {path}\n"
                "Run precompute_descriptions.py first."
            )
        count = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                idx = obj["split_idx"]
                self._precomputed[idx] = obj.get("descriptions", {})
                count += 1
        logger.info(f"Loaded precomputed descriptions for {count} samples from {path}")

    def get(self, text: str, labels: List[str], split_idx: Optional[int] = None) -> Dict[str, str]:
        """
        Return {label: description} for the given text and label set.
        Labels not resolved get empty string description.
        """
        if self.mode == "none":
            return {l: "" for l in labels}

        if self.mode == "cache":
            return {l: self._cache.get(l.lower(), "") for l in labels}

        if self.mode == "precomputed":
            if split_idx is None:
                raise ValueError("split_idx required for precomputed mode")
            stored = self._precomputed.get(split_idx, {})
            return {l: stored.get(l.lower(), stored.get(l, "")) for l in labels}

        # LLM mode
        label_set_str = "\n".join(labels)
        prompt = CLOSED_TEMPLATE.format(TEXT=text[:2000], LABEL_SET=label_set_str)
        messages = [{"role": "user", "content": prompt}]
        input_text = self._llm_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self._llm_tokenizer(input_text, return_tensors="pt").to(
            self._llm_model.device
        )
        with torch.no_grad():
            output_ids = self._llm_model.generate(
                **inputs,
                max_new_tokens=self.cfg.llm_max_new_tokens,
                do_sample=False,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        raw = self._llm_tokenizer.decode(new_tokens, skip_special_tokens=True)

        parsed = _parse_llm_output(raw)
        # Align parsed keys (LLM might alter casing) with original labels
        result = {}
        label_lower_map = {l.lower(): l for l in labels}
        for k, v in parsed.items():
            if k in label_lower_map:
                result[label_lower_map[k]] = v
        # Fill missing labels with empty description
        for l in labels:
            if l not in result:
                result[l] = ""
        return result


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class SubsetMetrics:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    n_samples: int = 0

    def precision(self) -> float:
        return self.tp / max(self.tp + self.fp, 1)

    def recall(self) -> float:
        return self.tp / max(self.tp + self.fn, 1)

    def f1(self) -> float:
        p, r = self.precision(), self.recall()
        return 2 * p * r / max(p + r, 1e-8)

    def __repr__(self) -> str:
        return (f"P={self.precision():.4f} R={self.recall():.4f} "
                f"F1={self.f1():.4f} (n={self.n_samples})")


# ---------------------------------------------------------------------------
# GLiNER inference
# ---------------------------------------------------------------------------

def load_gliner(cfg: EvalConfig, device: torch.device) -> nn.Module:
    """Load a trained GLiNER checkpoint."""
    from train import build_model, TrainConfig

    ckpt = torch.load(cfg.gliner_ckpt, map_location="cpu")
    saved_cfg_dict = ckpt.get("config", {})

    # Build a TrainConfig from checkpoint config, overriding a few fields
    train_cfg = TrainConfig()
    for k, v in saved_cfg_dict.items():
        if hasattr(train_cfg, k):
            setattr(train_cfg, k, v)
    # Allow CLI override of arch / backbone
    train_cfg.model_arch = cfg.model_arch
    train_cfg.backbone = cfg.backbone
    train_cfg.max_span_width = cfg.max_span_width
    train_cfg.label_chunk_size = cfg.label_chunk_size

    model = build_model(train_cfg)
    model.load_state_dict(ckpt["model_state_dict"])

    # Move to device in bfloat16 if supported
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        model.to(device=device, dtype=torch.bfloat16)
    else:
        model.to(device=device, dtype=torch.float32)
    model.eval()
    return model


def encode_sample(
    text: str,
    label_descs: Dict[str, str],
    tokenizer,
    max_text_len: int,
    max_label_len: int,
) -> Optional[Dict]:
    """Tokenize one sample; returns None if no labels."""
    labels = list(label_descs.keys())
    if not labels:
        return None

    text_enc = tokenizer(
        text,
        max_length=max_text_len,
        truncation=True,
        padding=False,
        return_offsets_mapping=True,
    )
    label_texts = [
        f"{l}: {d}" if d else l
        for l, d in label_descs.items()
    ]
    label_encs = [
        tokenizer(lt, max_length=max_label_len, truncation=True, padding=False)
        for lt in label_texts
    ]
    return {
        "text_input_ids": text_enc["input_ids"],
        "text_attention_mask": text_enc["attention_mask"],
        "offset_mapping": text_enc["offset_mapping"],
        "label_input_ids": [e["input_ids"] for e in label_encs],
        "label_attention_mask": [e["attention_mask"] for e in label_encs],
        "labels": labels,
    }


def collate_for_eval(samples: List[Dict], tokenizer) -> Tuple[object, List]:
    """Collate encoded samples into a SpanBatch (M = max M in batch)."""
    from ldggliner.model import SpanBatch, SpanEnumerator

    pad_id = tokenizer.pad_token_id or 0
    cls_id = tokenizer.cls_token_id or 0
    sep_id = tokenizer.sep_token_id or 0
    dummy_ids = [cls_id, sep_id]
    dummy_mask = [1, 1]

    B = len(samples)
    M = max(len(s["labels"]) for s in samples)

    # Pad text
    max_N = max(len(s["text_input_ids"]) for s in samples)
    text_ids = torch.full((B, max_N), pad_id, dtype=torch.long)
    text_mask = torch.zeros(B, max_N, dtype=torch.long)
    for i, s in enumerate(samples):
        n = len(s["text_input_ids"])
        text_ids[i, :n] = torch.tensor(s["text_input_ids"])
        text_mask[i, :n] = torch.tensor(s["text_attention_mask"])

    # Pad labels
    all_lids, all_lmasks = [], []
    for s in samples:
        lids = list(s["label_input_ids"])
        lmasks = list(s["label_attention_mask"])
        while len(lids) < M:
            lids.append(dummy_ids)
            lmasks.append(dummy_mask)
        all_lids.extend(lids)
        all_lmasks.extend(lmasks)

    max_L = max(len(ids) for ids in all_lids)
    label_ids_t = torch.full((B * M, max_L), pad_id, dtype=torch.long)
    label_masks_t = torch.zeros(B * M, max_L, dtype=torch.long)
    for i, (ids, masks) in enumerate(zip(all_lids, all_lmasks)):
        l = len(ids)
        label_ids_t[i, :l] = torch.tensor(ids)
        label_masks_t[i, :l] = torch.tensor(masks)

    batch = SpanBatch(
        text_input_ids=text_ids,
        text_attention_mask=text_mask,
        label_input_ids=label_ids_t,
        label_attention_mask=label_masks_t,
        num_labels=M,
        span_targets=None,
    )
    return batch, [s["offset_mapping"] for s in samples]


def normalize_gold_spans(
    raw_gold: List[Tuple[int, int, str]],
    offset_mapping: List[Tuple[int, int]],
    max_span_width: int,
) -> Set[Tuple[int, int, str]]:
    """
    Run gold char spans through the same token encode-decode pipeline as predictions.

    mDeBERTa (SentencePiece) tokens include a leading space in their offset, e.g.
    '▁Dr' → offset (6, 8) even though 'Dr' starts at char 7.  decode_predictions
    uses offset_map[tok][0] as char_start, so gold must go through the same
    round-trip to stay in the same coordinate space.
    """
    from ldggliner.data_processor import NERSpanDataset
    result: Set[Tuple[int, int, str]] = set()
    for cs, ce, etype in raw_gold:
        tok_s, tok_e = NERSpanDataset._char_to_token_span(offset_mapping, cs, ce)
        if tok_s is None or tok_e is None:
            continue
        if tok_e - tok_s + 1 > max_span_width:
            continue
        if tok_e >= len(offset_mapping):
            continue
        norm_cs = offset_mapping[tok_s][0]
        norm_ce = offset_mapping[tok_e][1]
        if norm_cs >= norm_ce:
            continue
        result.add((norm_cs, norm_ce, etype))
    return result


def decode_predictions(
    logits: torch.Tensor,           # (B, S, M)
    span_mask: torch.Tensor,        # (B, S)
    start_idx: torch.Tensor,        # (S,)
    end_idx: torch.Tensor,          # (S,)
    offset_mappings: List[List[Tuple[int, int]]],
    label_lists: List[List[str]],
    threshold: float,
) -> List[Set[Tuple[int, int, str]]]:
    """
    Decode model outputs into sets of (char_start, char_end, label) predictions.
    Returns one set per sample in the batch.
    """
    probs = torch.sigmoid(logits)   # (B, S, M)
    results = []

    for b in range(logits.size(0)):
        offset_map = offset_mappings[b]
        labels = label_lists[b]
        M_b = len(labels)
        preds: Set[Tuple[int, int, str]] = set()

        # Indices of valid spans for this sample
        valid_s = span_mask[b].nonzero(as_tuple=False).squeeze(-1)  # variable length

        for s_idx in valid_s.tolist():
            for m in range(M_b):
                if probs[b, s_idx, m].item() > threshold:
                    tok_s = int(start_idx[s_idx])
                    tok_e = int(end_idx[s_idx])
                    # Guard against out-of-range (truncated text)
                    if tok_e >= len(offset_map):
                        continue
                    cs, _ = offset_map[tok_s]
                    _, ce = offset_map[tok_e]
                    if cs >= ce:
                        continue
                    preds.add((cs, ce, labels[m]))
        results.append(preds)

    return results


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(cfg: EvalConfig):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )
    os.makedirs(cfg.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ---- Load test data ----
    logger.info(f"Loading test data from {cfg.test_data}")
    from datasets import load_from_disk
    ds = load_from_disk(cfg.test_data)
    test_split = ds["test"]
    logger.info(f"Test samples total: {len(test_split)}")

    # Group by subset (dataset field)
    subset_indices: Dict[str, List[int]] = defaultdict(list)
    for i, sample in enumerate(test_split):
        subset_indices[sample["dataset"]].append(i)

    subsets = sorted(subset_indices.keys())
    logger.info(f"Available subsets ({len(subsets)}): {subsets}")

    # Filter to requested subsets (if specified)
    if cfg.subsets:
        unknown = set(cfg.subsets) - set(subsets)
        if unknown:
            logger.warning(f"Requested subsets not found in data (ignored): {sorted(unknown)}")
        subsets = [s for s in subsets if s in cfg.subsets]
        logger.info(f"Evaluating subsets ({len(subsets)}): {subsets}")

    # Apply per-subset cap
    if cfg.max_samples_per_subset:
        for k in subset_indices:
            subset_indices[k] = subset_indices[k][: cfg.max_samples_per_subset]

    # ---- Load tokenizer ----
    logger.info(f"Loading tokenizer from {cfg.backbone}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.backbone)

    # ---- Load description provider ----
    logger.info(f"Description mode: {cfg.desc_mode}")
    desc_provider = DescriptionProvider(cfg)

    # ---- Load GLiNER model ----
    logger.info(f"Loading GLiNER checkpoint: {cfg.gliner_ckpt}")
    model = load_gliner(cfg, device)
    logger.info("GLiNER model loaded.")

    # ---- Determine AMP dtype ----
    amp_dtype = None
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16

    # ---- Evaluate per subset ----
    subset_metrics: Dict[str, SubsetMetrics] = {s: SubsetMetrics() for s in subsets}
    overall = SubsetMetrics()

    pred_file = None
    if cfg.save_predictions:
        os.makedirs(cfg.output_dir, exist_ok=True)
        pred_path = os.path.join(cfg.output_dir, "predictions.jsonl")
        pred_file = open(pred_path, "w", encoding="utf-8")
        logger.info(f"Saving per-sample predictions to {pred_path}")

    for subset_name in subsets:
        indices = subset_indices[subset_name]
        sm = subset_metrics[subset_name]
        n_batches = (len(indices) + cfg.batch_size - 1) // cfg.batch_size

        pbar = tqdm(
            range(0, len(indices), cfg.batch_size),
            total=n_batches,
            desc=f"{subset_name}",
            unit="batch",
            dynamic_ncols=True,
        )

        # Process in batches
        for batch_start in pbar:
            batch_indices = indices[batch_start: batch_start + cfg.batch_size]
            raw_samples = [test_split[i] for i in batch_indices]

            encoded_batch = []
            label_lists = []
            gold_sets: List[Set[Tuple[int, int, str]]] = []
            desc_sets: List[Dict[str, str]] = []   # label -> description per sample

            for split_idx, raw in zip(batch_indices, raw_samples):
                text = raw["sentence"]

                # Gold entities (raw char positions from annotation)
                raw_gold: List[Tuple[int, int, str]] = []
                for ent in raw.get("entities", []):
                    if len(ent["pos"]) >= 2:
                        raw_gold.append((ent["pos"][0], ent["pos"][1], ent["type"].lower()))

                # Build label set from the actual entity types in this sample.
                # Using entities[].type (not the "types" field) ensures the model
                # always sees the same label names as the gold annotations.
                types = list(dict.fromkeys(t for _, _, t in raw_gold))  # unique, order-preserving
                if not types:
                    continue

                # Get descriptions
                label_descs = desc_provider.get(text, types, split_idx=split_idx)

                # Encode
                enc = encode_sample(text, label_descs, tokenizer,
                                    cfg.max_text_len, cfg.max_label_len)
                if enc is None:
                    continue

                # Normalize gold through the same token→char pipeline as predictions
                # so that SentencePiece prefix-space offsets don't cause mismatches.
                gold = normalize_gold_spans(raw_gold, enc["offset_mapping"], cfg.max_span_width)

                encoded_batch.append(enc)
                label_lists.append(enc["labels"])
                gold_sets.append(gold)
                desc_sets.append(label_descs)

            if not encoded_batch:
                continue

            # Collate and run model
            batch, offset_mappings = collate_for_eval(encoded_batch, tokenizer)

            # Move to device
            from ldggliner.model import SpanBatch
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

            # Decode predictions
            pred_sets = decode_predictions(
                logits=out["logits"].cpu(),
                span_mask=out["span_mask"].cpu(),
                start_idx=out["start_idx"].cpu(),
                end_idx=out["end_idx"].cpu(),
                offset_mappings=offset_mappings,
                label_lists=label_lists,
                threshold=cfg.threshold,
            )

            # Update metrics
            for (s_idx, raw_s), preds, gold, label_descs in zip(
                zip(batch_indices, raw_samples), pred_sets, gold_sets, desc_sets
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
                        return [
                            {"char_start": cs, "char_end": ce, "type": t}
                            for cs, ce, t in sorted(span_set)
                        ]
                    sentence = raw_s.get("sentence", "")
                    pred_file.write(json.dumps({
                        "split_idx": s_idx,
                        "dataset": subset_name,
                        "sentence": sentence,
                        "label_descriptions": label_descs,
                        "gold": _fmt(gold),
                        "pred": _fmt(preds),
                        "tp": _fmt(preds & gold),
                        "fp": _fmt(preds - gold),
                        "fn": _fmt(gold - preds),
                    }, ensure_ascii=False) + "\n")

            pbar.set_postfix(P=f"{sm.precision():.3f}", R=f"{sm.recall():.3f}", F1=f"{sm.f1():.3f}")

        pbar.close()
        logger.info(f"  {subset_name}: {sm}")

    # ---- Print final results ----
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
            "precision": sm.precision(),
            "recall": sm.recall(),
            "f1": sm.f1(),
            "n_samples": sm.n_samples,
            "tp": sm.tp, "fp": sm.fp, "fn": sm.fn,
        })
    print("-" * 70)
    print(f"{'OVERALL':<35} {overall.precision():>7.4f} {overall.recall():>7.4f} "
          f"{overall.f1():>7.4f} {overall.n_samples:>8}")
    print("=" * 70)

    if pred_file is not None:
        pred_file.close()

    # ---- Save results ----
    results = {
        "config": vars(cfg),
        "overall": {
            "precision": overall.precision(),
            "recall": overall.recall(),
            "f1": overall.f1(),
            "n_samples": overall.n_samples,
            "tp": overall.tp, "fp": overall.fp, "fn": overall.fn,
        },
        "subsets": rows,
    }
    out_path = os.path.join(cfg.output_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    cfg = parse_args()
    run_evaluation(cfg)
