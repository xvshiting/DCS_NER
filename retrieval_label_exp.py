#!/usr/bin/env python3
"""
retrieval_label_exp.py
======================
Retrieval-augmented label description experiment for LDG-GLiNER.

Instead of encoding a test sample's own LLM description, this script:
  1. Retrieves the k most similar corpus entries (training samples with precomputed
     descriptions) by BM25 or dense (text-encoder) similarity.
  2. For each target label, collects description embeddings from retrieved entries
     that have a description for that label (others are silently skipped).
  3. Averages those embeddings → injected directly as Q into the GLiNER model,
     bypassing encode_labels().
  4. Decodes with the same NMS logic as evaluate.py.

Stages with disk caching (--cache_dir):
  A. encode_descriptions  — CLS-pool each unique "label: desc" string
  B. encode_corpus_texts  — (dense only) CLS-pool corpus texts
  C. encode_test_texts    — (dense only) CLS-pool test texts
  D. evaluate_k           — runs per k, reads from caches above

Usage:
    python retrieval_label_exp.py \\
        --gliner_ckpt  checkpoints/run9/best/checkpoint.pt \\
        --backbone     /data/model_hub/mdeberta-v3-base \\
        --corpus_descs descriptions_cache.jsonl \\
        --corpus_split_name train \\
        --test_data    /data/dataset/ner/instruct_uie_ner \\
        --retrieval_mode bm25 \\
        --k_values 1 3 5 10 20 \\
        --output_dir   eval_results/retrieval_exp
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer

# ── project root on path ──────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from evaluate import (
    EvalConfig,
    SubsetMetrics,
    load_gliner,
    decode_predictions,
    normalize_gold_spans,
)
from ldggliner.model import SpanBatch

logger = logging.getLogger(__name__)

try:
    from rank_bm25 import BM25Okapi
    _HAS_BM25 = True
except ImportError:
    _HAS_BM25 = False


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RetConfig:
    # GLiNER model
    gliner_ckpt: str = "checkpoints/run9/best/checkpoint.pt"
    backbone: str = "/data/model_hub/mdeberta-v3-base"
    model_arch: str = "deberta_span_v1"
    max_span_width: int = 12
    label_chunk_size: int = 16
    threshold: float = 0.5
    flat_ner: bool = True
    multi_label: bool = False
    max_text_len: int = 256
    max_label_len: int = 128

    # Corpus (retrieval source)
    corpus_descs: str = "descriptions_cache.jsonl"
    corpus_data: str = "/data/dataset/ner/instruct_uie_ner"
    corpus_split_name: str = "train"   # which dataset split corpus_descs indices refer to

    # Test data
    test_data: str = "/data/dataset/ner/instruct_uie_ner"
    subsets: Optional[List[str]] = None
    max_samples_per_subset: Optional[int] = None

    # Retrieval
    retrieval_mode: str = "bm25"       # bm25 | dense
    k_values: List[int] = field(default_factory=lambda: [1, 3, 5, 10, 20])
    encode_batch_size: int = 32

    # Cache & output
    cache_dir: str = ".retrieval_cache"
    force_recompute: bool = False
    output_dir: str = "eval_results/retrieval_exp"
    save_per_sample: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Model: forward with pre-computed Q
# ─────────────────────────────────────────────────────────────────────────────

def forward_with_precomputed_q(
    model,
    batch: SpanBatch,
    Q: torch.Tensor,       # (B, M, d) — already on device+dtype
) -> Dict[str, torch.Tensor]:
    """
    Run the GLiNER forward pass with a pre-computed label representation Q,
    bypassing model.encode_labels().

    Works with all three model variants:
      V1 — no fusion_stacks: Q is used directly for span scoring
      V2 — FusionStack updates both Q and H
      V3 — LabelOnlyFusionStack updates only Q; H stays fixed
    """
    B, N = batch.text_input_ids.shape
    device = batch.text_input_ids.device

    H = model.encode_text(
        batch.text_input_ids, batch.text_attention_mask
    )   # (B, N, d)

    if hasattr(model, "fusion_stacks"):
        pad_mask = batch.text_attention_mask == 0   # (B, N) True → PAD
        for stack in model.fusion_stacks:
            Q, H = stack(Q, H, h_key_padding_mask=pad_mask)

    start_idx, end_idx, width = model.span_enum.enumerate(
        seq_len=N, device=device
    )
    S = start_idx.numel()

    H_s = H.index_select(1, start_idx)   # (B, S, d)
    H_e = H.index_select(1, end_idx)     # (B, S, d)

    if model.use_width_embedding:
        W = model.width_emb(width).unsqueeze(0).expand(B, S, model.d_model)
        span_in = torch.cat([H_s, H_e, W], dim=-1)
    else:
        span_in = torch.cat([H_s, H_e], dim=-1)

    span_vec = model.span_ffn(span_in)
    logits = torch.einsum("bsd,bmd->bsm", span_vec, Q) + model.label_bias

    attn = batch.text_attention_mask.bool()
    span_mask = (
        attn.index_select(1, start_idx) & attn.index_select(1, end_idx)
    )

    return {
        "logits": logits,
        "span_mask": span_mask,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "width": width,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Corpus
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CorpusEntry:
    corpus_idx: int                    # position in corpus list
    split_idx: int                     # index in the HuggingFace dataset split
    text: str
    descriptions: Dict[str, str]       # {label_lower: desc_text}


def load_corpus(descs_path: str, dataset_split) -> List[CorpusEntry]:
    """
    Merge precomputed descriptions JSONL with texts from a HuggingFace split.
    Each JSONL line: {"split_idx": int, "descriptions": {label: desc}}
    """
    desc_by_idx: Dict[int, Dict[str, str]] = {}
    with open(descs_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            sid = int(obj["split_idx"])
            desc_by_idx[sid] = {
                k.lower(): v
                for k, v in obj.get("descriptions", {}).items()
            }

    corpus: List[CorpusEntry] = []
    n_skip = 0
    for sid in sorted(desc_by_idx):
        if sid >= len(dataset_split):
            n_skip += 1
            continue
        corpus.append(CorpusEntry(
            corpus_idx=len(corpus),
            split_idx=sid,
            text=dataset_split[sid]["sentence"],
            descriptions=desc_by_idx[sid],
        ))

    if n_skip:
        logger.warning(
            f"Skipped {n_skip} corpus entries with out-of-range split_idx "
            f"(split has {len(dataset_split)} samples)."
        )
    logger.info(f"Corpus loaded: {len(corpus)} entries from '{descs_path}'")
    return corpus


# ─────────────────────────────────────────────────────────────────────────────
# Embedding computation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cls_encode(
    encoder,
    tokenizer,
    texts: List[str],
    max_len: int,
    batch_size: int,
    device: torch.device,
    amp_dtype,
    desc: str = "Encoding",
) -> torch.Tensor:
    """
    Encode a list of strings with `encoder` (a HuggingFace model or sub-module),
    return CLS-pooled float32 embeddings (N, d) on CPU.
    """
    all_embs: List[torch.Tensor] = []
    pad_id = tokenizer.pad_token_id or 0

    for start in tqdm(range(0, len(texts), batch_size), desc=desc, leave=False):
        chunk = texts[start: start + batch_size]
        encs = [
            tokenizer(t, max_length=max_len, truncation=True, padding=False)
            for t in chunk
        ]
        L = max(len(e["input_ids"]) for e in encs)
        ids = torch.full((len(encs), L), pad_id, dtype=torch.long)
        msk = torch.zeros(len(encs), L, dtype=torch.long)
        for i, e in enumerate(encs):
            l = len(e["input_ids"])
            ids[i, :l] = torch.tensor(e["input_ids"])
            msk[i, :l] = torch.tensor(e["attention_mask"])

        with torch.no_grad():
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype,
                enabled=amp_dtype is not None,
            ):
                out = encoder(
                    input_ids=ids.to(device),
                    attention_mask=msk.to(device),
                )
        all_embs.append(out.last_hidden_state[:, 0, :].float().cpu())

    return torch.cat(all_embs, dim=0)   # (N, d)


def _load_or_compute(
    cache_path: str,
    force: bool,
    compute_fn,
    label: str,
):
    """Load tensor/dict from cache if available, else compute and save."""
    if not force and os.path.exists(cache_path):
        logger.info(f"Loading {label} from cache: {cache_path}")
        return torch.load(cache_path, map_location="cpu")
    logger.info(f"Computing {label} …")
    result = compute_fn()
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    torch.save(result, cache_path)
    logger.info(f"Cached {label} → {cache_path}")
    return result


# ── Stage A: description embeddings ──────────────────────────────────────────

def build_desc_emb_store(
    model,
    tokenizer,
    corpus: List[CorpusEntry],
    max_label_len: int,
    batch_size: int,
    device: torch.device,
    amp_dtype,
    cache_path: str,
    force: bool,
) -> Dict[str, torch.Tensor]:
    """
    Encode every unique "label: desc" string in the corpus with the label encoder.
    Returns {desc_key: tensor(d,)} where desc_key = f"{label}: {desc}" or just label.
    """
    def _compute():
        unique_keys: List[str] = []
        seen: Set[str] = set()
        for entry in corpus:
            for label, desc in entry.descriptions.items():
                key = f"{label}: {desc}" if desc else label
                if key not in seen:
                    unique_keys.append(key)
                    seen.add(key)
        logger.info(f"  {len(unique_keys)} unique description strings to encode")
        embs = _cls_encode(
            model.label_encoder, tokenizer,
            unique_keys, max_label_len, batch_size, device, amp_dtype,
            "Encoding descriptions",
        )
        return {k: embs[i] for i, k in enumerate(unique_keys)}

    return _load_or_compute(cache_path, force, _compute, "description embeddings")


# ── Stage B/C: text embeddings ────────────────────────────────────────────────

def build_text_embeddings(
    model,
    tokenizer,
    texts: List[str],
    max_text_len: int,
    batch_size: int,
    device: torch.device,
    amp_dtype,
    cache_path: str,
    force: bool,
    label: str = "text embeddings",
) -> torch.Tensor:
    """
    Encode texts with text_encoder; L2-normalise; return (N, d) float32 CPU tensor.
    """
    def _compute():
        embs = _cls_encode(
            model.text_encoder, tokenizer,
            texts, max_text_len, batch_size, device, amp_dtype,
            f"Encoding {label}",
        )
        return F.normalize(embs, dim=-1)

    return _load_or_compute(cache_path, force, _compute, label)


# ─────────────────────────────────────────────────────────────────────────────
# Retrievers
# ─────────────────────────────────────────────────────────────────────────────

class BM25Retriever:
    def __init__(self, corpus: List[CorpusEntry]):
        if not _HAS_BM25:
            raise ImportError(
                "rank_bm25 not installed.  Run: pip install rank-bm25"
            )
        self.corpus = corpus
        self.texts = [e.text for e in corpus]
        logger.info(f"Building BM25 index over {len(self.texts)} corpus texts …")
        self._bm25 = BM25Okapi([t.lower().split() for t in self.texts])
        logger.info("BM25 index ready.")

    def retrieve(
        self,
        query: str,
        k: int,
        exclude_text: Optional[str] = None,
        **_,
    ) -> List[int]:
        """Return up to k corpus indices (not split_idx) for query."""
        scores = self._bm25.get_scores(query.lower().split())
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
        return _top_k_exclude(ranked, self.texts, exclude_text, k)


class DenseRetriever:
    def __init__(self, corpus: List[CorpusEntry], corpus_embs: torch.Tensor):
        self.corpus = corpus
        self.texts = [e.text for e in corpus]
        self.embs = corpus_embs   # (N, d) L2-normalised, CPU

    def retrieve(
        self,
        query: str,
        k: int,
        exclude_text: Optional[str] = None,
        query_emb: Optional[torch.Tensor] = None,
        **_,
    ) -> List[int]:
        if query_emb is None:
            raise ValueError("DenseRetriever requires query_emb.")
        q = F.normalize(query_emb.unsqueeze(0), dim=-1)   # (1, d)
        sims = (q @ self.embs.T).squeeze(0)               # (N,)
        ranked = sims.argsort(descending=True).tolist()
        return _top_k_exclude(ranked, self.texts, exclude_text, k)


def _top_k_exclude(
    ranked: List[int],
    texts: List[str],
    exclude_text: Optional[str],
    k: int,
) -> List[int]:
    result: List[int] = []
    for ci in ranked:
        if exclude_text is not None and texts[ci] == exclude_text:
            continue
        result.append(ci)
        if len(result) >= k:
            break
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation for a single k
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_k(
    k: int,
    model,
    tokenizer,
    corpus: List[CorpusEntry],
    retriever,
    desc_emb_store: Dict[str, torch.Tensor],
    fallback_emb_cache: Dict[str, torch.Tensor],   # mutated in-place (shared)
    test_split,
    subset_indices: Dict[str, List[int]],
    subsets: List[str],
    eval_cfg: EvalConfig,
    device: torch.device,
    amp_dtype,
    test_text_embs: Optional[torch.Tensor],        # (T, d) for dense retrieval
    test_idx_to_pos: Dict[int, int],               # split_idx → row in test_text_embs
    save_per_sample: bool = False,
) -> Tuple[Dict[str, SubsetMetrics], SubsetMetrics, List[dict]]:
    """
    For each test sample:
      1. Retrieve top-k corpus entries (excluding self by text equality).
      2. Build Q: for each label, average desc embeddings from retrieved entries
         that have a description for that label; fall back to label-name encoding.
      3. Run forward_with_precomputed_q and decode.
    Returns (subset_metrics, overall, per_sample_records).
    """
    model_dtype = next(model.parameters()).dtype
    subset_metrics: Dict[str, SubsetMetrics] = {}
    overall = SubsetMetrics()
    per_sample: List[dict] = []

    pad_id = tokenizer.pad_token_id or 0

    for subset_name in subsets:
        indices = subset_indices[subset_name]
        sm = SubsetMetrics()
        subset_metrics[subset_name] = sm

        pbar = tqdm(
            indices,
            desc=f"k={k} | {subset_name}",
            unit="sample",
            dynamic_ncols=True,
        )

        for split_idx in pbar:
            raw = test_split[split_idx]
            text: str = raw["sentence"]

            # ── Gold ──────────────────────────────────────────────────────
            raw_gold: List[Tuple[int, int, str]] = [
                (e["pos"][0], e["pos"][1], e["type"].lower())
                for e in raw.get("entities", [])
                if len(e.get("pos", [])) >= 2
            ]
            types = list(dict.fromkeys(t for _, _, t in raw_gold))
            if not types:
                continue

            # ── Tokenise text ─────────────────────────────────────────────
            text_enc = tokenizer(
                text,
                max_length=eval_cfg.max_text_len,
                truncation=True,
                padding=False,
                return_offsets_mapping=True,
            )
            offset_map = text_enc["offset_mapping"]
            gold = normalize_gold_spans(raw_gold, offset_map, eval_cfg.max_span_width)

            # ── Retrieve ──────────────────────────────────────────────────
            q_emb = (
                test_text_embs[test_idx_to_pos[split_idx]]
                if (test_text_embs is not None and split_idx in test_idx_to_pos)
                else None
            )
            top_k_ci = retriever.retrieve(
                text, k=k, exclude_text=text, query_emb=q_emb
            )
            retrieved = [corpus[ci] for ci in top_k_ci]

            # ── Build Q ───────────────────────────────────────────────────
            label_vecs: List[torch.Tensor] = []
            n_hit = 0   # labels that found ≥1 retrieved description

            for label in types:
                embs: List[torch.Tensor] = []
                for entry in retrieved:
                    desc = entry.descriptions.get(label, "")
                    if not desc:
                        continue
                    key = f"{label}: {desc}"
                    if key in desc_emb_store:
                        embs.append(desc_emb_store[key])

                if embs:
                    label_vecs.append(torch.stack(embs).mean(0))  # (d,)
                    n_hit += 1
                else:
                    # Fallback: encode label name alone (cached)
                    if label not in fallback_emb_cache:
                        enc = tokenizer(
                            label,
                            max_length=eval_cfg.max_label_len,
                            truncation=True,
                            padding=False,
                        )
                        ids_t = torch.tensor(enc["input_ids"]).unsqueeze(0).to(device)
                        msk_t = torch.tensor(enc["attention_mask"]).unsqueeze(0).to(device)
                        with torch.no_grad():
                            out = model.label_encoder(
                                input_ids=ids_t, attention_mask=msk_t
                            )
                        fallback_emb_cache[label] = (
                            out.last_hidden_state[0, 0, :].float().cpu()
                        )
                    label_vecs.append(fallback_emb_cache[label])

            M = len(types)
            Q = (
                torch.stack(label_vecs, dim=0)
                .unsqueeze(0)
                .to(device=device, dtype=model_dtype)
            )   # (1, M, d)

            # ── SpanBatch (label encoder is bypassed) ────────────────────
            text_ids = torch.tensor(text_enc["input_ids"]).unsqueeze(0).to(device)
            text_msk = torch.tensor(text_enc["attention_mask"]).unsqueeze(0).to(device)
            batch = SpanBatch(
                text_input_ids=text_ids,
                text_attention_mask=text_msk,
                # label_input_ids/mask are placeholders — not used in forward
                label_input_ids=torch.zeros(M, 2, dtype=torch.long, device=device),
                label_attention_mask=torch.ones(M, 2, dtype=torch.long, device=device),
                num_labels=M,
                span_targets=None,
            )

            # ── Forward ───────────────────────────────────────────────────
            with torch.no_grad():
                with torch.autocast(
                    device_type=device.type, dtype=amp_dtype,
                    enabled=amp_dtype is not None,
                ):
                    out = forward_with_precomputed_q(model, batch, Q)

            # ── Decode ────────────────────────────────────────────────────
            pred_sets = decode_predictions(
                logits=out["logits"].cpu(),
                span_mask=out["span_mask"].cpu(),
                start_idx=out["start_idx"].cpu(),
                end_idx=out["end_idx"].cpu(),
                offset_mappings=[offset_map],
                label_lists=[types],
                threshold=eval_cfg.threshold,
                flat_ner=eval_cfg.flat_ner,
                multi_label=eval_cfg.multi_label,
            )
            preds = pred_sets[0]

            # ── Metrics ───────────────────────────────────────────────────
            sm.tp += len(preds & gold)
            sm.fp += len(preds - gold)
            sm.fn += len(gold - preds)
            sm.n_samples += 1
            overall.tp += len(preds & gold)
            overall.fp += len(preds - gold)
            overall.fn += len(gold - preds)
            overall.n_samples += 1

            if save_per_sample:
                def _fmt(s):
                    return [{"cs": cs, "ce": ce, "type": t} for cs, ce, t in sorted(s)]
                per_sample.append({
                    "split_idx": split_idx,
                    "subset": subset_name,
                    "text": text,
                    "labels": types,
                    "n_retrieved": len(retrieved),
                    "n_label_hits": n_hit,
                    "gold": _fmt(gold),
                    "pred": _fmt(preds),
                    "tp": _fmt(preds & gold),
                    "fp": _fmt(preds - gold),
                    "fn": _fmt(gold - preds),
                })

            pbar.set_postfix(
                P=f"{sm.precision():.3f}",
                R=f"{sm.recall():.3f}",
                F1=f"{sm.f1():.3f}",
                hit=f"{n_hit}/{M}",
            )

        pbar.close()
        logger.info(f"  k={k} | {subset_name}: {sm}")

    return subset_metrics, overall, per_sample


def macro_f1(subset_metrics: Dict[str, SubsetMetrics]) -> float:
    """Unweighted mean of per-subset F1 scores (only subsets with ≥1 sample)."""
    f1s = [sm.f1() for sm in subset_metrics.values() if sm.n_samples > 0]
    return sum(f1s) / len(f1s) if f1s else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> RetConfig:
    p = argparse.ArgumentParser(
        description="Retrieval-augmented label description experiment for LDG-GLiNER",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    d = RetConfig()

    g = p.add_argument_group("GLiNER model")
    g.add_argument("--gliner_ckpt", default=d.gliner_ckpt)
    g.add_argument("--backbone", default=d.backbone)
    g.add_argument("--model_arch", default=d.model_arch,
                   choices=["deberta_span_v1", "deberta_span_v2", "deberta_span_v3"])
    g.add_argument("--max_span_width", type=int, default=d.max_span_width)
    g.add_argument("--label_chunk_size", type=int, default=d.label_chunk_size)
    g.add_argument("--threshold", type=float, default=d.threshold)
    g.add_argument("--flat_ner", action=argparse.BooleanOptionalAction, default=d.flat_ner,
                   help="Greedy NMS to suppress overlapping spans")
    g.add_argument("--multi_label", action=argparse.BooleanOptionalAction,
                   default=d.multi_label,
                   help="Global NMS across all labels (vs per-label NMS)")
    g.add_argument("--max_text_len", type=int, default=d.max_text_len)
    g.add_argument("--max_label_len", type=int, default=d.max_label_len)

    g = p.add_argument_group("Corpus (retrieval source)")
    g.add_argument("--corpus_descs", default=d.corpus_descs,
                   help="JSONL: {split_idx, descriptions:{label:desc}}")
    g.add_argument("--corpus_data", default=d.corpus_data,
                   help="HuggingFace Arrow dataset dir containing corpus texts")
    g.add_argument("--corpus_split_name", default=d.corpus_split_name,
                   help="Split name in corpus_data whose indices corpus_descs uses")

    g = p.add_argument_group("Test data")
    g.add_argument("--test_data", default=d.test_data,
                   help="HuggingFace Arrow dataset dir; 'test' split is used")
    g.add_argument("--subsets", nargs="+", default=None, metavar="SUBSET",
                   help="Subset name(s) to evaluate; default: all")
    g.add_argument("--max_samples_per_subset", type=int, default=d.max_samples_per_subset)

    g = p.add_argument_group("Retrieval")
    g.add_argument("--retrieval_mode", default=d.retrieval_mode,
                   choices=["bm25", "dense"],
                   help="bm25: tokenise & BM25Okapi  |  dense: text-encoder cosine")
    g.add_argument("--k_values", nargs="+", type=int,
                   default=d.k_values,
                   help="k values to evaluate (e.g. --k_values 1 3 5 10 20)")
    g.add_argument("--encode_batch_size", type=int, default=d.encode_batch_size,
                   help="Batch size for embedding computation")

    g = p.add_argument_group("Cache & output")
    g.add_argument("--cache_dir", default=d.cache_dir,
                   help="Directory to store cached embeddings")
    g.add_argument("--force_recompute", action="store_true",
                   help="Ignore existing caches and recompute everything")
    g.add_argument("--output_dir", default=d.output_dir)
    g.add_argument("--save_per_sample", action="store_true",
                   help="Write per-sample predictions to per_sample_k{k}.jsonl")

    args = p.parse_args()
    cfg_fields = {f.name for f in RetConfig.__dataclass_fields__.values()}
    return RetConfig(**{k: v for k, v in vars(args).items() if k in cfg_fields})


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    cfg = parse_args()

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.cache_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else None
    )
    logger.info(f"Device: {device}  amp_dtype: {amp_dtype}")
    logger.info(f"Retrieval mode : {cfg.retrieval_mode}")
    logger.info(f"k values       : {cfg.k_values}")

    # ── Load datasets ─────────────────────────────────────────────────────────
    from datasets import load_from_disk

    logger.info(f"Loading corpus dataset: {cfg.corpus_data}/{cfg.corpus_split_name}")
    corpus_ds = load_from_disk(cfg.corpus_data)
    corpus_split = corpus_ds[cfg.corpus_split_name]

    if cfg.test_data == cfg.corpus_data:
        test_split = corpus_ds["test"]
    else:
        logger.info(f"Loading test dataset: {cfg.test_data}")
        test_split = load_from_disk(cfg.test_data)["test"]

    # ── Load tokenizer & GLiNER model ────────────────────────────────────────
    logger.info(f"Loading tokenizer: {cfg.backbone}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.backbone)

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
    model.eval()

    # ── Load corpus ───────────────────────────────────────────────────────────
    corpus = load_corpus(cfg.corpus_descs, corpus_split)
    if not corpus:
        raise RuntimeError(
            "Corpus is empty — check --corpus_descs and --corpus_split_name."
        )

    # ── Stage A: description embeddings ──────────────────────────────────────
    desc_embs_path = os.path.join(cfg.cache_dir, "desc_embs.pt")
    desc_emb_store = build_desc_emb_store(
        model, tokenizer, corpus,
        cfg.max_label_len, cfg.encode_batch_size, device, amp_dtype,
        cache_path=desc_embs_path, force=cfg.force_recompute,
    )
    logger.info(f"Description embedding store: {len(desc_emb_store)} entries")

    # ── Stage B/C: build retriever ────────────────────────────────────────────
    test_text_embs: Optional[torch.Tensor] = None
    test_idx_to_pos: Dict[int, int] = {}

    if cfg.retrieval_mode == "bm25":
        retriever = BM25Retriever(corpus)

    elif cfg.retrieval_mode == "dense":
        # Corpus text embeddings (Stage B)
        corpus_emb_path = os.path.join(cfg.cache_dir, "corpus_text_embs.pt")
        corpus_text_embs = build_text_embeddings(
            model, tokenizer,
            [e.text for e in corpus],
            cfg.max_text_len, cfg.encode_batch_size, device, amp_dtype,
            cache_path=corpus_emb_path, force=cfg.force_recompute,
            label="corpus text embeddings",
        )
        retriever = DenseRetriever(corpus, corpus_text_embs)

        # Test text embeddings (Stage C)
        # Collect all test indices across all subsets first
        all_si: Dict[str, List[int]] = defaultdict(list)
        for i, s in enumerate(test_split):
            all_si[s["dataset"]].append(i)
        all_test_idxs: List[int] = []
        for name in sorted(all_si):
            all_test_idxs.extend(all_si[name])

        test_idx_to_pos = {sid: pos for pos, sid in enumerate(all_test_idxs)}

        test_emb_path = os.path.join(cfg.cache_dir, "test_text_embs.pt")
        test_text_embs = build_text_embeddings(
            model, tokenizer,
            [test_split[i]["sentence"] for i in all_test_idxs],
            cfg.max_text_len, cfg.encode_batch_size, device, amp_dtype,
            cache_path=test_emb_path, force=cfg.force_recompute,
            label="test text embeddings",
        )

    else:
        raise ValueError(f"Unknown retrieval_mode: {cfg.retrieval_mode}")

    # ── Prepare test subset indices ───────────────────────────────────────────
    subset_indices: Dict[str, List[int]] = defaultdict(list)
    for i, s in enumerate(test_split):
        subset_indices[s["dataset"]].append(i)

    subsets = sorted(subset_indices.keys())
    logger.info(f"Test subsets ({len(subsets)}): {subsets}")

    if cfg.subsets:
        unknown = set(cfg.subsets) - set(subsets)
        if unknown:
            logger.warning(f"Unknown subsets (ignored): {sorted(unknown)}")
        subsets = [s for s in subsets if s in set(cfg.subsets)]
        logger.info(f"Evaluating ({len(subsets)}): {subsets}")

    if cfg.max_samples_per_subset:
        for name in subset_indices:
            subset_indices[name] = subset_indices[name][: cfg.max_samples_per_subset]

    # ── Shared fallback embedding cache (reused across all k values) ──────────
    fallback_emb_cache: Dict[str, torch.Tensor] = {}

    # ── Evaluate each k ───────────────────────────────────────────────────────
    all_results: Dict[int, dict] = {}

    for k in sorted(cfg.k_values):
        logger.info(f"\n{'=' * 64}")
        logger.info(f"Evaluating k = {k}  [{cfg.retrieval_mode}]")
        logger.info(f"{'=' * 64}")

        sm_dict, overall, per_sample = evaluate_k(
            k=k,
            model=model,
            tokenizer=tokenizer,
            corpus=corpus,
            retriever=retriever,
            desc_emb_store=desc_emb_store,
            fallback_emb_cache=fallback_emb_cache,
            test_split=test_split,
            subset_indices=subset_indices,
            subsets=subsets,
            eval_cfg=eval_cfg,
            device=device,
            amp_dtype=amp_dtype,
            test_text_embs=test_text_embs,
            test_idx_to_pos=test_idx_to_pos,
            save_per_sample=cfg.save_per_sample,
        )

        mf1 = macro_f1(sm_dict)

        # ── Print per-subset table ────────────────────────────────────────
        print(f"\n{'=' * 70}")
        print(f"k = {k}   [{cfg.retrieval_mode}]")
        print(f"{'Subset':<35} {'P':>7} {'R':>7} {'F1':>7} {'N':>8}")
        print(f"{'-' * 70}")
        rows = []
        for name in sorted(sm_dict):
            sm = sm_dict[name]
            if sm.n_samples == 0:
                continue
            print(
                f"{name:<35} {sm.precision():>7.4f} {sm.recall():>7.4f} "
                f"{sm.f1():>7.4f} {sm.n_samples:>8}"
            )
            rows.append({
                "subset": name,
                "precision": sm.precision(), "recall": sm.recall(), "f1": sm.f1(),
                "n_samples": sm.n_samples,
                "tp": sm.tp, "fp": sm.fp, "fn": sm.fn,
            })
        print(f"{'-' * 70}")
        print(
            f"{'MICRO (overall)':<35} {overall.precision():>7.4f} "
            f"{overall.recall():>7.4f} {overall.f1():>7.4f} {overall.n_samples:>8}"
        )
        print(f"{'MACRO F1':<35} {mf1:>7.4f}")
        print(f"{'=' * 70}\n")

        all_results[k] = {
            "k": k,
            "retrieval_mode": cfg.retrieval_mode,
            "subsets": rows,
            "micro": {
                "precision": overall.precision(),
                "recall": overall.recall(),
                "f1": overall.f1(),
                "n_samples": overall.n_samples,
                "tp": overall.tp, "fp": overall.fp, "fn": overall.fn,
            },
            "macro_f1": mf1,
        }

        if cfg.save_per_sample and per_sample:
            ps_path = os.path.join(cfg.output_dir, f"per_sample_k{k}.jsonl")
            with open(ps_path, "w", encoding="utf-8") as f:
                for rec in per_sample:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            logger.info(f"Per-sample predictions → {ps_path}")

    # ── Cross-k summary ───────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"Summary  —  {cfg.retrieval_mode} retrieval")
    print(f"{'k':>5} {'Micro-P':>9} {'Micro-R':>9} {'Micro-F1':>10} {'Macro-F1':>10}")
    print(f"{'-' * 70}")
    for k in sorted(all_results):
        r = all_results[k]
        m = r["micro"]
        print(
            f"{k:>5}  {m['precision']:>8.4f}  {m['recall']:>8.4f}  "
            f"{m['f1']:>9.4f}  {r['macro_f1']:>9.4f}"
        )
    print(f"{'=' * 70}\n")

    # ── Save JSON results ─────────────────────────────────────────────────────
    results_path = os.path.join(cfg.output_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(
            {"config": {k: v for k, v in vars(cfg).items()},
             "results_by_k": {str(k): v for k, v in all_results.items()}},
            f, indent=2, ensure_ascii=False,
        )
    logger.info(f"Results saved → {results_path}")


if __name__ == "__main__":
    main()
