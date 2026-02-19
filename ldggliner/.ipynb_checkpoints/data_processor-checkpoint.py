"""
Data processing for LDG-GLiNER training.

Dataset loads from JSONL files with the following format:
{
    "sentence": "...",
    "entities": [{"name": "...", "pos": [start_char, end_char], "type": "..."}],
    "types": ["type1", "type2", ...],
    "description": [
        {"label": "type1", "definitions": {"D1": "...", "D2": "...", ..., "D6": "..."}}
    ]
}

Label set per sample:
  - ALL positive labels (entity types that appear in the sample and have descriptions)
  - Negative labels sampled from the global pool to fill up to max_total_labels
  - M is variable per batch: pad to the largest M in the current batch
"""

from __future__ import annotations

import json
import os
import random
from functools import partial
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from ldggliner.model import SpanBatch, SpanEnumerator


class NERSpanDataset(Dataset):
    """
    Loads NER data from JSONL files with LLM-generated label descriptions.

    Label set per sample:
      - ALL positive labels (entity types present in the sample that have descriptions)
      - Random negative labels sampled from the global label pool to fill up to
        max_total_labels (a compute/memory budget, not a data constraint)
    """

    def __init__(
        self,
        data_dir: str,
        tokenizer,
        max_text_len: int = 256,
        max_label_len: int = 128,
        max_span_width: int = 12,
        max_total_labels: int = 25,
        desc_key: Optional[str] = None,
        file_prefix: str = "instruct_uie_ner_converted_description_",
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            data_dir: Directory containing JSONL files.
            tokenizer: HuggingFace tokenizer.
            max_text_len: Max tokens for text sequences.
            max_label_len: Max tokens for label+description sequences.
            max_span_width: Max span width in tokens (must match model config).
            max_total_labels: Budget cap on M per sample (positive + negative).
                              ALL positive labels are kept; negatives fill remaining slots.
                              This is a memory/compute constraint, not a data constraint.
            desc_key: If set (e.g. "D1"), always use this description key.
                      If None, randomly sample from D1-D6 each time (augmentation).
            file_prefix: Filename prefix to filter JSONL files.
            max_samples: Limit dataset size (useful for debugging).
        """
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.max_label_len = max_label_len
        self.max_span_width = max_span_width
        self.max_total_labels = max_total_labels
        self.desc_key = desc_key

        self.data = self._load_data(data_dir, file_prefix, max_samples)
        self._build_label_pool()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_data(
        self,
        data_dir: str,
        file_prefix: str,
        max_samples: Optional[int],
    ) -> List[Dict]:
        filenames = sorted(
            f for f in os.listdir(data_dir)
            if f.startswith(file_prefix) and f.endswith(".jsonl")
        )
        if not filenames:
            raise FileNotFoundError(
                f"No JSONL files with prefix '{file_prefix}' found in {data_dir}"
            )

        data = []
        for fname in filenames:
            fpath = os.path.join(data_dir, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if obj.get("description") and obj.get("types"):
                        data.append(obj)
                    if max_samples and len(data) >= max_samples:
                        return data
        return data

    def _parse_desc_map(self, sample: Dict) -> Dict[str, Dict[str, str]]:
        """Return {label: {D1: ..., D2: ..., ...}} from either list or dict format."""
        desc = sample.get("description", {})
        if isinstance(desc, list):
            return {
                d["label"]: d["definitions"]
                for d in desc
                if "label" in d and "definitions" in d
            }
        elif isinstance(desc, dict):
            return desc
        return {}

    def _build_label_pool(self):
        """Collect all labels and their definitions seen across the dataset."""
        pool: Dict[str, Dict[str, str]] = {}
        for sample in self.data:
            desc_map = self._parse_desc_map(sample)
            pool.update(desc_map)
        self.label_pool: Dict[str, Dict[str, str]] = pool
        self.all_labels: List[str] = list(pool.keys())

    # ------------------------------------------------------------------
    # Per-sample processing
    # ------------------------------------------------------------------

    def _select_desc(self, definitions: Dict[str, str]) -> str:
        """Randomly pick one description from D1-D6 (training-time augmentation)."""
        if not definitions:
            return ""
        if self.desc_key and self.desc_key in definitions:
            return definitions[self.desc_key]
        return definitions[random.choice(list(definitions.keys()))]

    @staticmethod
    def _char_to_token_span(
        offset_mapping: List[Tuple[int, int]],
        char_start: int,
        char_end: int,
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Map character span [char_start, char_end) to inclusive token span [tok_start, tok_end].
        Returns (None, None) if no overlapping token is found.
        """
        tok_start = None
        tok_end = None
        for i, (cs, ce) in enumerate(offset_mapping):
            if cs >= ce:
                continue  # special / empty token ([CLS], [SEP], padding)
            if ce > char_start and cs < char_end:
                if tok_start is None:
                    tok_start = i
                tok_end = i
        return tok_start, tok_end

    def _process_sample(self, sample: Dict) -> Dict:
        text = sample.get("sentence", sample.get("text", ""))
        entities = sample.get("entities", [])
        types = sample.get("types", [])
        desc_map = self._parse_desc_map(sample)

        # ---- Label set construction ----
        # Step 1: ALL positive labels (entity types in this sample that have descriptions).
        #         Never truncate positive labels.
        pos_labels = [t for t in types if t in desc_map]

        # Step 2: Fill remaining budget slots with randomly sampled negative labels.
        #         Negatives are labels from the global pool that do NOT appear in this sample.
        n_neg = max(0, self.max_total_labels - len(pos_labels))
        if n_neg > 0:
            neg_candidates = [l for l in self.all_labels if l not in types]
            neg_labels = random.sample(neg_candidates, k=min(n_neg, len(neg_candidates)))
        else:
            neg_labels = []

        # Shuffle so the model doesn't learn position bias (positives always first)
        all_labels = pos_labels + neg_labels
        random.shuffle(all_labels)
        label_to_idx = {label: i for i, label in enumerate(all_labels)}

        # ---- Tokenize text ----
        text_enc = self.tokenizer(
            text,
            max_length=self.max_text_len,
            truncation=True,
            padding=False,
            return_offsets_mapping=True,
        )
        offset_mapping: List[Tuple[int, int]] = text_enc["offset_mapping"]

        # ---- Tokenize labels (label name + randomly selected description) ----
        label_texts = []
        for label in all_labels:
            defs = desc_map.get(label) or self.label_pool.get(label, {})
            desc = self._select_desc(defs)
            label_texts.append(f"{label}: {desc}" if desc else label)

        label_encs = [
            self.tokenizer(
                lt,
                max_length=self.max_label_len,
                truncation=True,
                padding=False,
            )
            for lt in label_texts
        ]

        # ---- Map entity char positions to token spans ----
        entity_spans: List[Tuple[int, int, int]] = []  # (tok_start, tok_end, label_idx)
        for ent in entities:
            etype = ent.get("type", "")
            if etype not in label_to_idx:
                continue
            pos = ent.get("pos", [])
            if len(pos) < 2:
                continue
            cs, ce = int(pos[0]), int(pos[1])
            tok_s, tok_e = self._char_to_token_span(offset_mapping, cs, ce)
            if tok_s is None or tok_e is None:
                continue
            if tok_e - tok_s + 1 > self.max_span_width:
                continue
            entity_spans.append((tok_s, tok_e, label_to_idx[etype]))

        return {
            "text_input_ids": text_enc["input_ids"],
            "text_attention_mask": text_enc["attention_mask"],
            "label_input_ids": [e["input_ids"] for e in label_encs],
            "label_attention_mask": [e["attention_mask"] for e in label_encs],
            "num_labels": len(all_labels),
            "entity_spans": entity_spans,
        }

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        return self._process_sample(self.data[idx])


# ------------------------------------------------------------------
# Collate function
# ------------------------------------------------------------------

def collate_fn(
    batch: List[Dict],
    tokenizer,
    max_span_width: int,
) -> SpanBatch:
    """
    Assembles a list of processed samples into a SpanBatch.

    M is dynamic: pad all samples to the largest num_labels in the current batch.
    This avoids dummy slots when samples have similar label counts, and handles
    samples with different numbers of positive labels naturally.

    Pads:
      - Text to max N in batch
      - Label slots to max M in batch (dummy [CLS][SEP] only when needed)
      - Label sequences to max L across all B*M labels
    Builds span_targets (B, S, M).
    """
    B = len(batch)
    # M = largest label count in this batch (variable per batch)
    M = max(s["num_labels"] for s in batch)

    pad_id = tokenizer.pad_token_id or 0
    cls_id = tokenizer.cls_token_id or 0
    sep_id = tokenizer.sep_token_id or 0
    dummy_ids = [cls_id, sep_id]
    dummy_mask = [1, 1]

    # ---- Pad text ----
    max_N = max(len(s["text_input_ids"]) for s in batch)
    text_ids = torch.full((B, max_N), pad_id, dtype=torch.long)
    text_mask = torch.zeros(B, max_N, dtype=torch.long)
    for i, s in enumerate(batch):
        n = len(s["text_input_ids"])
        text_ids[i, :n] = torch.tensor(s["text_input_ids"], dtype=torch.long)
        text_mask[i, :n] = torch.tensor(s["text_attention_mask"], dtype=torch.long)

    # ---- Collect labels, pad each sample to M slots ----
    all_label_ids: List[List[int]] = []
    all_label_masks: List[List[int]] = []
    for s in batch:
        lids = list(s["label_input_ids"])
        lmasks = list(s["label_attention_mask"])
        # Pad sample to M (only samples with fewer labels than the batch max need padding)
        while len(lids) < M:
            lids.append(dummy_ids)
            lmasks.append(dummy_mask)
        all_label_ids.extend(lids)
        all_label_masks.extend(lmasks)

    # Pad all label sequences to max L in this batch
    max_L = max(len(ids) for ids in all_label_ids)
    label_ids_t = torch.full((B * M, max_L), pad_id, dtype=torch.long)
    label_masks_t = torch.zeros(B * M, max_L, dtype=torch.long)
    for i, (ids, masks) in enumerate(zip(all_label_ids, all_label_masks)):
        l = len(ids)
        label_ids_t[i, :l] = torch.tensor(ids, dtype=torch.long)
        label_masks_t[i, :l] = torch.tensor(masks, dtype=torch.long)

    # ---- Build span targets (B, S, M) ----
    span_enum = SpanEnumerator(max_width=max_span_width)
    start_idx, end_idx, _ = span_enum.enumerate(seq_len=max_N, device=torch.device("cpu"))
    S = start_idx.numel()

    span_to_idx: Dict[Tuple[int, int], int] = {
        (int(start_idx[k]), int(end_idx[k])): k for k in range(S)
    }

    span_targets = torch.zeros(B, S, M, dtype=torch.float)
    for i, s in enumerate(batch):
        for tok_s, tok_e, label_idx in s["entity_spans"]:
            key = (tok_s, tok_e)
            if key in span_to_idx:
                span_targets[i, span_to_idx[key], label_idx] = 1.0

    return SpanBatch(
        text_input_ids=text_ids,
        text_attention_mask=text_mask,
        label_input_ids=label_ids_t,
        label_attention_mask=label_masks_t,
        num_labels=M,
        span_targets=span_targets,
    )


def build_collate_fn(tokenizer, max_span_width: int):
    """Returns a collate_fn closure suitable for DataLoader."""
    return partial(
        collate_fn,
        tokenizer=tokenizer,
        max_span_width=max_span_width,
    )
