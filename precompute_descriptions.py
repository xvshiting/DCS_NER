"""
Pre-compute LLM descriptions for every sample in the test split.

Saves results to a JSONL file (one line per sample):
    {"dataset": "ACE_2004", "split_idx": 0, "descriptions": {"label": "desc", ...}}

The output file can then be used by evaluate.py with --desc_mode precomputed.

Supports resume: already-computed samples are skipped on restart.

Usage:
    python precompute_descriptions.py \\
        --test_data /data/dataset/ner/instruct_uie_ner \\
        --llm_base /data/model_hub/qwen/Qwen3-1.7B \\
        --llm_adapter /home/will/Projects/LLaMA-Factory/saves/Qwen3-1.7B-Thinking/lora/train_2026-02-09-12-46-24 \\
        --output descriptions_cache.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


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


@dataclass
class Config:
    test_data: str = "/data/dataset/ner/instruct_uie_ner"
    llm_base: str = "/data/model_hub/qwen/Qwen3-1.7B"
    llm_adapter: Optional[str] = "/home/will/Projects/LLaMA-Factory/saves/Qwen3-1.7B-Thinking/lora/train_2026-02-09-12-46-24"
    llm_max_new_tokens: int = 4096
    output: str = "descriptions_cache.jsonl"
    split: str = "test"
    max_samples: Optional[int] = None   # limit for debugging
    verbose: int = 0                    # print raw LLM output for first N samples


def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--test_data", default=Config.test_data)
    p.add_argument("--llm_base", default=Config.llm_base)
    p.add_argument("--llm_adapter", default=Config.llm_adapter)
    p.add_argument("--llm_max_new_tokens", type=int, default=Config.llm_max_new_tokens)
    p.add_argument("--output", default=Config.output)
    p.add_argument("--split", default=Config.split)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--verbose", type=int, default=0,
                   help="Print raw LLM output for first N samples (for debugging)")
    args = p.parse_args()
    return Config(**vars(args))


def _parse_llm_output(text: str) -> Dict[str, str]:
    # Strip <think>...</think> blocks (Qwen3 thinking model)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    result = {}
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^(.+?):\s+(.+)$", line)
        if m:
            result[m.group(1).strip().lower()] = m.group(2).strip()
    return result


def generate_descriptions(
    text: str,
    labels: List[str],
    model,
    tokenizer,
    max_new_tokens: int,
) -> Dict[str, str]:
    label_set_str = "\n".join(labels)
    prompt = CLOSED_TEMPLATE.format(TEXT=text[:2000], LABEL_SET=label_set_str)
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True)

    parsed = _parse_llm_output(raw)
    # Align back to original label casing; fill missing with ""
    label_lower_map = {l.lower(): l for l in labels}
    result: Dict[str, str] = {}
    for k, v in parsed.items():
        if k in label_lower_map:
            result[label_lower_map[k]] = v
    for l in labels:
        if l not in result:
            result[l] = ""
    return result


def load_done(output_path: str) -> Set[Tuple[str, int]]:
    """Return set of (dataset, split_idx) already written to the output file."""
    done: Set[Tuple[str, int]] = set()
    if not os.path.exists(output_path):
        return done
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                done.add((obj["dataset"], obj["split_idx"]))
            except Exception:
                pass
    return done


def main():
    cfg = parse_args()

    # ---- Load LLM ----
    logger.info(f"Loading tokenizer from {cfg.llm_base}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.llm_base, trust_remote_code=True)

    logger.info(f"Loading LLM from {cfg.llm_base}")
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        cfg.llm_base,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if cfg.llm_adapter:
        from peft import PeftModel
        logger.info(f"Loading LoRA adapter from {cfg.llm_adapter}")
        model = PeftModel.from_pretrained(model, cfg.llm_adapter)
    model.eval()

    # ---- Load test split ----
    logger.info(f"Loading dataset from {cfg.test_data}")
    from datasets import load_from_disk
    ds = load_from_disk(cfg.test_data)
    test_split = ds[cfg.split]
    total = len(test_split)
    if cfg.max_samples:
        total = min(total, cfg.max_samples)
    logger.info(f"Test split size: {total} samples")

    # ---- Resume support ----
    done = load_done(cfg.output)
    logger.info(f"Already done: {len(done)} samples (will skip)")

    # ---- Generate ----
    out_f = open(cfg.output, "a", encoding="utf-8")
    try:
        pbar = tqdm(range(total), desc="Precomputing descriptions", unit="sample")
        for i in pbar:
            raw = test_split[i]
            dataset_name = raw.get("dataset", "unknown")
            key = (dataset_name, i)

            if key in done:
                continue

            text = raw.get("sentence", raw.get("text", ""))
            types = list(dict.fromkeys(
                ent["type"].lower()
                for ent in raw.get("entities", [])
                if ent.get("type") and len(ent.get("pos", [])) >= 2
            ))
            if not types:
                # Write empty entry so we know it's been processed
                obj = {"dataset": dataset_name, "split_idx": i, "descriptions": {}}
                out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                out_f.flush()
                continue

            pbar.set_postfix(subset=dataset_name, labels=len(types))

            try:
                descs = generate_descriptions(
                    text, types, model, tokenizer, cfg.llm_max_new_tokens
                )
            except Exception as e:
                logger.warning(f"Sample {i} ({dataset_name}) failed: {e}")
                descs = {l: "" for l in types}

            obj = {
                "dataset": dataset_name,
                "split_idx": i,
                "descriptions": descs,
            }
            out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            out_f.flush()
    finally:
        out_f.close()

    logger.info(f"Done. Descriptions saved to {cfg.output}")


if __name__ == "__main__":
    main()
