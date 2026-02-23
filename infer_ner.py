"""
Interactive NER inference script for LDG-GLiNER.

Given a text and a set of entity type labels, runs the trained GLiNER model
and prints extracted entities with their character positions.

Usage:
    python infer_ner.py \\
        --gliner_ckpt checkpoints/run9/best/checkpoint.pt \\
        --backbone /data/model_hub/mdeberta-v3-base \\
        --desc_mode none

Commands during the interactive session:
    quit / exit        - exit the program
    threshold <float>  - change decision threshold (e.g. "threshold 0.3")
"""

import argparse
import sys

import torch
from transformers import AutoTokenizer

from evaluate import (
    EvalConfig,
    DescriptionProvider,
    load_gliner,
    encode_sample,
    collate_for_eval,
    decode_predictions,
)
from ldggliner.model import SpanBatch


# ---------------------------------------------------------------------------
# Single-sample inference
# ---------------------------------------------------------------------------

def predict(
    text: str,
    labels: list,
    desc_provider: DescriptionProvider,
    model,
    tokenizer,
    cfg: EvalConfig,
    device: torch.device,
    amp_dtype,
    threshold: float,
) -> list:
    """Return a sorted list of (char_start, char_end, label) tuples."""
    label_descs = desc_provider.get(text, labels)

    enc = encode_sample(text, label_descs, tokenizer, cfg.max_text_len, cfg.max_label_len)
    if enc is None:
        return []

    batch, offset_mappings = collate_for_eval([enc], tokenizer)
    batch = SpanBatch(
        text_input_ids=batch.text_input_ids.to(device),
        text_attention_mask=batch.text_attention_mask.to(device),
        label_input_ids=batch.label_input_ids.to(device),
        label_attention_mask=batch.label_attention_mask.to(device),
        num_labels=batch.num_labels,
        span_targets=None,
    )

    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            out = model(batch)

    pred_sets = decode_predictions(
        logits=out["logits"].cpu(),
        span_mask=out["span_mask"].cpu(),
        start_idx=out["start_idx"].cpu(),
        end_idx=out["end_idx"].cpu(),
        offset_mappings=offset_mappings,
        label_lists=[enc["labels"]],
        threshold=threshold,
        flat_ner=cfg.flat_ner,
        multi_label=cfg.multi_label,
    )
    return sorted(pred_sets[0])


# ---------------------------------------------------------------------------
# Interactive loop
# ---------------------------------------------------------------------------

def interactive(model, tokenizer, desc_provider, cfg, device, amp_dtype):
    threshold = cfg.threshold
    print(f"\nDesc mode : {cfg.desc_mode}")
    print(f"Threshold : {threshold}")
    print("Commands  : 'threshold <float>' to adjust, 'quit' to exit\n")

    while True:
        print("=" * 60)
        try:
            text = input("Text: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not text:
            continue
        if text.lower() in ("quit", "exit"):
            print("Bye.")
            break
        if text.lower().startswith("threshold "):
            try:
                threshold = float(text.split()[1])
                print(f"Threshold set to {threshold}")
            except (IndexError, ValueError):
                print("Usage: threshold <float>")
            continue

        try:
            labels_raw = input("Entity types (comma-separated): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not labels_raw or labels_raw.lower() in ("quit", "exit"):
            continue

        labels = [l.strip() for l in labels_raw.split(",") if l.strip()]
        if not labels:
            print("No labels given, skipping.")
            continue

        entities = predict(
            text, labels, desc_provider, model, tokenizer,
            cfg, device, amp_dtype, threshold,
        )

        print("\nExtracted entities:")
        if not entities:
            print("  (none)")
        else:
            for cs, ce, etype in entities:
                span_text = text[cs:ce]
                print(f"  [{etype}]  \"{span_text}\"  (chars {cs}–{ce})")
        print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Interactive NER inference with LDG-GLiNER",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--gliner_ckpt", default="checkpoints/run9/best/checkpoint.pt",
                        help="Path to trained GLiNER checkpoint (.pt)")
    parser.add_argument("--backbone", default="/data/model_hub/mdeberta-v3-base",
                        help="mDeBERTa backbone model path or HuggingFace name")
    parser.add_argument("--model_arch", default="deberta_span_v1",
                        choices=["deberta_span_v1", "deberta_span_v2", "deberta_span_v3"])
    parser.add_argument("--max_span_width", type=int, default=12)
    parser.add_argument("--label_chunk_size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Sigmoid threshold for span prediction")
    parser.add_argument("--max_text_len", type=int, default=256)
    parser.add_argument("--max_label_len", type=int, default=128)
    parser.add_argument("--desc_mode", default="none",
                        choices=["none", "cache", "precomputed", "llm"],
                        help="Description source: none | cache | precomputed | llm")
    parser.add_argument("--desc_cache_dir", default="dataset/",
                        help="Dir containing instruct_uie_ner_converted_description_*.jsonl")
    parser.add_argument("--precomputed_descs", default="descriptions_cache.jsonl")
    parser.add_argument("--llm_desc_key", default="D6",
                        help="Which definition key to use from cache (D1-D6)")
    parser.add_argument("--llm_base", default=None,
                        help="Local LLM model path (required for llm desc mode)")
    parser.add_argument("--llm_adapter", default=None,
                        help="LoRA adapter path for the LLM")
    parser.add_argument("--llm_max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    if args.desc_mode == "llm" and not args.llm_base:
        parser.error("--llm_base is required when --desc_mode llm")

    cfg = EvalConfig(
        gliner_ckpt=args.gliner_ckpt,
        backbone=args.backbone,
        model_arch=args.model_arch,
        max_span_width=args.max_span_width,
        label_chunk_size=args.label_chunk_size,
        threshold=args.threshold,
        max_text_len=args.max_text_len,
        max_label_len=args.max_label_len,
        desc_mode=args.desc_mode,
        desc_cache_dir=args.desc_cache_dir,
        precomputed_descs=args.precomputed_descs,
        llm_desc_key=args.llm_desc_key,
        llm_base=args.llm_base or "/data/model_hub/qwen/Qwen3-1.7B",
        llm_adapter=args.llm_adapter,
        llm_max_new_tokens=args.llm_max_new_tokens,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else None
    )
    print(f"Device: {device}")

    print(f"Loading tokenizer: {cfg.backbone}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.backbone)

    print(f"Loading model: {cfg.gliner_ckpt}")
    model = load_gliner(cfg, device)

    print(f"Loading descriptions ({cfg.desc_mode} mode)...")
    desc_provider = DescriptionProvider(cfg)

    interactive(model, tokenizer, desc_provider, cfg, device, amp_dtype)


if __name__ == "__main__":
    main()
