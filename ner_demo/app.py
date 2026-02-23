"""
Flask web demo for LDG-GLiNER — interactive NER with editable descriptions.

Run from the project root:
    python ner_demo/app.py \\
        --gliner_ckpt checkpoints/run9/best/checkpoint.pt \\
        --backbone /data/model_hub/mdeberta-v3-base \\
        [--llm_base /data/model_hub/qwen/Qwen3-1.7B] \\
        [--llm_adapter /path/to/lora/adapter] \\
        --port 5000

Description modes (selectable in UI at runtime):
    none        Always available. Labels sent as-is (no description).
    cache       Available if JSONL files exist under --desc_cache_dir.
    llm         Available if --llm_base is provided.
"""

import argparse
import glob as glob_mod
import os
import sys
from dataclasses import replace as dc_replace

import torch
from flask import Flask, jsonify, render_template, request
from transformers import AutoTokenizer

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from evaluate import (
    EvalConfig,
    DescriptionProvider,
    load_gliner,
    encode_sample,
    collate_for_eval,
    decode_predictions,
)
from ldggliner.model import SpanBatch

app = Flask(__name__, template_folder=os.path.join(_HERE, "templates"))

# Global state
_model = None
_tokenizer = None
_providers: dict = {}        # mode -> DescriptionProvider
_available_modes: list = []
_cfg: EvalConfig = None
_device: torch.device = None
_amp_dtype = None


# ---------------------------------------------------------------------------
# Core inference (accepts descriptions dict directly)
# ---------------------------------------------------------------------------

def _run_gliner(text: str, labels: list, descriptions: dict, threshold: float) -> list:
    """
    Run GLiNER on one text sample.
    descriptions: {label: desc_str}  — empty string means label-only.
    Returns list of {"text", "start", "end", "label"} dicts.
    """
    label_descs = {l: descriptions.get(l, "") for l in labels}

    enc = encode_sample(text, label_descs, _tokenizer, _cfg.max_text_len, _cfg.max_label_len)
    if enc is None:
        return []

    batch, offset_mappings = collate_for_eval([enc], _tokenizer)
    batch = SpanBatch(
        text_input_ids=batch.text_input_ids.to(_device),
        text_attention_mask=batch.text_attention_mask.to(_device),
        label_input_ids=batch.label_input_ids.to(_device),
        label_attention_mask=batch.label_attention_mask.to(_device),
        num_labels=batch.num_labels,
        span_targets=None,
    )

    with torch.no_grad():
        with torch.autocast(
            device_type=_device.type,
            dtype=_amp_dtype,
            enabled=_amp_dtype is not None,
        ):
            out = _model(batch)

    pred_sets = decode_predictions(
        logits=out["logits"].cpu(),
        span_mask=out["span_mask"].cpu(),
        start_idx=out["start_idx"].cpu(),
        end_idx=out["end_idx"].cpu(),
        offset_mappings=offset_mappings,
        label_lists=[enc["labels"]],
        threshold=threshold,
    )

    return [
        {"text": text[cs:ce], "start": cs, "end": ce, "label": etype}
        for cs, ce, etype in sorted(pred_sets[0])
    ]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/info")
def info():
    return jsonify({
        "model_arch": _cfg.model_arch,
        "backbone": os.path.basename(_cfg.backbone),
        "threshold": _cfg.threshold,
        "device": str(_device),
        "available_modes": _available_modes,
    })


@app.route("/describe", methods=["POST"])
def describe():
    """Generate descriptions for a set of labels given the input text.

    Request JSON: {text, labels, desc_mode}
    Response JSON: {descriptions: {label: desc_str}}
    """
    data = request.get_json(force=True, silent=True) or {}
    text = (data.get("text") or "").strip()
    labels = [l.strip() for l in data.get("labels", []) if isinstance(l, str) and l.strip()]
    desc_mode = data.get("desc_mode", "none")

    if not text:
        return jsonify({"error": "text is required"}), 400
    if not labels:
        return jsonify({"error": "at least one label is required"}), 400
    if desc_mode not in _providers:
        return jsonify({
            "error": f"Mode '{desc_mode}' not available. Available: {_available_modes}"
        }), 400

    try:
        descs = _providers[desc_mode].get(text, labels)
        return jsonify({"descriptions": descs})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    """Run NER with caller-supplied descriptions (may be user-edited).

    Request JSON: {text, labels, descriptions, threshold}
      descriptions: {label: desc_str}  — optional, defaults to empty strings
    Response JSON: {entities: [{text, start, end, label}, ...]}
    """
    data = request.get_json(force=True, silent=True) or {}
    text = (data.get("text") or "").strip()
    labels = [l.strip() for l in data.get("labels", []) if isinstance(l, str) and l.strip()]
    descriptions = data.get("descriptions") or {}
    threshold = float(data.get("threshold", _cfg.threshold))

    if not text:
        return jsonify({"error": "text is required"}), 400
    if not labels:
        return jsonify({"error": "at least one entity type label is required"}), 400

    try:
        entities = _run_gliner(text, labels, descriptions, threshold)
        return jsonify({"entities": entities})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ---------------------------------------------------------------------------
# Provider initialization
# ---------------------------------------------------------------------------

def _init_providers(base_cfg: EvalConfig, args):
    global _providers, _available_modes

    # none — always available, zero cost
    _providers["none"] = DescriptionProvider(dc_replace(base_cfg, desc_mode="none"))
    _available_modes.append("none")
    print("  none   : ready")

    # cache — available if JSONL description files are present
    pattern = os.path.join(
        base_cfg.desc_cache_dir,
        "instruct_uie_ner_converted_description_*.jsonl",
    )
    if glob_mod.glob(pattern):
        try:
            _providers["cache"] = DescriptionProvider(
                dc_replace(base_cfg, desc_mode="cache", llm_desc_key=args.llm_desc_key)
            )
            _available_modes.append("cache")
            print("  cache  : ready")
        except Exception as exc:
            print(f"  cache  : FAILED — {exc}")

    # llm — available only if --llm_base is supplied at startup
    if args.llm_base:
        try:
            _providers["llm"] = DescriptionProvider(
                dc_replace(
                    base_cfg,
                    desc_mode="llm",
                    llm_base=args.llm_base,
                    llm_adapter=args.llm_adapter,
                    llm_max_new_tokens=args.llm_max_new_tokens,
                )
            )
            _available_modes.append("llm")
            print("  llm    : ready")
        except Exception as exc:
            print(f"  llm    : FAILED — {exc}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    global _model, _tokenizer, _cfg, _device, _amp_dtype

    parser = argparse.ArgumentParser(
        description="LDG-GLiNER NER Web Demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # GLiNER
    parser.add_argument(
        "--gliner_ckpt",
        default=os.path.join(_ROOT, "checkpoints/run9/best/checkpoint.pt"),
    )
    parser.add_argument("--backbone", default="/data/model_hub/mdeberta-v3-base")
    parser.add_argument(
        "--model_arch", default="deberta_span_v1",
        choices=["deberta_span_v1", "deberta_span_v2", "deberta_span_v3"],
    )
    parser.add_argument("--max_span_width", type=int, default=12)
    parser.add_argument("--label_chunk_size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max_text_len", type=int, default=256)
    parser.add_argument("--max_label_len", type=int, default=128)
    # Description cache
    parser.add_argument(
        "--desc_cache_dir", default=os.path.join(_ROOT, "dataset/"),
        help="Dir containing instruct_uie_ner_converted_description_*.jsonl",
    )
    parser.add_argument("--llm_desc_key", default="D6")
    # LLM (optional — enables llm desc mode)
    parser.add_argument(
        "--llm_base", default=None,
        help="Local LLM model path; required to enable llm description mode",
    )
    parser.add_argument("--llm_adapter", default=None, help="LoRA adapter path")
    parser.add_argument("--llm_max_new_tokens", type=int, default=512)
    # Server
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    _cfg = EvalConfig(
        gliner_ckpt=args.gliner_ckpt,
        backbone=args.backbone,
        model_arch=args.model_arch,
        max_span_width=args.max_span_width,
        label_chunk_size=args.label_chunk_size,
        threshold=args.threshold,
        max_text_len=args.max_text_len,
        max_label_len=args.max_label_len,
        desc_mode="none",
        desc_cache_dir=args.desc_cache_dir,
        llm_desc_key=args.llm_desc_key,
        llm_max_new_tokens=args.llm_max_new_tokens,
    )

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _amp_dtype = (
        torch.bfloat16
        if _device.type == "cuda" and torch.cuda.is_bf16_supported()
        else None
    )

    print(f"Device     : {_device}")
    print(f"Tokenizer  : {_cfg.backbone}")
    _tokenizer = AutoTokenizer.from_pretrained(_cfg.backbone)

    print(f"GLiNER     : {_cfg.gliner_ckpt}")
    _model = load_gliner(_cfg, _device)

    print("Providers  :")
    _init_providers(_cfg, args)
    print(f"Available  : {_available_modes}")

    print(f"\nServer ready → http://localhost:{args.port}\n")
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
