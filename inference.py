import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ──────────────────────────────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────────────────────────────

DEFAULT_BASE_MODEL = "/data/model_hub/qwen/Qwen3-1.7B"
DEFAULT_ADAPTER = (
    "/home/will/Projects/LLaMA-Factory/saves/Qwen3-1.7B-Thinking/lora/"
    "train_2026-02-09-12-46-24"
)

# ──────────────────────────────────────────────────────────────────────
# Prompt templates (must match training)
# ──────────────────────────────────────────────────────────────────────

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

OPEN_TEMPLATE = """You are given a text.

Identify the labels that are relevant to this text,
and generate a description for each identified label.

<Text>
{TEXT}
</Text>

Output format:
Each line must be:
LABEL: DESCRIPTION
"""

# ──────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────

def load_model(base_model_path, adapter_path):
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompt, max_new_tokens=1024):
    messages = [
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    # Decode only newly generated tokens
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)

# ──────────────────────────────────────────────────────────────────────
# Interactive loop
# ──────────────────────────────────────────────────────────────────────

def interactive(model, tokenizer, mode):
    print(f"\nMode: {'open' if mode == 'open' else 'closed'}")
    print("Type 'quit' to exit, 'switch' to toggle mode.\n")

    while True:
        # --- Input text ---
        print("=" * 60)
        text = input("Text: ").strip()
        if text.lower() == "quit":
            break
        if text.lower() == "switch":
            mode = "open" if mode == "closed" else "closed"
            print(f"Switched to: {mode}")
            continue

        # --- Build prompt ---
        if mode == "closed":
            labels_input = input("Labels (comma-separated): ").strip()
            if labels_input.lower() == "quit":
                break
            labels = [l.strip() for l in labels_input.split(",") if l.strip()]
            prompt = CLOSED_TEMPLATE.format(
                TEXT=text,
                LABEL_SET="\n".join(labels),
            )
        else:
            prompt = OPEN_TEMPLATE.format(TEXT=text)

        # --- Generate ---
        print("-" * 60)
        result = generate(model, tokenizer, prompt)
        print(result)
        print()


def main():
    parser = argparse.ArgumentParser(description="NER label description inference")
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter", type=str, default=DEFAULT_ADAPTER)
    parser.add_argument("--mode", type=str, default="closed", choices=["open", "closed"],
                        help="open: auto-identify labels; closed: specify label set")
    # Single-shot mode (non-interactive)
    parser.add_argument("--text", type=str, default=None, help="Input text (non-interactive)")
    parser.add_argument("--labels", type=str, default=None,
                        help="Comma-separated labels for closed mode (non-interactive)")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    args = parser.parse_args()

    print(f"Loading base model: {args.base_model}")
    print(f"Loading adapter: {args.adapter}")
    model, tokenizer = load_model(args.base_model, args.adapter)
    print("Model loaded.\n")

    # Single-shot mode
    if args.text is not None:
        if args.mode == "closed":
            if not args.labels:
                parser.error("--labels is required in closed mode")
            labels = [l.strip() for l in args.labels.split(",")]
            prompt = CLOSED_TEMPLATE.format(
                TEXT=args.text,
                LABEL_SET="\n".join(labels),
            )
        else:
            prompt = OPEN_TEMPLATE.format(TEXT=args.text)

        result = generate(model, tokenizer, prompt, args.max_new_tokens)
        print(result)
        return

    # Interactive mode
    interactive(model, tokenizer, args.mode)


if __name__ == "__main__":
    main()
