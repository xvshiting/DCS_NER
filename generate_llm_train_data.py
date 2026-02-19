import json
import os
import random
import argparse
from datetime import date
from tqdm import tqdm

def next_version(output_dir, today, prefix):
    """Auto-increment version by scanning existing files for the same date."""
    v = 1
    for fname in os.listdir(output_dir):
        # Match pattern: {prefix}_{today}_v{N}.json
        if fname.startswith(f"{prefix}_{today}_v") and fname.endswith(".json"):
            try:
                n = int(fname.rsplit("_v", 1)[1].replace(".json", ""))
                v = max(v, n + 1)
            except ValueError:
                continue
    return v

# ──────────────────────────────────────────────────────────────────────
# Templates
# ──────────────────────────────────────────────────────────────────────

TASK1_TEMPLATE = """You are given a text and a set of target labels.

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

TASK2_TEMPLATE = """You are given a text.

Identify the labels that are relevant to this text,
and generate a description for each identified label.

<Text>
{TEXT}
</Text>

Output format:
Each line must be:
LABEL: DESCRIPTION

Only include labels clearly supported by the text.
Do not include speculative or irrelevant labels."""

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def load_all_data(dataset_dir):
    """Load all description JSONL files from dataset directory."""
    data = []
    files = sorted(f for f in os.listdir(dataset_dir)
                   if f.startswith("instruct_uie_ner_converted_description_") and f.endswith(".jsonl"))
    print(f"Found {len(files)} JSONL files in {dataset_dir}")
    for fname in tqdm(files, desc="Loading data"):
        fpath = os.path.join(dataset_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                # Skip records with missing description
                if obj.get("description") is None:
                    continue
                data.append(obj)
    print(f"Loaded {len(data)} records")
    return data


def build_label_desc_map(record):
    """Build {label: {D1: ..., D2: ...}} from a record's description field."""
    desc_map = {}
    for item in record["description"]:
        label = item["label"]
        if "definitions" in item:
            defs = item["definitions"]
        else:
            # Flat format: D1-D6 directly on the item
            defs = {k: v for k, v in item.items() if k.startswith("D")}
        desc_map[label] = defs
    return desc_map


def pick_description(defs):
    """Randomly choose D1 or D2 as the description. Fall back if either is missing."""
    candidates = [defs[k] for k in ("D1", "D2") if k in defs]
    if not candidates:
        # Last resort: pick any available definition
        candidates = list(defs.values())
    return random.choice(candidates)


def format_output(labels, desc_map):
    """Format output lines: LABEL: DESCRIPTION"""
    lines = []
    for label in labels:
        desc = pick_description(desc_map[label])
        lines.append(f"{label}: {desc}")
    return "\n".join(lines)


def format_label_set(labels):
    """Format LABEL_SET block for task 1 template."""
    return "\n".join(labels)


def sample_label_count(all_labels):
    """Sample a random subset of labels (at least 1, at most len-1 if len>1)."""
    n = len(all_labels)
    if n <= 1:
        return list(all_labels)
    k = random.randint(1, n - 1)
    return random.sample(all_labels, k)


# ──────────────────────────────────────────────────────────────────────
# Task generators
# ──────────────────────────────────────────────────────────────────────

def generate_task1(record):
    """
    Task 1 (Closed-world NER): Given text + label set → generate descriptions.
    Returns 3 alpaca samples per record:
      1. All labels
      2. Sampled subset of labels (round 1)
      3. Sampled subset of labels (round 2)
    """
    text = record["sentence"]
    desc_map = build_label_desc_map(record)
    all_labels = list(desc_map.keys())

    if not all_labels:
        return []

    samples = []

    # --- Round 1: all labels ---
    instruction = TASK1_TEMPLATE.format(
        TEXT=text,
        LABEL_SET=format_label_set(all_labels),
    )
    output = format_output(all_labels, desc_map)
    samples.append({
        "instruction": instruction,
        "input": "",
        "output": output,
    })

    # --- Round 2 & 3: sampled labels ---
    for _ in range(2):
        sampled = sample_label_count(all_labels)
        instruction = TASK1_TEMPLATE.format(
            TEXT=text,
            LABEL_SET=format_label_set(sampled),
        )
        output = format_output(sampled, desc_map)
        samples.append({
            "instruction": instruction,
            "input": "",
            "output": output,
        })

    return samples


def generate_task2(record):
    """
    Task 2 (Open NER): Given text → identify all labels and generate descriptions.
    Returns 1 alpaca sample per record.
    """
    text = record["sentence"]
    desc_map = build_label_desc_map(record)
    all_labels = list(desc_map.keys())

    if not all_labels:
        return []

    instruction = TASK2_TEMPLATE.format(TEXT=text)
    output = format_output(all_labels, desc_map)

    return [{
        "instruction": instruction,
        "input": "",
        "output": output,
    }]


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate alpaca-format training data for LLamaFactory")
    parser.add_argument("--dataset_dir", type=str, default="./dataset/",
                        help="Directory containing description JSONL files")
    parser.add_argument("--output_dir", type=str, default="./dataset/",
                        help="Directory to write output alpaca JSON files")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--task", type=str, default="all", choices=["1", "2", "all"],
                        help="Which task to generate: 1, 2, or all")
    args = parser.parse_args()

    random.seed(args.seed)
    data = load_all_data(args.dataset_dir)
    today = date.today().strftime("%Y%m%d")

    all_samples = []

    if args.task in ("1", "all"):
        task1_samples = []
        for record in tqdm(data, desc="Generating Task 1"):
            task1_samples.extend(generate_task1(record))
        print(f"Task 1: {len(task1_samples)} samples")
        all_samples.extend(task1_samples)

    if args.task in ("2", "all"):
        task2_samples = []
        for record in tqdm(data, desc="Generating Task 2"):
            task2_samples.extend(generate_task2(record))
        print(f"Task 2: {len(task2_samples)} samples")
        all_samples.extend(task2_samples)

    if args.task == "all":
        random.shuffle(all_samples)
        prefix = "alpaca"
    else:
        prefix = f"task{args.task}_alpaca"

    v = next_version(args.output_dir, today, prefix)
    out_path = os.path.join(args.output_dir, f"{prefix}_{today}_v{v}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)
    print(f"{len(all_samples)} samples → {out_path}")


if __name__ == "__main__":
    main()
