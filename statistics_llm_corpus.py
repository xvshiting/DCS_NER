"""
Statistics for the LLM training corpus (Alpaca JSON format).

Computes:
  - Total number of samples
  - Average number of label:desc pairs per sample
  - Average description length (chars) per sample / overall
  - Number of unique source documents

Usage:
    python statistics_llm_corpus.py --input dataset/alpaca_20260209_v1.json
    python statistics_llm_corpus.py --input dataset/alpaca_20260209_v1.json --verbose
"""

import argparse
import json
import re
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

_TEXT_RE = re.compile(r"<Text>\s*(.*?)\s*</Text>", re.DOTALL)
_LABEL_DESC_RE = re.compile(r"^(.+?):\s+(.+)$")


def extract_text(instruction: str) -> str:
    """Extract source document text from the <Text>...</Text> block."""
    m = _TEXT_RE.search(instruction)
    return m.group(1).strip() if m else ""


def parse_output(output: str) -> List[Tuple[str, str]]:
    """Parse 'LABEL: DESC' lines from the output field."""
    pairs = []
    for line in output.strip().splitlines():
        line = line.strip()
        m = _LABEL_DESC_RE.match(line)
        if m:
            pairs.append((m.group(1).strip(), m.group(2).strip()))
    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Statistics for LLM training corpus")
    parser.add_argument("--input", required=True,
                        help="Path to alpaca JSON file (e.g. dataset/alpaca_20260209_v1.json)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-decile breakdown of label counts and desc lengths")
    args = parser.parse_args()

    print(f"Loading {args.input} ...", flush=True)
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Done. {len(data):,} samples loaded.\n")

    total_samples = len(data)
    unique_docs = set()

    label_counts: List[int] = []          # #labels per sample
    desc_lens: List[int] = []             # desc char length, one per (sample, label)
    samples_with_zero_pairs = 0

    for item in data:
        instruction = item.get("instruction", "")
        output = item.get("output", "")

        # Unique doc tracking
        doc_text = extract_text(instruction)
        if doc_text:
            unique_docs.add(doc_text)

        # Parse label:desc pairs
        pairs = parse_output(output)
        label_counts.append(len(pairs))
        if len(pairs) == 0:
            samples_with_zero_pairs += 1
        for _, desc in pairs:
            desc_lens.append(len(desc))

    # ---------------------------------------------------------------------------
    # Aggregate stats
    # ---------------------------------------------------------------------------
    n = total_samples
    avg_labels = sum(label_counts) / n if n else 0
    total_pairs = sum(label_counts)
    avg_desc_len = sum(desc_lens) / len(desc_lens) if desc_lens else 0

    label_counts_sorted = sorted(label_counts)
    desc_lens_sorted = sorted(desc_lens)

    def percentile(lst, p):
        if not lst:
            return 0
        idx = int(len(lst) * p / 100)
        return lst[min(idx, len(lst) - 1)]

    print("=" * 60)
    print("  LLM Training Corpus Statistics")
    print("=" * 60)
    print(f"  Total samples             : {total_samples:>12,}")
    print(f"  Unique source documents   : {len(unique_docs):>12,}")
    print(f"  Samples with 0 pairs      : {samples_with_zero_pairs:>12,}")
    print()
    print(f"  Label:desc pairs per sample")
    print(f"    Total pairs             : {total_pairs:>12,}")
    print(f"    Mean                    : {avg_labels:>12.2f}")
    print(f"    Min / Max               : {min(label_counts):>5} / {max(label_counts)}")
    print(f"    p25 / p50 / p75 / p95   : "
          f"{percentile(label_counts_sorted,25):>3} / "
          f"{percentile(label_counts_sorted,50):>3} / "
          f"{percentile(label_counts_sorted,75):>3} / "
          f"{percentile(label_counts_sorted,95):>3}")
    print()
    print(f"  Description length (chars)")
    print(f"    Total descriptions      : {len(desc_lens):>12,}")
    print(f"    Mean                    : {avg_desc_len:>12.1f}")
    print(f"    Min / Max               : {min(desc_lens):>5} / {max(desc_lens)}")
    print(f"    p25 / p50 / p75 / p95   : "
          f"{percentile(desc_lens_sorted,25):>3} / "
          f"{percentile(desc_lens_sorted,50):>3} / "
          f"{percentile(desc_lens_sorted,75):>3} / "
          f"{percentile(desc_lens_sorted,95):>3}")
    print("=" * 60)

    if args.verbose:
        # Label count distribution (bucketed)
        counter = Counter(label_counts)
        print("\n  Label count distribution (top 15 values):")
        for k, v in sorted(counter.items())[:15]:
            bar = "█" * min(40, v * 40 // max(counter.values()))
            print(f"    {k:>3} labels : {v:>7,}  {bar}")

        # Desc length buckets
        buckets = [0, 50, 100, 150, 200, 300, 500, 99999]
        print("\n  Desc length distribution:")
        for lo, hi in zip(buckets, buckets[1:]):
            count = sum(1 for l in desc_lens if lo <= l < hi)
            label = f"[{lo:>4},{hi if hi < 99999 else '∞':>5})"
            bar = "█" * min(40, count * 40 // len(desc_lens)) if desc_lens else ""
            print(f"    {label} : {count:>8,}  {bar}")


if __name__ == "__main__":
    main()
