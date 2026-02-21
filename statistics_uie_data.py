"""
Statistics for the Pile-NER → instruct_uie_ner converted JSONL dataset.

Computes:
  - Total samples
  - Average entities per sample
  - Average entity types per sample
  - Unique source datasets
  - Average text (sentence) length (chars)
  - Average entity mention length (chars)

Usage:
    python statistics_uie_data.py --data_dir dataset/
    python statistics_uie_data.py --data_dir dataset/ --verbose
"""

import argparse
import json
import os
from collections import Counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="dataset/")
    parser.add_argument(
        "--prefix",
        default="instruct_uie_ner_converted_description_",
        help="JSONL filename prefix to filter",
    )
    parser.add_argument("--verbose", action="store_true",
                        help="Show per-source breakdown and distributions")
    args = parser.parse_args()

    files = sorted(
        f for f in os.listdir(args.data_dir)
        if f.startswith(args.prefix) and f.endswith(".jsonl")
    )
    if not files:
        print(f"No JSONL files found in {args.data_dir} with prefix '{args.prefix}'")
        return
    print(f"Found {len(files)} JSONL files in {args.data_dir}\n")

    # Accumulators
    n_samples = 0
    n_entities_list = []      # per sample
    n_types_list = []         # per sample (unique types)
    text_len_list = []        # per sample (chars)
    entity_len_list = []      # per mention (chars)
    source_counter = Counter()  # dataset name → sample count

    for fname in files:
        fpath = os.path.join(args.data_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                sentence = obj.get("sentence", obj.get("text", ""))
                entities = obj.get("entities", [])
                types = obj.get("types", [])
                source = obj.get("dataset", "unknown")

                n_samples += 1
                text_len_list.append(len(sentence))
                n_types_list.append(len(types))
                source_counter[source] += 1

                valid_ents = [
                    ent for ent in entities
                    if ent.get("name", "").strip()
                ]
                n_entities_list.append(len(valid_ents))

                for ent in valid_ents:
                    entity_len_list.append(len(ent["name"].strip()))

    if n_samples == 0:
        print("No samples found.")
        return

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    def percentile(lst, p):
        if not lst:
            return 0
        s = sorted(lst)
        idx = int(len(s) * p / 100)
        return s[min(idx, len(s) - 1)]

    total_entities = sum(n_entities_list)

    print("=" * 60)
    print("  Instruct-UIE NER Dataset Statistics")
    print("=" * 60)
    print(f"  Total samples             : {n_samples:>12,}")
    print(f"  Unique source datasets    : {len(source_counter):>12,}")
    print()

    print(f"  Entities per sample")
    print(f"    Total entity mentions   : {total_entities:>12,}")
    print(f"    Mean                    : {avg(n_entities_list):>12.2f}")
    print(f"    Min / Max               : {min(n_entities_list):>5} / {max(n_entities_list)}")
    print(f"    p25 / p50 / p75 / p95   : "
          f"{percentile(n_entities_list,25):>3} / "
          f"{percentile(n_entities_list,50):>3} / "
          f"{percentile(n_entities_list,75):>3} / "
          f"{percentile(n_entities_list,95):>3}")
    print()

    print(f"  Entity types per sample")
    print(f"    Mean                    : {avg(n_types_list):>12.2f}")
    print(f"    Min / Max               : {min(n_types_list):>5} / {max(n_types_list)}")
    print(f"    p25 / p50 / p75 / p95   : "
          f"{percentile(n_types_list,25):>3} / "
          f"{percentile(n_types_list,50):>3} / "
          f"{percentile(n_types_list,75):>3} / "
          f"{percentile(n_types_list,95):>3}")
    print()

    print(f"  Text (sentence) length (chars)")
    print(f"    Mean                    : {avg(text_len_list):>12.1f}")
    print(f"    Min / Max               : {min(text_len_list):>5} / {max(text_len_list)}")
    print(f"    p25 / p50 / p75 / p95   : "
          f"{percentile(text_len_list,25):>4} / "
          f"{percentile(text_len_list,50):>4} / "
          f"{percentile(text_len_list,75):>4} / "
          f"{percentile(text_len_list,95):>4}")
    print()

    print(f"  Entity mention length (chars)")
    print(f"    Total mentions          : {len(entity_len_list):>12,}")
    print(f"    Mean                    : {avg(entity_len_list):>12.1f}")
    print(f"    Min / Max               : {min(entity_len_list):>5} / {max(entity_len_list)}")
    print(f"    p25 / p50 / p75 / p95   : "
          f"{percentile(entity_len_list,25):>3} / "
          f"{percentile(entity_len_list,50):>3} / "
          f"{percentile(entity_len_list,75):>3} / "
          f"{percentile(entity_len_list,95):>3}")
    print("=" * 60)

    if args.verbose:
        print("\n  Per-source breakdown:")
        print(f"  {'Source':<40} {'Samples':>8}")
        print("  " + "-" * 50)
        for src, cnt in sorted(source_counter.items(), key=lambda x: -x[1]):
            bar = "█" * min(30, cnt * 30 // max(source_counter.values()))
            print(f"  {src:<40} {cnt:>8,}  {bar}")


if __name__ == "__main__":
    main()
