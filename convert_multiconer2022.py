#!/usr/bin/env python3
"""
convert_multiconer2022.py
=========================
Convert MultiCoNER2022 CoNLL files into a HuggingFace Arrow DatasetDict
that evaluate.py can consume directly via --test_data.

Output schema (same as instruct_uie_ner):
    sentence : str
    entities : list[{"pos": [char_start, char_end], "type": str}]
    dataset  : str   ← language name; evaluate.py groups results by this field

Deduplication:
    Within each language, sentences with identical text are merged (entity sets
    are unioned), so the LLM description generator is never called twice for
    the same text.

File selection per language:
    - If a hash-suffixed file exists (EN, DE, BN) it is used by default —
      this is the original MultiCoNER2022 competition test set.
    - Pass --no_prefer_hash to use the larger main test file instead.

Label mapping (MultiCoNER2022 short tags → human-readable):
    CORP → corporation    CW   → creative work
    GRP  → group          LOC  → location
    PER  → person         PROD → product

Usage:
    python convert_multiconer2022.py \\
        --data_dir /data/dataset/ner/multiconer2022 \\
        --output   /data/dataset/ner/multiconer2022_arrow

    # Select specific languages only:
    python convert_multiconer2022.py \\
        --data_dir /data/dataset/ner/multiconer2022 \\
        --output   /data/dataset/ner/multiconer2022_arrow \\
        --langs EN-English DE-German ZH-Chinese

Then evaluate directly:
    python evaluate.py \\
        --test_data /data/dataset/ner/multiconer2022_arrow \\
        --desc_mode none \\
        --gliner_ckpt checkpoints/run9/best/checkpoint.pt \\
        --backbone  /data/model_hub/mdeberta-v3-base
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

# ── MultiCoNER2022 label → human-readable name ────────────────────────────────
LABEL_MAP: Dict[str, str] = {
    "CORP": "corporation",
    "CW":   "creative work",
    "GRP":  "group",
    "LOC":  "location",
    "PER":  "person",
    "PROD": "product",
}

# Languages that have a competition-test hash file (subset of main test)
_LANGS_WITH_HASH = {"BN-Bangla", "DE-German", "EN-English"}

# Skipped by default: MULTI overlaps with all individual languages;
# MIX is code-mixed data, separate from the monolingual tracks.
_DEFAULT_SKIP = {"MULTI_Multilingual", "MIX_Code_mixed"}


# ─────────────────────────────────────────────────────────────────────────────
# File discovery
# ─────────────────────────────────────────────────────────────────────────────

def _lang_prefix(lang_dir: str) -> str:
    """EN-English → en,  MIX_Code_mixed → mix,  MULTI_Multilingual → multi"""
    name = os.path.basename(lang_dir)
    return re.split(r"[-_]", name)[0].lower()


def _find_test_file(lang_dir: str, prefer_hash: bool) -> Optional[str]:
    """
    Return the path to the test CoNLL file for this language directory.
    Priority:
      prefer_hash=True  → hash file if present, else main file
      prefer_hash=False → main file always
    """
    prefix = _lang_prefix(lang_dir)
    main = os.path.join(lang_dir, f"{prefix}_test.conll")

    if prefer_hash:
        # Hash file: any file matching <prefix>_test.conll.<HEX>
        pattern = re.compile(rf"^{re.escape(prefix)}_test\.conll\.[0-9A-Fa-f]+$")
        candidates = [
            os.path.join(lang_dir, f)
            for f in sorted(os.listdir(lang_dir))
            if pattern.match(f)
        ]
        if candidates:
            return candidates[0]   # take the first (usually only one)

    return main if os.path.exists(main) else None


# ─────────────────────────────────────────────────────────────────────────────
# CoNLL parser
# ─────────────────────────────────────────────────────────────────────────────

def _build_sample(
    tokens: List[str],
    bio_tags: List[str],
    lang_name: str,
) -> Dict:
    """
    Reconstruct sentence text (space-joined tokens) and compute character-level
    entity spans.  char_end is exclusive (Python slice convention), consistent
    with the evaluate.py / mDeBERTa offset_mapping convention.
    """
    text = " ".join(tokens)

    # Precompute char offsets for every token
    char_starts: List[int] = []
    char_ends: List[int] = []
    pos = 0
    for tok in tokens:
        char_starts.append(pos)
        char_ends.append(pos + len(tok))   # exclusive end
        pos += len(tok) + 1                # +1 for the space separator

    # BIO → entity spans
    entities = []
    i = 0
    while i < len(bio_tags):
        tag = bio_tags[i]
        if tag.startswith("B-"):
            etype = tag[2:]
            j = i + 1
            # Consume consecutive I- tokens of the same type
            while j < len(bio_tags) and bio_tags[j] == f"I-{etype}":
                j += 1
            cs = char_starts[i]
            ce = char_ends[j - 1]
            mapped = LABEL_MAP.get(etype, etype.lower())
            entities.append({
                "name": text[cs:ce],
                "pos":  [cs, ce],
                "type": mapped,
            })
            i = j
        else:
            i += 1  # O or orphan I- (skip)

    # Unique entity types present in this sample (order-preserving)
    types = list(dict.fromkeys(e["type"] for e in entities))

    return {
        "sentence": text,
        "entities": entities,
        "dataset":  lang_name,
        "types":    types,
    }


def parse_conll_file(path: str, lang_name: str) -> List[Dict]:
    """Parse a single MultiCoNER2022 CoNLL file; return list of sample dicts."""
    samples: List[Dict] = []
    tokens: List[str] = []
    bio_tags: List[str] = []

    def _flush():
        if tokens:
            samples.append(_build_sample(list(tokens), list(bio_tags), lang_name))
            tokens.clear()
            bio_tags.clear()

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")

            if line.startswith("# id"):
                _flush()
                continue

            if not line.strip():
                _flush()
                continue

            parts = line.split()          # handles tabs and spaces
            if len(parts) >= 4:
                tokens.append(parts[0])   # token is first column
                bio_tags.append(parts[-1])  # BIO tag is last column

    _flush()
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Deduplication
# ─────────────────────────────────────────────────────────────────────────────

def merge_duplicates(samples: List[Dict]) -> List[Dict]:
    """
    Within each (dataset, sentence) key, union all entity sets and keep one
    record.  Preserves first-seen order.
    """
    # key → ordered set of (cs, ce, type)
    key_to_ents: Dict[Tuple[str, str], Set[Tuple[int, int, str]]] = defaultdict(set)
    key_order: List[Tuple[str, str]] = []
    seen_keys: Set[Tuple[str, str]] = set()

    for s in samples:
        key = (s["dataset"], s["sentence"])
        if key not in seen_keys:
            key_order.append(key)
            seen_keys.add(key)
        for ent in s["entities"]:
            key_to_ents[key].add((ent["pos"][0], ent["pos"][1], ent["type"]))

    result: List[Dict] = []
    for dataset, text in key_order:
        key = (dataset, text)
        entities = [
            {"name": text[cs:ce], "pos": [cs, ce], "type": t}
            for cs, ce, t in sorted(key_to_ents[key])
        ]
        types = list(dict.fromkeys(e["type"] for e in entities))
        result.append({
            "sentence": text,
            "entities": entities,
            "dataset":  dataset,
            "types":    types,
        })
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def convert(
    data_dir: str,
    output: str,
    langs: Optional[List[str]],
    prefer_hash: bool,
    skip_multi: bool,
    merge_dups: bool,
):
    from datasets import Dataset, DatasetDict, Features, Value

    # Discover language directories
    all_lang_dirs = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])

    if langs:
        # Validate requested languages
        unknown = set(langs) - set(all_lang_dirs)
        if unknown:
            print(f"WARNING: unknown language dirs (ignored): {sorted(unknown)}")
        selected = [d for d in all_lang_dirs if d in set(langs)]
    else:
        selected = [
            d for d in all_lang_dirs
            if not (skip_multi and d in _DEFAULT_SKIP)
        ]

    print(f"Processing {len(selected)} language(s): {selected}\n")

    all_samples: List[Dict] = []
    stats: Dict[str, Dict] = {}

    for lang in selected:
        lang_dir = os.path.join(data_dir, lang)
        test_file = _find_test_file(lang_dir, prefer_hash)

        if test_file is None:
            print(f"  {lang:30s} SKIP — no test file found")
            continue

        file_label = ("hash" if prefer_hash and lang in _LANGS_WITH_HASH
                      else "main")
        samples = parse_conll_file(test_file, lang_name=lang)

        raw_count = len(samples)
        entity_count = sum(len(s["entities"]) for s in samples)

        if merge_dups:
            samples = merge_duplicates(samples)

        dedup_count = len(samples)
        all_samples.extend(samples)

        stats[lang] = {
            "file": os.path.basename(test_file),
            "file_type": file_label,
            "raw_sentences": raw_count,
            "after_dedup": dedup_count,
            "entities": entity_count,
        }
        dup_note = f"→ {dedup_count} after dedup" if merge_dups and dedup_count != raw_count else ""
        print(
            f"  {lang:30s} [{file_label}]  "
            f"{raw_count:>7} sentences  {entity_count:>7} entities  {dup_note}"
        )

    if not all_samples:
        print("ERROR: no samples collected.")
        sys.exit(1)

    print(f"\nTotal: {len(all_samples)} samples across {len(stats)} language(s)")

    # Build Arrow dataset — schema matches instruct_uie_ner exactly
    features = Features({
        "sentence": Value("string"),
        "entities": [{"name": Value("string"), "pos": [Value("int64")], "type": Value("string")}],
        "dataset":  Value("string"),
        "types":    [Value("string")],
    })

    dataset = Dataset.from_list(all_samples, features=features)
    ds_dict = DatasetDict({"test": dataset})
    ds_dict.save_to_disk(output)
    print(f"Saved → {output}")
    print(f"  Splits   : {list(ds_dict.keys())}")
    print(f"  Features : {list(ds_dict['test'].features.keys())}")


def main():
    p = argparse.ArgumentParser(
        description="Convert MultiCoNER2022 CoNLL → HuggingFace Arrow (evaluate.py compatible)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data_dir", default="/data/dataset/ner/multiconer2022",
        help="Root directory containing per-language subdirectories",
    )
    p.add_argument(
        "--output", default="/data/dataset/ner/multiconer2022_arrow",
        help="Where to save the Arrow DatasetDict",
    )
    p.add_argument(
        "--langs", nargs="+", default=None, metavar="LANG",
        help="Language dir names to include (default: all except MULTI_Multilingual). "
             "Example: EN-English DE-German ZH-Chinese",
    )
    p.add_argument(
        "--prefer_hash", action=argparse.BooleanOptionalAction, default=True,
        help="For BN/DE/EN: use hash-suffixed file (competition test set, ~18k EN) "
             "instead of the larger main test file. (default: on)",
    )
    p.add_argument(
        "--include_skipped", action="store_true", default=False,
        help="Include MULTI_Multilingual and MIX_Code_mixed (skipped by default)",
    )
    p.add_argument(
        "--no_merge_duplicates", action="store_true", default=False,
        help="Skip duplicate-sentence merging (not recommended when using LLM descriptions)",
    )
    args = p.parse_args()

    convert(
        data_dir=args.data_dir,
        output=args.output,
        langs=args.langs,
        prefer_hash=args.prefer_hash,
        skip_multi=not args.include_skipped,
        merge_dups=not args.no_merge_duplicates,
    )


if __name__ == "__main__":
    main()
