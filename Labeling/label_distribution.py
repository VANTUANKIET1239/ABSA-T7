"""Count ABSA aspect/sentiment pairs without modifying the input CSV.

Usage:
    python Labeling/label_distribution.py
    python Labeling/label_distribution.py --input path/to/reviews.csv
    python Labeling/label_distribution.py --output path/to/distribution.csv

Each input row counts once, regardless of duplicate_count. A review can count
in multiple aspects and multiple sentiments within an aspect. Duplicate pairs
within one review count only once. Empty labels do not count as neutral.
"""

import argparse
import csv
import re
from collections import Counter
from pathlib import Path


DEFAULT_INPUT = Path(__file__).resolve().parent / "data/unique_texts_gemini_labeled.csv"
ASPECTS = (
    "CleansingEffectiveness", "HydrationSkinComfort", "SkinCompatibilitySafety",
    "TextureFoaming", "Fragrance", "IngredientsFormulation", "Packaging",
    "PriceValue", "ShopService", "Delivery", "General",
)
SENTIMENTS = ("positive", "neutral", "negative")
PAIR_PATTERN = re.compile(r"\{([A-Za-z]+)#(positive|neutral|negative)\}")


def parse_label(label):
    """Validate labels and return distinct aspect/sentiment pairs."""
    if not label.strip():
        return set()
    pairs = set()
    for part in label.strip().split(";"):
        match = PAIR_PATTERN.fullmatch(part)
        if not match or match.group(1) not in ASPECTS:
            raise ValueError(f"Invalid ABSA pair: {part!r}")
        pairs.add(match.groups())
    return pairs


def summarize(input_path):
    counts = Counter()
    mentions = Counter()
    statuses = Counter()
    total_rows = empty_rows = 0
    with input_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if "gemini_label" not in (reader.fieldnames or []):
            raise ValueError("CSV must contain a gemini_label column")
        for record_number, row in enumerate(reader, 1):
            try:
                pairs = parse_label(row.get("gemini_label") or "")
            except ValueError as exc:
                raise ValueError(f"CSV record {record_number}: {exc}") from exc
            counts.update(pairs)
            mentions.update({aspect for aspect, _ in pairs})
            statuses[row.get("absa_status", "(missing)")] += 1
            total_rows += 1
            empty_rows += not pairs
    table = [
        {"aspect": aspect,
         **{sentiment: counts[aspect, sentiment] for sentiment in SENTIMENTS},
         "total_pairs": sum(counts[aspect, sentiment] for sentiment in SENTIMENTS),
         "reviews": mentions[aspect]}
        for aspect in ASPECTS
    ]
    return table, total_rows, empty_rows, statuses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, help="Optional CSV summary output")
    args = parser.parse_args()
    if args.output and args.output.resolve() == args.input.resolve():
        parser.error("Output must differ from input")
    table, total, empty, statuses = summarize(args.input)
    print(f"Reviews: {total:,} | Empty labels: {empty:,}")
    print("Status: " + ", ".join(f"{key}={value:,}" for key, value in statuses.items()))
    print()
    print(f"{'Aspect':<25} {'positive':>9} {'neutral':>9} {'negative':>9} {'Pairs':>9} {'Reviews':>9}")
    print("-" * 75)
    for row in table:
        print(f"{row['aspect']:<25} " + " ".join(
            f"{row[key]:>9,}" for key in (*SENTIMENTS, "total_pairs", "reviews")))
    print("-" * 75)
    print(f"{'TOTAL':<25} " + " ".join(
        f"{sum(row[key] for row in table):>9,}"
        for key in (*SENTIMENTS, "total_pairs")))
    print("\nPairs = aspect/sentiment pairs; Reviews = rows mentioning the aspect.")
    print("Counts are unweighted; empty labels are excluded, not neutral.")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=table[0].keys())
            writer.writeheader()
            writer.writerows(table)
        print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
