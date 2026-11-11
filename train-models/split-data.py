"""Split cleanser ABSA reviews 70/15/15, keeping repeated texts together.

Defaults to the labeled unique dataset. Preserve all original columns and labels.
Ratios apply to unique normalized text groups; no API or model is called.
"""
import argparse
import csv
import io
import math
import random
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Labeling"))
from label_distribution import parse_label

DEFAULT_INPUT = ROOT / "Labeling/data/unique_texts_gemini_labeled.csv"
DEFAULT_OUTPUT = ROOT / "train-models/data"


def text_key(text):
    return " ".join(unicodedata.normalize("NFC", text).lower().split())


def split_rows(rows, *, seed=42):
    groups = defaultdict(list)
    skipped = 0
    for record, row in enumerate(rows, 2):
        key = text_key(row["cleaned_comment"])
        if not key:
            skipped += 1
            continue
        if "absa_status" in row and row["absa_status"] != "done":
            raise ValueError(f"Record {record}: review is not marked done")
        try:
            # Empty labels on completed reviews are valid; never infer neutral.
            parse_label(row["label"])
        except ValueError as error:
            raise ValueError(f"Record {record}: {error}") from error
        groups[key].append(row)
    if len(groups) < 3:
        raise ValueError("At least 3 distinct nonempty review texts are required")
    for repeated in groups.values():
        labels = {frozenset(parse_label(r["label"])) for r in repeated}
        if len(labels) != 1:
            raise ValueError("Conflicting labels for repeated review text")
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    n_test = math.ceil(len(keys) * 0.15)
    n_val = math.ceil((len(keys) - n_test) * (15 / 85))
    if len(keys) - n_test - n_val < 1:
        raise ValueError("Too few text groups for a 70/15/15 split")
    assigned = {key: "test" if i < n_test else "val" if i < n_test + n_val else "train"
                for i, key in enumerate(keys)}
    result = {name: [] for name in ("train", "val", "test")}
    for row in rows:
        key = text_key(row["cleaned_comment"])
        if key:
            result[assigned[key]].append(row)
    key_sets = [{text_key(r["cleaned_comment"]) for r in part} for part in result.values()]
    if any(a & b for i, a in enumerate(key_sets) for b in key_sets[i + 1:]):
        raise RuntimeError("Review text leakage between splits")
    if sum(map(len, result.values())) + skipped != len(rows):
        raise RuntimeError("Split row counts do not match input")
    return result, skipped


def split_dataset(input_path=DEFAULT_INPUT, output_dir=DEFAULT_OUTPUT, *, seed=42):
    source, destination = Path(input_path).resolve(), Path(output_dir).resolve()
    original = source.read_bytes()
    reader = csv.DictReader(io.StringIO(original.decode("utf-8-sig"), newline=""))
    columns = reader.fieldnames or []
    rows = list(reader)
    if len(columns) != len(set(columns)) or not {"cleaned_comment", "label"} <= set(columns):
        raise ValueError("Input needs distinct columns including cleaned_comment and label")
    if any(None in r or any(v is None for v in r.values()) for r in rows):
        raise ValueError("Malformed CSV record")
    parts, skipped = split_rows(rows, seed=seed)
    paths = {name: destination / f"{name}_cleanser.csv" for name in parts}
    for path in paths.values():
        if path == source or path.exists():
            raise FileExistsError(f"Output already exists: {path}; choose another --output-dir")
    destination.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        for name, path in paths.items():
            with path.open("x", encoding="utf-8-sig", newline="") as stream:
                created.append(path)
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                writer.writerows(parts[name])
            with path.open(encoding="utf-8-sig", newline="") as stream:
                saved = csv.DictReader(stream)
                if saved.fieldnames != columns or list(saved) != parts[name]:
                    raise RuntimeError(f"Saved CSV verification failed: {path}")
        if source.read_bytes() != original:
            raise RuntimeError("Source changed during splitting")
    except Exception:
        for path in created:
            path.unlink()
        raise
    print(f"Input: {source}")
    print(f"Rows: {len(rows):,}; skipped empty texts: {skipped:,}; seed: {seed}")
    print("Random split by text group; labels are not stratified.")
    for name, part in parts.items():
        print(f"{name}: {len(part):,} rows -> {paths[name]}")
    return parts, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    split_dataset(args.input, args.output_dir, seed=args.seed)


if __name__ == "__main__":
    main()
