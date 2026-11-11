"""Copy reviewed ABSA labels to every mapped source review, without an API.

Preview: python reprocessing/sync_review_labels.py
Save:    python reprocessing/sync_review_labels.py --apply
"""

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from create_unique_text_dataset import normalize_text_key, text_group_id

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Labeling"))
from label_distribution import parse_label

SOURCE = ROOT / "crawl-data/data/combined_reviews_brand_filled_language_filtered.csv"
MAPPING = ROOT / "crawl-data/data/unique_texts/review_text_mapping.csv"
LABELED = ROOT / "Labeling/data/unique_texts_gemini_labeled.csv"
SUMMARY = MAPPING.with_name("deduplication_summary.json")
LABEL_COLUMNS = ["label", "absa_status"]


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read(raw):
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
    rows = list(reader)
    columns = reader.fieldnames
    assert columns and len(columns) == len(set(columns))
    assert all(None not in r and all(v is not None for v in r.values()) for r in rows)
    return columns, rows


def encode(columns, rows, original):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns,
                            lineterminator="\r\n" if b"\r\n" in original else "\n")
    writer.writeheader()
    writer.writerows(rows)
    return (b"\xef\xbb\xbf" if original.startswith(b"\xef\xbb\xbf") else b"") + stream.getvalue().encode("utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    before = {p: p.read_bytes() for p in (SOURCE, MAPPING, LABELED, SUMMARY)}
    source_cols, source = read(before[SOURCE])
    mapping_cols, mapping = read(before[MAPPING])
    _, labeled = read(before[LABELED])
    by_group = {r["text_group_id"]: r for r in labeled}
    assert len(by_group) == len(labeled)
    for r in labeled:
        assert r["absa_status"] == "done" and parse_label(r["label"])
        assert text_group_id(normalize_text_key(r["cleaned_comment"])) == r["text_group_id"]
    assert len(source) == len(mapping)
    base_cols = [c for c in source_cols if c not in LABEL_COLUMNS]
    counts = Counter()
    out_source, out_mapping = [], []
    for index, (s, m) in enumerate(zip(source, mapping), 1):
        assert m["source_row"] == str(index)
        assert all(s[c] == m[c] for c in base_cols), f"Mapping mismatch at {index}"
        key = normalize_text_key(s["cleaned_comment"])
        if key:
            group = text_group_id(key)
            assert m["text_group_id"] == group and m["text_status"] == "mapped"
            r = by_group[group]
            assert normalize_text_key(r["cleaned_comment"]) == key
            labels = {c: r[c] for c in LABEL_COLUMNS}
            counts[group] += 1
        else:
            assert not m["text_group_id"] and m["text_status"] == "empty_text"
            labels = {"label": "", "absa_status": "empty_text"}
        out_source.append({**s, **labels})
        out_mapping.append({**m, **labels})
    assert set(counts) == set(by_group)
    assert all(counts[g] == int(r["duplicate_count"]) for g, r in by_group.items())
    output = {}
    for path, cols, rows in [(SOURCE, source_cols, out_source), (MAPPING, mapping_cols, out_mapping)]:
        cols = cols + [c for c in LABEL_COLUMNS if c not in cols]
        output[path] = encode(cols, rows, before[path])
        saved_cols, saved = read(output[path])
        assert saved_cols == cols and saved == rows
    summary = json.loads(before[SUMMARY])
    summary["source_sha256"] = digest(output[SOURCE])
    summary["output_sha256"][MAPPING.name] = digest(output[MAPPING])
    summary["label_sync"] = {
        "labeled_unique_rows": len(labeled), "labeled_source_rows": sum(counts.values()),
        "empty_text_rows": len(source) - sum(counts.values()),
        "log": "label_sync_log.json",
    }
    output[SUMMARY] = (json.dumps(summary, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "source_rows": len(source), **summary["label_sync"],
              "label_columns": LABEL_COLUMNS,
              "before_sha256": {str(p.relative_to(ROOT)): digest(b) for p,b in before.items()},
              "after_sha256": {str(p.relative_to(ROOT)): digest(b) for p,b in output.items()}}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not args.apply:
        return
    changed = {p:b for p,b in output.items() if b != before[p]}
    if not changed:
        print("Already synchronized; no files changed.")
        return
    backup = ROOT / "label_sync_backup.zip"
    if backup.exists():
        raise FileExistsError(f"Backup already exists: {backup}")
    for p, raw in before.items():
        assert p.read_bytes() == raw, f"Input changed: {p}"
    with zipfile.ZipFile(backup, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for p, raw in before.items():
            z.writestr(p.relative_to(ROOT).as_posix(), raw)
        z.writestr("manifest.json", json.dumps(report, ensure_ascii=False, indent=2))
    with zipfile.ZipFile(backup) as z:
        assert z.testzip() is None
        for p, raw in before.items():
            assert z.read(p.relative_to(ROOT).as_posix()) == raw
    staged = {}
    try:
        for p, raw in changed.items():
            temp = p.with_name(p.name + ".label-sync.tmp")
            with temp.open("xb") as stream:
                stream.write(raw)
            staged[p] = temp
        replaced = []
        try:
            for p, temp in staged.items():
                os.replace(temp, p)
                replaced.append(p)
        except Exception:
            for p in reversed(replaced):
                p.write_bytes(before[p])
            raise
        for p, raw in output.items():
            assert p.read_bytes() == raw
        assert LABELED.read_bytes() == before[LABELED]
        (ROOT / "label_sync_log.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    finally:
        for temp in staged.values():
            if temp.exists():
                temp.unlink()
    print("Saved and verified; original source values and row order preserved.")


if __name__ == "__main__":
    main()
