"""Remove reviewed CSV records after validating their original positions and values."""

import argparse
import csv
import hashlib
import io
import json
import os
from datetime import datetime
from pathlib import Path
import tempfile


def read_csv_bytes(data):
    with io.StringIO(data.decode("utf-8-sig"), newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fields = reader.fieldnames
    if not fields or len(set(fields)) != len(fields):
        raise ValueError("Missing or duplicate CSV column names")
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("Malformed CSV record")
    return fields, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("lists", nargs="+", type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    original = source.read_bytes()
    fields, rows = read_csv_bytes(original)
    removals = set()
    counts = {}
    for flagged_path in args.lists:
        flagged_path = flagged_path.resolve()
        if flagged_path == source:
            raise ValueError("A removal list cannot be the source file")
        flagged_fields, flagged_rows = read_csv_bytes(flagged_path.read_bytes())
        if flagged_fields[:len(fields)] != fields:
            raise ValueError(f"Original columns differ: {flagged_path}")
        if "filter_source_record" not in flagged_fields:
            raise ValueError(f"Missing original record position: {flagged_path}")
        indices = set()
        for flagged in flagged_rows:
            index = int(flagged["filter_source_record"]) - 2
            if not 0 <= index < len(rows):
                raise ValueError(f"Invalid record position: {flagged_path}")
            if any(rows[index][field] != flagged[field] for field in fields):
                raise ValueError(f"Source record changed at record {index + 2}")
            indices.add(index)
        counts[str(flagged_path)] = {
            "listed_rows": len(flagged_rows),
            "unique_source_rows": len(indices),
            "overlap_with_prior_lists": len(indices & removals),
        }
        removals.update(indices)
    if not removals:
        raise ValueError("No matching records to remove")
    retained = [row for index, row in enumerate(rows) if index not in removals]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup = source.with_name(f"{source.stem}.before_relevance_removal_{stamp}{source.suffix}")
    report = args.lists[0].resolve().parent / f"removal_summary_{stamp}.json"
    staged_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8-sig", newline="", suffix=".csv",
            prefix="relevance_removal_", dir=source.parent, delete=False,
        ) as staged:
            staged_path = Path(staged.name)
            writer = csv.DictWriter(staged, fieldnames=fields)
            writer.writeheader()
            writer.writerows(retained)
            staged.flush()
            os.fsync(staged.fileno())
        expected = staged_path.read_bytes()
        if read_csv_bytes(expected) != (fields, retained):
            raise ValueError("Staging verification failed")
        if source.read_bytes() != original:
            raise ValueError("Source changed during processing; no update applied")
        with backup.open("xb") as stream:
            stream.write(original)
            stream.flush()
            os.fsync(stream.fileno())
        if backup.read_bytes() != original:
            raise ValueError("Backup verification failed; no update applied")
        if source.read_bytes() != original:
            raise ValueError("Source changed before replacement; no update applied")
        os.replace(staged_path, source)
        staged_path = None
        actual = source.read_bytes()
        if actual != expected or read_csv_bytes(actual) != (fields, retained):
            raise ValueError(f"Post-write verification failed; backup: {backup}")
        summary = {
            "source": str(source), "backup": str(backup),
            "rows_before": len(rows), "rows_removed": len(removals),
            "rows_after": len(retained), "columns": len(fields),
            "removal_lists": counts,
            "sha256_before": hashlib.sha256(original).hexdigest(),
            "sha256_after": hashlib.sha256(actual).hexdigest(),
            "backup_matches_original": True,
            "retained_values_and_order_verified": True,
        }
        with report.open("x", encoding="utf-8") as stream:
            json.dump(summary, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        print(json.dumps({**summary, "report": str(report)}, ensure_ascii=False, indent=2))
    finally:
        if staged_path is not None:
            staged_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
