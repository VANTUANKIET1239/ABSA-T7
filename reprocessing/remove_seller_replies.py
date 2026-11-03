"""Preview/apply removal of appended seller replies, with a verified backup."""

import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tempfile

from seller_reply_rules import strip_seller_reply, fold_with_offsets


def has_customer_content(text):
    value, _ = fold_with_offsets(text)
    value = re.sub(r"(?:chat luong|hieu qua|gia ca|thanh phan|bao bi|meo|huong thom|"
                   r"mui huong|loi ich|cong dung|lam sach|tao bot|duong am|ket cau|danh cho da)\s*[:：]", " ", value)
    # An isolated salutation or field label is not a customer review.
    value = re.sub(r"^\s*(?:xin )?chao\s*[,!. ]*\s*$", "", value)
    return bool(re.search(r"[^\W_]", value, re.UNICODE))


def read_csv(data):
    with io.StringIO(data.decode("utf-8-sig"), newline="") as stream:
        reader = csv.DictReader(stream)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    if not columns or len(columns) != len(set(columns)):
        raise ValueError("Invalid/duplicate CSV headers")
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("Malformed CSV row")
    return columns, rows


def write_csv(path, columns, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    source = args.input.resolve()
    original = source.read_bytes()
    digest = hashlib.sha256(original).hexdigest()
    if args.expected_sha256 and digest != args.expected_sha256:
        raise ValueError("Source differs from reviewed preview")
    columns, rows = read_csv(original)
    changes, removed, retained, changed_indices = [], [], [], []
    counts, platforms = Counter(), Counter()
    for index, row in enumerate(rows):
        customer, evidence = strip_seller_reply(row["comment"])
        if evidence is None:
            retained.append(row.copy())
            continue
        counts[evidence["rule"]] += 1
        platforms[row.get("platform", "")] += 1
        has_customer = has_customer_content(customer)
        edited = row.copy()
        edited["comment"] = customer
        if "review_text_length" in columns:
            edited["review_text_length"] = str(len(customer))
        audit = {"source_record": index + 2, "platform": row.get("platform", ""),
                 "brand": row.get("brand", ""), "rule": evidence["rule"],
                 "action": "strip_reply" if has_customer else "remove_seller_only_row",
                 "original_comment": row["comment"], "customer_comment": customer,
                 "removed_seller_reply": evidence["removed_text"],
                 "previous_cleaned_comment": row.get("cleaned_comment", ""),
                 "updated_cleaned_comment": ""}
        changes.append(audit)
        if has_customer:
            changed_indices.append((len(retained), len(changes) - 1))
            retained.append(edited)
        else:
            removed.append(row)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    backup = None
    regenerated = 0
    if args.apply and changes:
        from preprocess_cleanser_reviews import WordSegmenter, clean_and_tokenize, normalize_comment, to_topic_text
        segmenter = WordSegmenter(True)
        segmenter.setup()
        if segmenter.segmenter is None:
            raise RuntimeError("VNCoreNLP unavailable; source not modified")
        for index, audit_index in changed_indices:
            row = retained[index]
            normalized = normalize_comment(row["comment"])
            cleaned = clean_and_tokenize(row["comment"], segmenter)
            if "normalized_comment" in columns:
                row["normalized_comment"] = normalized
            if "cleaned_comment" in columns:
                row["cleaned_comment"] = cleaned
            if "topic_comment" in columns:
                row["topic_comment"] = to_topic_text(cleaned)
            changes[audit_index]["updated_cleaned_comment"] = cleaned
            regenerated += 1
        if segmenter._warned:
            raise RuntimeError("Segmentation failed; source not modified")
        # Verify the full source-to-result correspondence, including duplicate rows.
        expected = []
        mutable = {"comment", "review_text_length", "cleaned_comment", "normalized_comment", "topic_comment"}
        for row in rows:
            customer, evidence = strip_seller_reply(row["comment"])
            if evidence is not None and not has_customer_content(customer):
                continue
            saved = retained[len(expected)]
            if evidence is None:
                if saved != row:
                    raise ValueError("An unaffected record changed")
            else:
                if any(saved[key] != row[key] for key in columns if key not in mutable):
                    raise ValueError("Metadata changed")
                if saved["comment"] != customer:
                    raise ValueError("Customer prefix changed")
            expected.append(saved)
        if len(expected) != len(retained):
            raise ValueError("Row correspondence failed")
        staging = None
        try:
            with tempfile.NamedTemporaryFile(dir=source.parent, prefix="seller_cleanup_", suffix=".tmp",
                                             delete=False) as stream:
                staging = Path(stream.name)
            write_csv(staging, columns, retained)
            saved_bytes = staging.read_bytes()
            if read_csv(saved_bytes) != (columns, retained):
                raise ValueError("Staging verification failed")
            if source.read_bytes() != original:
                raise ValueError("Source changed while processing")
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            backup = source.with_name(f"{source.stem}.before_seller_reply_cleanup_{stamp}{source.suffix}")
            with backup.open("xb") as stream:
                stream.write(original)
                stream.flush()
                os.fsync(stream.fileno())
            if backup.read_bytes() != original or source.read_bytes() != original:
                raise ValueError("Backup/source verification failed")
            os.replace(staging, source)
            staging = None
            if source.read_bytes() != saved_bytes:
                raise ValueError(f"Post-write verification failed; backup: {backup}")
        finally:
            if staging is not None:
                staging.unlink(missing_ok=True)
    audit_columns = ["source_record", "platform", "brand", "rule", "action", "original_comment",
                     "customer_comment", "removed_seller_reply", "previous_cleaned_comment", "updated_cleaned_comment"]
    write_csv(out / "seller_reply_changes.csv", audit_columns, changes)
    write_csv(out / "removed_seller_only_reviews.csv", columns, removed)
    summary = {"source": str(source), "applied": args.apply,
               "rows_before": len(rows), "rows_with_seller_reply": len(changes),
               "replies_stripped_customer_kept": len(changed_indices),
               "seller_only_rows_removed": len(removed), "rows_after": len(retained),
               "columns": len(columns), "by_rule": dict(counts), "by_platform": dict(platforms),
               "cleaned_comments_regenerated": regenerated, "backup": str(backup) if backup else None,
               "source_sha256_before": digest,
               "source_sha256_after": hashlib.sha256(source.read_bytes()).hexdigest(),
               "known_reply_markers_remaining": sum(strip_seller_reply(row["comment"])[1] is not None
                                                    for row in retained),
               "metadata_and_customer_prefix_verified": bool(args.apply),
               "note": "Rule-based matching of reviewed seller templates; customer thanks to shop are retained. "
                       "Source positions refer to CSV records including the header, not physical lines."}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
