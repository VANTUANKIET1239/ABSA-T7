"""Audit raw comments without modifying the input CSV."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import unicodedata

from preprocess_cleanser_reviews import restore_missing_diacritics


def fold(text):
    return "".join(c for c in unicodedata.normalize("NFD", text.lower())
                   if unicodedata.category(c) != "Mn").replace("đ", "d")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    original = args.input.read_bytes()
    with args.input.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    tokens = [re.findall(r"[^\W\d_]+", unicodedata.normalize("NFC", row["comment"]).lower())
              for row in rows]
    frequencies = Counter(token for sentence in tokens for token in sentence)
    variants = defaultdict(Counter)
    for token, count in frequencies.items():
        if token != fold(token):
            variants[fold(token)][token] += count
    examples = defaultdict(list)
    candidates = {token for token in frequencies if token.isascii() and token in variants}
    row_counts = Counter()
    for record, (row, sentence) in enumerate(zip(rows, tokens), 2):
        for token in set(sentence) & candidates:
            row_counts[token] += 1
            if len(examples[token]) < 4:
                position = re.search(r"(?<!\w)" + token + r"(?!\w)", row["comment"], re.I)
                start = max(0, position.start() - 60) if position else 0
                examples[token].append({"source_record": record,
                                        "context": row["comment"][start:start + 200]})
    inventory = [{"token": token, "occurrences": frequencies[token], "review_count": row_counts[token],
                  "accented_variants_in_corpus": dict(variants[token].most_common()),
                  "examples": examples[token]}
                 for token in sorted(candidates, key=lambda x: (-frequencies[x], x))]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "candidate_tokens.json").write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {"source": str(args.input.resolve()), "source_sha256": hashlib.sha256(original).hexdigest(),
               "rows": len(rows), "unique_raw_tokens": len(frequencies),
               "potentially_ambiguous_ascii_tokens": len(inventory),
               "note": "Fold matches are candidates, not confirmed spelling errors."
               }
    pairs = Counter()
    pair_reviews = Counter()
    pair_examples = defaultdict(list)
    word_pairs = Counter()
    word_reviews = Counter()
    word_contexts = defaultdict(set)
    previews = []
    second_pass_changes = []
    for record, row in enumerate(rows, 2):
        changes = []
        restored = restore_missing_diacritics(row["comment"], changes=changes)
        if not changes:
            continue
        repeated = restore_missing_diacritics(restored)
        if repeated != restored:
            second_pass_changes.append(record)
        normalized_changes = [(" ".join(src.lower().split()), dest) for src, dest in changes]
        pairs.update(normalized_changes)
        pair_reviews.update(set(normalized_changes))
        review_words = set()
        for src, dest in normalized_changes:
            for source_word, target_word in zip(src.split(), dest.split()):
                if source_word != target_word:
                    pair = (source_word, target_word)
                    word_pairs[pair] += 1
                    review_words.add(pair)
                    word_contexts[pair].add(src + " → " + dest)
        word_reviews.update(review_words)
        for pair in set(normalized_changes):
            if len(pair_examples[pair]) < 3:
                pair_examples[pair].append(str(record))
        previews.append({"source_record": record, "comment": row["comment"],
                         "accent_restored_preview": restored,
                         "replacements": json.dumps(changes, ensure_ascii=False)})
    with (args.output_dir / "missing_diacritics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["raw_form", "replacement", "occurrences",
                                                    "review_count", "example_source_records"])
        writer.writeheader()
        writer.writerows({"raw_form": src, "replacement": dest, "occurrences": count,
                          "review_count": pair_reviews[src, dest],
                          "example_source_records": "; ".join(pair_examples[src, dest])}
                         for (src, dest), count in pairs.most_common())
    with (args.output_dir / "missing_diacritics_words.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["raw_word", "replacement", "occurrences_in_matched_contexts",
                                                    "review_count", "contexts"])
        writer.writeheader()
        writer.writerows({"raw_word": src, "replacement": dest,
                          "occurrences_in_matched_contexts": count,
                          "review_count": word_reviews[src, dest],
                          "contexts": "; ".join(sorted(word_contexts[src, dest])[:8])}
                         for (src, dest), count in word_pairs.most_common())
    with (args.output_dir / "review_changes_preview.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["source_record", "comment", "accent_restored_preview",
                                                    "replacements"])
        writer.writeheader()
        writer.writerows(previews)
    summary.update({"observed_replacement_pairs": len(pairs),
                    "observed_word_replacement_pairs": len(word_pairs),
                    "unique_raw_word_forms_in_replacements": len({src for src, dest in word_pairs}),
                    "word_replacement_occurrences": sum(word_pairs.values()),
                    "replacement_occurrences": sum(pairs.values()),
                    "reviews_with_accent_replacements": len(previews),
                    "source_unchanged": args.input.read_bytes() == original,
                    "non_idempotent_preview_records": second_pass_changes,
                    "method": "Manually selected domain phrases and words; exact token boundaries; "
                              "mixed accented/unaccented syllables allowed in phrases; reviewed partial "
                              "accent forms and explicit field-value context. Candidate inventory "
                              "is not an exhaustive semantic classification of spelling errors.",
                    "source_record_definition": "CSV record position including header (first data row=2), "
                                                "not physical line number.",
                    "preview_scope": "Only accent restoration; does not rerun teencode or VNCoreNLP. "
                                     "Existing cleaned_comment in source is not updated."})
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                                  encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for (src, dest), count in pairs.most_common(25):
        print(f"{src}\t{dest}\t{count}")


if __name__ == "__main__":
    main()
