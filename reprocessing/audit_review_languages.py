"""Screen raw reviews for English and export hash-guarded review lists.

Ignore Vietnamese UI labels, borrowed words and product names in language decisions."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import unicodedata

from seller_reply_rules import fold_with_offsets


ENGLISH_WORDS = set("""
good great nice excellent perfect amazing lovely wonderful horrible terrible bad worse worst
product products quality delivery delivered deliver package packaging packaged packing packed
received receive arrived arrive ordered order seller picture description described exactly
thank thanks recommend recommended happy disappointed purchase shopping experience online
cheap price prices expensive refund replacement missing wrong original authentic fake
skin sensitive dry drying mature gentle cleanser cleansers cleanse cleanses cleansing wash
face facial foaming foam acne breakouts prevents prevent future daily enough use useful
leaves leave matte finish gives deep helps controls production cooling sensation smooth soft
suitable fresh clean feeling looking radiant glowing formula types moisturizing effect deeply
pores clearer complexion easily removes makeup impurities dirt effectively hydrating
hyaluronic whitening brightening improves overall texture refreshed alcohol care non
delicate fragrance smell smells scented smell efficient products long term effects
beautiful bust cleavage enhances lifts soothing soothes redness natural effective
effective effective effective exellent efective cleancer gentel everething realy verry
seems love liked lovely super best before sealed sample kit oily water cream lotion
buy bought fast free gifts gift use using used tried try easy works expected
the this that these those it its is are was were be been being does doesn do don
and or but with without for from into of out off to too my me i im your you our
very so really really also definitely again everything well safely quickly
can could would should will have has had not no yes just like as than then now
an a all any each every in on at by about after before when where who which
first second third time week weeks month months years more less high low open
took back give split minimum consumption says sure repeat worth job made china
information introduced removing rich suffer someone able pretty smell late check aging lifting
""".split())
STRONG_WORDS = ENGLISH_WORDS - set("""
the this that these those it its is are was were be been being does doesn do don
and or but with without for from into of out off to too my me i im your you our
very so also again well can could would should will have has had not no yes
just like as than then now an a all any each every in on at by about after before
when where who which time more less skin care face facial wash cleanser cleansing
foam gel cream lotion makeup product products natural original fake free gift gifts
recommend recommended love super best mini sample kit oil non made china
""".split())
FUNCTION_WORDS = ENGLISH_WORDS - STRONG_WORDS
NEUTRAL_ONLY = re.compile(r"(?:ok(?:ay|e|ie|i)?[a-z]*|k+|h+|hi|ha|hic|bt|sp|srm|nsx|hsd|ml|g|date|shop|tiki|lazada|shopee|\d+)", re.I)
LABEL = re.compile(r"(?<!\w)(?:chat luong|ket cau|hieu qua|gia ca|thanh phan|cong dung|danh cho da|"
                   r"phu hop voi loai da|huong thom|mui huong|loi ich|meo|bao bi|kinh nghiem su dung|"
                   r"lam sach|tao bot|duong am|lam dep|do diu nhe|diu nhe|da kho|da mem min)\s*[:：]", re.I)
RATING_TITLE = re.compile(r"^\s*(?:(?:cuc ki|cuc ky|rat)(?: khong)? hai long|khong hai long|hai long|binh thuong|tot)\s*[.!:]\s*", re.I)


def content_for_language(text):
    text = unicodedata.normalize("NFC", text)
    folded, offsets = fold_with_offsets(text)
    spans = [(match.start(), match.end()) for match in LABEL.finditer(folded)]
    title = RATING_TITLE.match(folded)
    if title:
        spans.append((title.start(), title.end()))
    for start, end in sorted(spans, reverse=True):
        text = text[:offsets[start]] + " " + text[offsets[end - 1] + 1:]
    return text.strip()


def signals(text):
    content = content_for_language(text)
    tokens = re.findall(r"[^\W\d_]+", content.lower())
    strong = sorted(set(tokens) & STRONG_WORDS)
    function = sorted(set(tokens) & FUNCTION_WORDS)
    meaningful = [token for token in tokens if not NEUTRAL_ONLY.fullmatch(token)]
    latin_only = bool(meaningful) and all(token.isascii() for token in meaningful)
    candidate = bool(strong or latin_only or len(function) >= 3)
    return {"candidate": candidate, "content": content, "strong_words": strong,
            "function_words": function, "latin_only": latin_only}


def export(path, columns, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--decisions", type=Path)
    args = parser.parse_args()
    original = args.input.read_bytes()
    digest = hashlib.sha256(original).hexdigest()
    with args.input.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = list(reader.fieldnames)
        rows = list(reader)
    candidates = []
    for record, row in enumerate(rows, 2):
        info = signals(row["comment"])
        if info["candidate"]:
            candidates.append({"source_record": record, "platform": row["platform"],
                               "comment": row["comment"], **info})
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "candidates.json").write_text(json.dumps(candidates, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {"source": str(args.input.resolve()), "source_sha256": digest,
               "rows_scanned": len(rows), "candidate_records": len(candidates)}
    if args.decisions:
        manifest = json.loads(args.decisions.read_text(encoding="utf-8"))
        if manifest["source_sha256"] != digest:
            raise ValueError("Language decisions refer to a different source snapshot")
        decisions = {int(key): value for key, value in manifest["decisions"].items()}
        candidate_ids = {entry["source_record"] for entry in candidates}
        if set(decisions) != candidate_ids:
            raise ValueError("Every screened candidate must have a reviewed decision")
        groups = {"english": [], "english_short": [], "mixed": [], "english_other_language": [], "uncertain": []}
        counts = Counter()
        for record, row in enumerate(rows, 2):
            decision = decisions.get(record, {"status": "retained", "reason": "No English language signals"})
            status = decision["status"]
            counts[status] += 1
            if status == "retained":
                continue
            if status not in groups:
                raise ValueError(f"Unknown language decision: {status}")
            groups[status].append({**row, "language_source_record": record,
                                   "language_status": status, "language_reason": decision["reason"]})
        extra = ["language_source_record", "language_status", "language_reason"]
        english = groups["english"] + groups["english_short"]
        english.sort(key=lambda row: row["language_source_record"])
        export(out / "english_reviews.csv", columns + extra, english)
        export(out / "mixed_language_reviews.csv", columns + extra, groups["mixed"])
        export(out / "english_other_language_reviews.csv", columns + extra, groups["english_other_language"])
        export(out / "language_reviews_to_check.csv", columns + extra, groups["uncertain"])
        summary.update({"classification_counts": dict(counts), "english_total_including_short": len(english),
                        "english_by_platform": dict(Counter(row["platform"] for row in english)),
                        "definition": "English substantive content; Vietnamese field labels and generic rating "
                                      "titles do not count as Vietnamese review text. Single-word English "
                                      "ratings are separate. Vietnamese loanwords/product names alone are retained.",
                        "method": "Local lexical screening of all raw comments, followed by manual reading "
                                  "of every screened candidate; no trained language identification model.",
                        "source_record_definition": "1-based CSV record including header; first review=2"})
    if args.input.read_bytes() != original:
        raise ValueError("Source changed during audit")
    summary["source_unchanged"] = True
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
