"""Screen cleanser reviews and export candidates; missing keywords require manual review."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path


def fold(text):
    text = unicodedata.normalize("NFKC", text).lower().replace("đ", "d")
    text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", text.replace("_", " ")).strip()


SELLER = re.compile(
    r"eucerin men chao ban|(?:rohto(?: mentholatum)?(?: viet nam| vietnam)?|hada labo(?: viet nam)?) (?:xin chao|rat vui|rat xin loi)|"
    r"chao (?:ban|anh/chi|quy khach)[,! .]*(?=.{0,90}(?:cam on|rat tiec|xin loi))|"
    r"xin chao quy khach|"
    r"cam on (?:ban|quy khach)(?: vi| da| rat| .{0,20})? (?:tin tuong|chon mua|ung ho|chia se|trai nghiem|danh tinh yeu|lua chon)|"
    r"co (?:mem )?(?:xin loi ban|rat xin loi|rat vui khi)|"
    r"co (?:mem )?(?:rat vui vi|chao ban|cam on ban)|"
    r"cam on ban da (?:mua hang tai|tin yeu|dong hanh)|"
    r"shop (?:rat )?(?:cam on ban|xin loi ban|hi vong)|"
    r"chao ban[,! ]+(?:bioderma|cam on|cam on|shop|gian hang)", re.I)
FIELD_LABEL = re.compile(
    r"\b(?:chat luong|ket cau|hieu qua|cong dung|mui huong|huong thom|danh cho da|loi ich|"
    r"meo|bao bi|kinh nghiem su dung|lam sach|tao bot|duong am|gia ca|thanh phan|lam dep|"
    r"do diu nhe|do sach|giam viem|bot)\s*[:：]", re.I)
PRODUCT = re.compile(
    r"\b(?:srm|srmat|san pham|spham|sp|sua rua|sua rua|sua r[ua]|rua mat|rua sach|"
    r"lam sach|sach|cleanser|cleansing|face wash|da mat|da dau|da kho|da mun|da nhay|"
    r"da hon hop|mun|kich ung|di ung|kho cang|mem min|min mang|mem da|min da|"
    r"am da|duong am|cap am|giu am|da am|cang rit|ngua|cham chich|rat da|cay mat|"
    r"ba nhon|lo chan long|kiem dau|nhon|phu hop|hop da|hop voi da|"
    r"mui|huong|thom|bot|gel|ket cau|long|dac|nong|ph|bha|aha|niacinamide|"
    r"thanh phan|hieu qua|cong dung|kem|my pham|skincare|skin care|duong da|"
    r"chua dung|chua xai|chua sai|su dung|dung thu|dung tot|dung on|dung ok|"
    r"xai|sai thu|sai tot|sai ok|cam giac|chan ai|chai thu|tuyp thu|"
    r"khong hop|ko hop|k hop|bi kho|bi rat|bi do|len mun|noi mun|giam mun|"
    r"an toan|lanh tinh|dau thua|tay trang|tinh chat|dung tich|ml|gram|"
    r"mem mai|muot|min|kho da|da sang|trang da|san chac|da mem|da rat|"
    r"tham|deo|soi ba nhon|te bao chet|se khit|sung|do da|phuc hoi)\b", re.I)
PURCHASE = re.compile(
    r"\b(?:giao|ship|shipper|shop|hang|mua|dat|gia|re|dat hang|dong goi|goi hang|"
    r"bao bi|chai|lo|tuyp|hop|tem|seal|niem phong|nap|voi|bom|chinh hang|hang gia|"
    r"fake|auth|gia tien|sale|voucher|qua tang|tang qua|khuyen mai|uy tin|mo ta|"
    r"nhan du|thieu|mop|be|vo|doi tra|hoan tien|cam on|tiki|lazada|shopee|"
    r"phuc vu|tu van|ung ho|cam ket|nhiet tinh|dong goi|chat luong)\b", re.I)
GENERIC_PHRASES = re.compile(
    r"\b(?:cuc ki|cuc ky|rat|khong|ko|k|hoi|tam|qua|sieu|luon|lam|nha|nhe|"
    r"hai long|tot|tottt|totttt|ok|oke|okee|okie|oki|okay|good|nice|great|excellent|"
    r"on|on ap|ung|tuyet voi|tuyet|dep|thich|chat luong|hi|hihi|haha|kkk|"
    r"cam on|thank you|thanks|thank|tks|thank shop|5 sao|5sao|5 star|5 stars|"
    r"khong co|nhan xet|binh thuong|moi nguoi|minh|a|ak|ah|ha|na|ne|nhen|"
    r"het|chua|biet|gi|ca|nua|danh gia|hihi|y|xuat sac|bad|te|kem)\b", re.I)
SPAM = {
    "clipboard": re.compile(r"gboard|bang nho tam|clipboard"),
    "bank_message": re.compile(r"\botp\b|vcb digibank|xac thuc gd|chuyen khoan nhanh qua so tk"),
    "film": re.compile(r"\bphim\b|dien vien|iqiyi|ngon tinh|co trang|nhat niem quan son"),
    "education": re.compile(r"\bielts\b|training cham diem|human.s future development|robots are very important"),
    "poem_story": re.compile(r"bai tho|vao giuong di ngu|to[iy] tung dat bao thuc|loi bai hat"),
    "finance_news": re.compile(r"monetary aggregate|policymakers|economists|blockchain|ung dung goi xe|chiet khau tu tai xe"),
    "unrelated_ad": re.compile(r"lyly fruits|roja elixir|rojaelixir|benh vien mat trung uong|duc thuy tinh|giac mac|moc khoa|shoppe?\.ee|zalo.*(?:tu van|gia si)"),
    "food": re.compile(r"mon an|cha ca lang|thit bo|nuoc mam|cong thuc nau|trai cay|hoa qua tuoi|dau bep"),
    "keyboard": re.compile(r"qwerty|asdfgh|zxcvbn|abcdefgh"),
    "sms": re.compile(r"viettel tb|goi 5g10|tai khoan vien thong|bam phim 19|nap the du day|goi \*098"),
    "lyrics_chatter": re.compile(r"under the moon|my heart still whispers|money, wealth, love|song:|composed:|mot loai thuc vat|moi loai thuc vat|loai buom dem"),
}


def analyze(text, brands):
    value = fold(text)
    match = SELLER.search(value)
    customer = value[:match.start()].strip() if match else value
    content = FIELD_LABEL.sub(" ", customer)
    # Image/video disclaimers do not make an otherwise valid review off-topic.
    content = re.sub(r"(?:hinh anh|hinh|anh|video|vid|clip).{0,65}?(?:nhan xu|lay xu|kiem xu|khong lien quan|minh hoa)", " ", content)
    content = re.sub(r"\s+", " ", content).strip()
    product_matches = set(PRODUCT.findall(content))
    purchase_matches = set(PURCHASE.findall(content))
    product = bool(product_matches)
    purchase = bool(purchase_matches)
    brand = any(re.search(r"(?<!\w)" + re.escape(b) + r"(?!\w)", content) for b in brands if len(b) > 2)
    spam = [name for name, pattern in SPAM.items() if pattern.search(content)]
    residual = GENERIC_PHRASES.sub(" ", content)
    generic = not re.search(r"[a-z0-9]", residual)
    tokens = re.findall(r"[a-z]+", content)
    long_tokens = [token for token in tokens if len(token) >= 14]
    repeated = bool(re.search(r"([a-z])\1{8,}", content))
    if spam:
        category = "spam_candidate"
    elif generic:
        category = "generic"
    elif repeated or len(long_tokens) >= 2 or any(len(token) >= 25 for token in long_tokens):
        category = "noise_candidate"
    elif len(tokens) >= 22 and len(product_matches | purchase_matches) <= 2:
        category = "low_evidence_candidate"
    elif product or purchase or brand:
        category = "related"
    else:
        category = "unmatched_candidate"
    return {"category": category, "signals": ",".join(spam), "customer_text": customer,
            "product_signal": product, "purchase_signal": purchase, "brand_signal": brand,
            "generic": generic, "long_token_count": len(long_tokens), "repeated_letters": repeated}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="crawl-data/data/combined_reviews_brand_filled.csv")
    parser.add_argument("--output-dir", default="crawl-data/analysis_output/relevance_filter")
    parser.add_argument("--decisions", help="Human-reviewed candidate decisions, keyed by comment SHA256")
    args = parser.parse_args()
    source = Path(args.input).resolve()
    source_bytes = source.read_bytes()
    with source.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = list(reader.fieldnames)
        rows = list(reader)
    brands = {fold(r["brand"]) for r in rows if r["brand"].strip()}
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    candidates = []
    categories = Counter()
    cache = {}
    for row_number, row in enumerate(rows, 2):
        text = row["comment"]
        if text not in cache:
            cache[text] = analyze(text, brands)
        decision = cache[text]
        categories[decision["category"]] += 1
        if decision["category"].endswith("candidate"):
            candidates.append({"source_row": row_number, "text_hash": hashlib.sha256(text.encode()).hexdigest(),
                               "comment": text, **decision})
    with (out / "candidates.jsonl").open("w", encoding="utf-8") as stream:
        for candidate in candidates:
            stream.write(json.dumps(candidate, ensure_ascii=False) + "\n")
    summary = {"rows": len(rows), "unique_comments": len(cache), "categories": dict(categories),
                      "candidate_rows": len(candidates),
                      "unique_candidate_comments": len({c["text_hash"] for c in candidates})}
    if args.decisions:
        manifest = json.loads(Path(args.decisions).read_text(encoding="utf-8"))
        source_hash = hashlib.sha256(source_bytes).hexdigest()
        if manifest["source_sha256"] != source_hash:
            raise RuntimeError("Reviewed decisions refer to a different version of the source CSV")
        reviewed = set(manifest["reviewed_comment_hashes"])
        overrides = {d["text_hash"]: d for d in manifest["decisions"]}
        if len(overrides) != len(manifest["decisions"]):
            raise RuntimeError("Conflicting duplicate manual decisions")
        irrelevant, to_check = [], []
        outcome_counts = Counter()
        reason_counts = Counter()
        for number, row in enumerate(rows, 2):
            key = hashlib.sha256(row["comment"].encode()).hexdigest()
            decision = overrides.get(key)
            if not decision and cache[row["comment"]]["category"].endswith("candidate") and key not in reviewed:
                decision = {"status": "uncertain", "category": "unreviewed_candidate",
                            "reason": "Ứng viên mới chưa được đọc đối chiếu; chưa xếp là lạc đề."}
            status = decision["status"] if decision else "retained"
            outcome_counts[status] += 1
            if decision:
                result = {**row, "filter_source_record": str(number),
                          "filter_status": status, "filter_category": decision["category"],
                          "filter_reason": decision["reason"]}
                if status == "irrelevant":
                    irrelevant.append(result)
                    reason_counts[decision["category"]] += 1
                elif status in {"mixed", "uncertain"}:
                    to_check.append(result)
                else:
                    raise ValueError(f"Unknown decision status: {status}")
        output_columns = columns + ["filter_source_record", "filter_status", "filter_category", "filter_reason"]
        for name, selected in (("irrelevant_reviews.csv", irrelevant), ("reviews_to_check.csv", to_check)):
            destination = out / name
            with destination.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=output_columns)
                writer.writeheader()
                writer.writerows(selected)
            with destination.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                if reader.fieldnames != output_columns:
                    raise RuntimeError("Exported column names do not match")
                saved = list(reader)
            if len(saved) != len(selected):
                raise RuntimeError("Exported record count does not match")
            for item in saved:
                original = rows[int(item["filter_source_record"]) - 2]
                if any(item[column] != original[column] for column in columns):
                    raise RuntimeError("An original field changed in the extracted CSV")
        if sum(outcome_counts.values()) != len(rows):
            raise RuntimeError("Outcome counts do not reconcile with source")
        summary.update({"source_file": str(source), "source_sha256": source_hash,
                        "classification_counts": dict(outcome_counts),
                        "irrelevant_by_category": dict(reason_counts),
                        "irrelevant_percentage": round(100 * len(irrelevant) / len(rows), 3),
                        "reviewed_unique_comment_hashes": len(reviewed),
                        "source_unchanged": True, "original_columns_preserved_in_exports": True,
                        "scope": manifest["scope"],
                        "method": "Local rule screening of every raw comment, followed by manual candidate review and additional topic scans. Not a trained semantic relevance classifier."})
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if source.read_bytes() != source_bytes:
        raise RuntimeError("Source changed during screening")


if __name__ == "__main__":
    main()
