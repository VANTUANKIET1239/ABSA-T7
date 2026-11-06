"""Identify appended merchant replies while retaining the customer's raw prefix."""

import re
import unicodedata


def fold_with_offsets(text):
    """Accent/whitespace insensitive matching with exact raw-character offsets."""
    folded, offsets = [], []
    for index, original in enumerate(text):
        for char in unicodedata.normalize("NFD", original.lower()):
            if unicodedata.category(char) == "Mn":
                continue
            char = "d" if char == "đ" else char
            if char.isspace() or char == "_":
                char = " "
                if folded and folded[-1] == " ":
                    continue
            folded.append(char)
            offsets.append(index)
    return "".join(folded), offsets


_BRAND = (
    r"(?:rohto(?: mentholatum)?|hada labo|eucerin|cetaphil|cerave|bioderma|"
    r"himalaya|sunlady(?: shop)?|omi|hoang\s?lan(?: health & beauty)?|"
    r"shopmyphamnga mockba|co(?: mem)?)"
    r"(?: (?:viet nam|vietnam|official store|lazada mall|lazada|shop|skincare))?"
)
_PERSON = r"(?:quy khach(?: hang)?|khach hang|ban|anh\s*/\s*chi|anh chi)"
_THANK = r"(?:(?:xin |rat )?(?:chan thanh )?cam on)"
_ACTION = (
    r"(?:da )?(?:luon )?(?:tin tuong|tin yeu|tin dung|chon mua|lua chon|"
    r"ung ho|chia se|trai nghiem|yeu thich|quan tam|phan hoi|danh gia|"
    r"danh (?:tinh yeu|su tin yeu|thoi gian)|hai long|dong hanh|mua hang tai|"
    r"bao dung|yeu thuong|su dung do gia dung elmich)"
)

# These are seller speech acts, not generic sentiment/product keywords.
_RULES = [
    ("brand_reply", rf"{_BRAND} (?:xin chao|men chao|chao ban|rat (?:vui|xin loi|hanh phuc|tiec|tran trong)|"
     rf"that hanh phuc|tran trong|xin loi|xin ghi nhan|xin duoc gui|{_THANK})"),
    ("shop_reply", rf"(?:shop|gian hang) (?:xin chao|rat (?:tiec|xin loi)|{_THANK}|"
     r"xin loi ban|hi vong|hy vong)"),
    ("greeting_reply", r"(?:xin )?chao (?:ban|quy khach|anh\s*/\s*chi|anh chi)"
     r"[,! .]*(?=.{0,130}(?:cam on|rat tiec|xin loi|eucerin|bioderma|cerave|cetaphil))"),
    ("customer_greeting", r"xin chao quy khach"),
    ("thanks_for_review", rf"{_THANK} {_PERSON} (?:{_ACTION}|vi (?:{_ACTION}|su hai long|(?:nhung )?trai nghiem hai long)|"
     r"trong thoi gian vua qua|rat nhieu vi danh gia)"),
    ("thanks_for_feedback", rf"{_THANK} (?:nhung )?phan hoi (?:tich cuc cua ban|cua ban)"),
    ("co_mem_reply", r"co(?: mem)? (?:xin rut kinh nghiem|xin phep nhan tin|"
     r"thuc su rat tiec|thanh that xin loi|chan thanh cam on|that hanh phuc|tran trong)"),
    ("team_reply", r"chung minh rat (?:vui|hanh phuc|biet on)"
     r"(?=.{0,400}(?:co mem|san pham cua co|lua chon co|dong hanh cung co|san pham khac cua co))"),
    ("co_mem_appreciation", r"that (?:vui|hanh phuc)(?: va tu hao)? (?:khi|vi)"
     r"(?=.{0,300}(?:san pham cua co|tin tuong co|ung ho co))"),
    ("shop_delivery_thanks", r"shop va don vi van chuyen xin cam on quy khach"),
    ("service_followup", r"quy khach dung het ung ho shop|shop co de kem hdsd quy khach"),
    ("seller_ad_only", r"^\s*cam ket mang den da dang san pham chinh hang"),
    ("service_reply", r"(?:rat )?vui duoc phuc vu (?:anh\s*/\s*chi|quy khach|quy khac|a\b)"),
    ("apology_reply", r"rat xin loi ban\s*[,!. ]+\s*rat cam on ban da thong cam"),
    ("delivery_auto_reply", r"(?:kinh gui quy khach|don hang cua ban da duoc giao thanh cong)"
     r"(?=.{0,300}(?:danh gia 5 sao|vui long lien he|san pham cua chung toi))"),
]
SELLER_REPLY_PATTERNS = [(name, re.compile(r"(?<!\w)(?:" + pattern + r")", re.I))
                         for name, pattern in _RULES]


def strip_seller_reply(text):
    """Return the exact customer prefix and evidence for an appended merchant reply.

    Match known reply patterns; the caller handles an empty customer prefix."""
    folded, offsets = fold_with_offsets(text)
    candidates = []
    for name, pattern in SELLER_REPLY_PATTERNS:
        match = pattern.search(folded)
        if match:
            candidates.append((match.start(), name, match.group()))
    if not candidates:
        return text, None
    position, name, marker = min(candidates, key=lambda item: item[0])
    start = offsets[position]
    # Include only a known sender immediately preceding the matched reply formula.
    prefix = folded[:position]
    sender = re.search(r"(?:" + _BRAND + r"|shop|gian hang)\s+(?:xin )?(?:chan thanh )?$", prefix)
    if sender:
        start = offsets[sender.start()]
    retained = text[:start].rstrip()
    return retained, {"rule": name, "marker": marker, "reply_start": start,
                      "removed_text": text[len(retained):]}
