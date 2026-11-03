"""Tiền xử lý review SỮA RỬA MẶT (Shopee, Lazada, Tiki).

Giữ raw comment + metadata; tạo normalized_comment, cleaned_comment và topic_comment.
Không loại từ phủ định / mô tả tác dụng phụ trong cleaned_comment.

Chạy:
    python preprocess_cleanser_reviews.py
    python preprocess_cleanser_reviews.py --input data/shopee_test_reviews.csv data/lazada_cleanser_reviews.csv data/tiki_cleanser_reviews.csv
    python preprocess_cleanser_reviews.py --no-vncorenlp
    python preprocess_cleanser_reviews.py --input crawl-data/data/combined_reviews_brand_filled.csv --append-text-only

Cài đặt:
    pip install pandas tqdm demoji py_vncorenlp
    # VNCoreNLP cần Java; model được tải lần đầu nếu chưa có.
"""
from __future__ import annotations

import argparse
import csv
import html
import os
import re
import numbers
import shutil
import tempfile
import unicodedata
import warnings
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from urllib.request import urlopen

import pandas as pd
from tqdm import tqdm

try:
    from .seller_reply_rules import strip_seller_reply
except ImportError:  # Chạy trực tiếp script từ command line.
    from seller_reply_rules import strip_seller_reply

try:
    import demoji
except ImportError:
    demoji = None

try:
    import py_vncorenlp
except ImportError:
    py_vncorenlp = None


# ============================ CẤU HÌNH ============================
DEFAULT_INPUT_FILES = [
    "data/shopee_test_reviews.csv",    # đổi nếu file Shopee của bạn tên khác
    "data/lazada_cleanser_reviews.csv",
    "data/tiki_cleanser_reviews.csv",
]
DEFAULT_OUTPUT_FILE = "data/cleanser_reviews_preprocessed.csv"
MIN_CLEANED_LENGTH = 6            # ít nhất 6 ký tự sau chuẩn hoá
FILTER_VIETNAMESE = False         # tránh loại review tiếng Việt không dấu
USE_VNCORENLP = True
MODEL_DIRECTORY = "vncorenlp_model"
MODEL_BASE_URL = "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/"

# Các trường từ 3 crawler được giữ nguyên, nếu có.
COLUMN_ALIASES = {
    "review_text": "comment", "reviewtext": "comment", "text": "comment",
    "review_date": "date", "reviewdate": "date", "created_at": "date",
    "productname": "product_name", "producturl": "product_url",
    "review_star": "rating", "stars": "rating",
}

# Chỉ giữ ánh xạ an toàn trong ngữ cảnh sữa rửa mặt.
# Tránh các sửa sai của code cũ như 'in' -> '', 'long' -> '', 'phạm' -> ''.
CUSTOM_TEENCODE = {
    "srm": "sữa rửa mặt", "sữa rm": "sữa rửa mặt", "sua rua mat": "sữa rửa mặt",
    "rua mat": "rửa mặt", "srmat": "sữa rửa mặt",
    "sp": "sản phẩm", "spham": "sản phẩm",
    "k": "không", "ko": "không", "khg": "không", "khum": "không",
    "khôngg": "không", "khong": "không", "hok": "không", "hông": "không",
    "dc": "được", "đc": "được", "đk": "được",
    "sd": "sử dụng", "su dung": "sử dụng",
    "xài": "dùng", "sài": "dùng", "xai": "dùng",
    "mng": "mọi người", "mn": "mọi người",
    "tks": "cảm ơn", "thank": "cảm ơn", "thanks": "cảm ơn",
    "nhìu": "nhiều", "nhiu": "nhiều", "rùi": "rồi", "lun": "luôn",
    "cx": "cũng", "nma": "nhưng mà", "ncl": "nói chung là",
    "hdsd": "hướng dẫn sử dụng", "hsd": "hạn sử dụng",
    "auth": "chính hãng", "fake": "hàng giả",
    "shop": "cửa hàng", "sóp": "cửa hàng",
    "ship": "giao hàng", "síp": "giao hàng", "shipper": "người giao hàng",
    "packaging": "đóng gói", "dong goi": "đóng gói", "đong goi": "đóng gói",
    "chất lg": "chất lượng", "chat luong": "chất lượng",
    "giao hang": "giao hàng", "giao nhanh": "giao nhanh",
    "mun": "mụn", "mun an": "mụn ẩn", "mun viem": "mụn viêm",
    "kich ung": "kích ứng", "kich thich": "kích thích",
    "nhay cam": "nhạy cảm", "diu nhe": "dịu nhẹ",
    "kho cang": "khô căng", "kho da": "khô da",
    "lam sach": "làm sạch", "sach se": "sạch sẽ", "tao bot": "tạo bọt",
    "kiem dau": "kiềm dầu", "duong am": "dưỡng ẩm", "cap am": "cấp ẩm",
    "bao bi": "bao bì", "voi bom": "vòi bơm", "nap chai": "nắp chai",
    "do ph": "độ ph", "p h": "ph",
    "mui": "mùi", "muif": "mùi", "thom": "thơm", "thơmm": "thơm",
    "de chiu": "dễ chịu", "min mang": "mịn màng",
    "tot": "tốt", "good": "tốt", "oke": "ok", "okee": "ok", "okie": "ok",
    "bt": "bình thường", "binh thuong": "bình thường",
    "rcm": "gợi ý", "recommend": "gợi ý",
    "nhej": "nhé", "raats": "rất", "raat": "rất",
}

# Các tên thương hiệu cần được so khớp riêng, KHÔNG giới hạn ở hãng thuần chay.
# Có thể bổ sung theo tập sản phẩm thực tế của nghiên cứu.
BRAND_ALIASES = {
    "la roche posay": "La Roche-Posay", "la roche-posay": "La Roche-Posay",
    "lrp": "La Roche-Posay", "cerave": "CeraVe", "cetaphil": "Cetaphil",
    "eucerin": "Eucerin", "bioderma": "Bioderma", "simple": "Simple",
    "senka": "Senka", "hada labo": "Hada Labo", "hadalabo": "Hada Labo",
    "cosrx": "COSRX", "innisfree": "Innisfree", "skin1004": "SKIN1004",
    "skin 1004": "SKIN1004", "some by mi": "Some By Mi", "anua": "Anua",
    "round lab": "Round Lab", "neutrogena": "Neutrogena",
    "garnier": "Garnier", "nivea": "Nivea", "biore": "Bioré",
    "ponds": "Pond's", "pond's": "Pond's", "acnes": "Acnes",
    "himalaya": "Himalaya", "cocoon": "Cocoon", "klairs": "Klairs",
    "dear klairs": "Klairs", "the face shop": "The Face Shop",
    "the body shop": "The Body Shop", "kiehls": "Kiehl's",
    "kiehl's": "Kiehl's", "vichy": "Vichy", "svr": "SVR",
    "avene": "Avène", "avène": "Avène", "dove": "Dove",
    "naruko": "Naruko", "purito": "Purito", "isntree": "Isntree",
    "rovectin": "Rovectin", "dr g": "Dr.G", "dr.g": "Dr.G",
    "huxley": "Huxley", "axis y": "AXIS-Y", "axis-y": "AXIS-Y",
    "ohui": "OHUI", "olay": "Olay", "sulwhasoo": "Sulwhasoo",
    "beplain": "Beplain", "torriden": "Torriden", "heimish": "Heimish",
    "physiogel": "Physiogel", "freeplus": "Freeplus", "skinceuticals": "SkinCeuticals",
    "d program": "d program", "senka perfect whip": "Senka",
}

# Tập trung vào thành phần, kết cấu, khả năng làm sạch, tình trạng da và trải nghiệm mua.
PHRASE_UNDERSCORE_DICT = {
    # loại da và biểu hiện
    "da nhạy cảm": "da_nhạy_cảm", "da dầu mụn": "da_dầu_mụn",
    "da hỗn hợp": "da_hỗn_hợp", "da thiên dầu": "da_thiên_dầu",
    "da thiên khô": "da_thiên_khô", "da dầu": "da_dầu", "da khô": "da_khô",
    "da mụn": "da_mụn", "da thường": "da_thường",
    "mụn đầu đen": "mụn_đầu_đen", "mụn đầu trắng": "mụn_đầu_trắng",
    "mụn ẩn": "mụn_ẩn", "mụn viêm": "mụn_viêm", "nổi mụn": "nổi_mụn",
    "lên mụn": "lên_mụn", "đẩy mụn": "đẩy_mụn",
    "lỗ chân lông": "lỗ_chân_lông", "bã nhờn": "bã_nhờn",
    "dầu thừa": "dầu_thừa", "bóng dầu": "bóng_dầu",
    # hiệu quả làm sạch & dưỡng da
    "sữa rửa mặt": "sữa_rửa_mặt", "gel rửa mặt": "gel_rửa_mặt",
    "bọt rửa mặt": "bọt_rửa_mặt", "rửa mặt": "rửa_mặt",
    "làm sạch sâu": "làm_sạch_sâu", "làm sạch": "làm_sạch",
    "sạch da": "sạch_da", "sạch sâu": "sạch_sâu",
    "tẩy tế bào chết": "tẩy_tế_bào_chết", "tế bào chết": "tế_bào_chết",
    "kiềm dầu": "kiềm_dầu", "kiểm soát dầu": "kiểm_soát_dầu",
    "dưỡng ẩm": "dưỡng_ẩm", "cấp ẩm": "cấp_ẩm",
    "giữ ẩm": "giữ_ẩm", "cân bằng độ ph": "cân_bằng_độ_ph",
    "độ ph": "độ_ph", "dịu nhẹ": "dịu_nhẹ", "không khô căng": "không_khô_căng",
    "khô căng": "khô_căng", "căng rít": "căng_rít",
    "căng da": "căng_da", "khô da": "khô_da",
    "mềm mịn": "mềm_mịn", "mịn màng": "mịn_màng",
    "thông thoáng": "thông_thoáng", "khô thoáng": "khô_thoáng",
    "châm chích": "châm_chích", "kích ứng": "kích_ứng",
    "đỏ da": "đỏ_da", "ngứa da": "ngứa_da", "rát da": "rát_da",
    "nóng rát": "nóng_rát", "cay mắt": "cay_mắt",
    "không kích ứng": "không_kích_ứng", "không gây mụn": "không_gây_mụn",
    "không sạch": "không_sạch", "không hợp": "không_hợp",
    # kết cấu, mùi hương và thành phần
    "tạo bọt": "tạo_bọt", "ít bọt": "ít_bọt", "nhiều bọt": "nhiều_bọt",
    "bọt mịn": "bọt_mịn", "dạng gel": "dạng_gel", "dạng kem": "dạng_kem",
    "kết cấu": "kết_cấu", "dễ rửa": "dễ_rửa", "khó rửa": "khó_rửa",
    "không mùi": "không_mùi", "mùi hương": "mùi_hương",
    "mùi thơm": "mùi_thơm", "mùi hắc": "mùi_hắc",
    "hương liệu": "hương_liệu", "mùi hóa chất": "mùi_hóa_chất",
    "axit salicylic": "axit_salicylic", "salicylic acid": "salicylic_acid",
    "hyaluronic acid": "hyaluronic_acid", "niacinamide": "niacinamide",
    "tràm trà": "tràm_trà", "rau má": "rau_má",
    # giá, bao bì và dịch vụ
    "giá thành": "giá_thành", "giá cả": "giá_cả",
    "đáng tiền": "đáng_tiền", "giá rẻ": "giá_rẻ", "giá cao": "giá_cao",
    "mã giảm giá": "mã_giảm_giá", "giảm giá": "giảm_giá",
    "đóng gói": "đóng_gói", "bao bì": "bao_bì", "vòi bơm": "vòi_bơm",
    "nắp chai": "nắp_chai", "chai lọ": "chai_lọ",
    "niêm phong": "niêm_phong", "tem chống giả": "tem_chống_giả",
    "hạn sử dụng": "hạn_sử_dụng", "hàng chính hãng": "hàng_chính_hãng",
    "giao hàng": "giao_hàng", "giao nhanh": "giao_nhanh",
    "giao chậm": "giao_chậm", "vận chuyển": "vận_chuyển",
    "chăm sóc khách hàng": "chăm_sóc_khách_hàng",
    "đúng mô tả": "đúng_mô_tả", "sản phẩm giả": "sản_phẩm_giả",
}

# Dành riêng cho topic model; KHÔNG áp dụng lên cleaned_comment (ABSA).
TOPIC_STOPWORDS = frozenset({
    "là", "và", "của", "đã", "đang", "rồi", "này", "kia", "đó", "đây",
    "mình", "tôi", "em", "chị", "anh", "bạn", "nha", "nhé", "ạ",
    "haha", "hihi", "kkk", "thì",
})

# Cụm được giữ nguyên dấu gạch dưới để biểu diễn khía cạnh hoàn chỉnh.
PHRASES_SORTED = sorted(PHRASE_UNDERSCORE_DICT.items(), key=lambda x: len(x[0]), reverse=True)

# Khôi phục dấu theo cụm trước khi mở rộng teencode / phân đoạn từ.
# Chấp nhận từng tiếng đúng dấu hoặc không dấu, và một số dạng thiếu dấu
# một phần đã rà riêng; KHÔNG
# bỏ dấu toàn bộ input rồi đoán lại (vd. 'rát da' phải khác 'rất tốt').
# Các cụm bổ sung được chọn từ ngữ cảnh của tập combined_reviews_brand_filled.
MISSING_DIACRITICS_PHRASES = tuple(dict.fromkeys([
    *(phrase for phrase in PHRASE_UNDERSCORE_DICT if phrase not in {"đỏ da", "dễ rửa"}),
    "sản phẩm", "sử dụng", "chất lượng", "chất gel", "chất kem",
    "hiệu quả", "công dụng", "thành phần", "lợi ích", "làm đẹp",
    "kinh nghiệm sử dụng", "độ dịu nhẹ", "độ ẩm", "giữ độ ẩm",
    "làn da", "da mặt", "mặt nạ", "làm sáng da", "trắng da",
    "sáng da", "da sáng", "da mềm", "da mịn", "da sạch", "da ẩm",
    "ẩm mịn", "ẩm da", "ẩm mềm", "mềm da", "mịn da", "mềm mại",
    "mềm mượt", "mướt da", "mịn sạch", "sạch mịn", "sạch mụn",
    "dầu mụn", "sạch dầu", "giảm dầu", "giảm nhờn", "dầu nhờn",
    "da nhờn", "nhờn rít", "khô rít", "rát da", "da rát",
    "ngứa da", "da ngứa", "nổi nhiều mụn", "nhiều mụn", "giảm mụn",
    "ngừa mụn", "trị mụn", "mụn viêm", "mụn thâm", "giảm thâm",
    "dị ứng", "không gây", "gây mụn", "gây khô", "gây rát", "bị rát",
    "bị khô", "bị mụn", "bị ngứa", "bị đỏ", "bị kích ứng",
    "nước sạch", "rửa sạch", "sạch mặt", "trên mặt", "trên da",
    "rửa xong", "sau khi rửa", "dễ dùng", "dễ lấy", "dễ dàng",
    "dễ tạo bọt", "tạo nhiều bọt", "bọt nhiều", "bọt ít", "bọt mềm",
    "vừa phải", "vừa đủ", "đủ làm sạch", "hương thơm", "thơm nhẹ",
    "hương dịu", "mùi dịu", "nồng gắt", "hương liệu", "mùi nghệ",
    "sữa bò", "thư giãn", "phù hợp", "hợp da", "hợp lý", "lành tính",
    "an toàn", "mọi loại da", "loại da", "dành cho da", "dạng gel",
    "dạng bọt", "da liễu", "bác sĩ", "bác sỹ", "dung tích",
    "thiết kế", "chai nhựa", "bụi bẩn", "bẩn bụi", "vỡ chai",
    "bóp chai", "méo móp", "chống sốc", "bọc chống sốc", "sơ sài",
    "cẩn thận", "chắc chắn", "kỹ lưỡng", "kĩ lưỡng", "gọn gàng",
    "nhanh chóng", "giao chậm", "giao lâu", "hàng chính hãng",
    "chính hãng", "hàng giả", "hàng gia công", "giao đúng hàng", "giao đúng sản phẩm", "đúng mẫu",
    "nhận đúng hàng", "nhận đúng sản phẩm", "giao hàng đúng sản phẩm",
    "đúng số lượng", "đúng chủng loại", "khách hàng", "đặt hàng",
    "nhận hàng", "nhận được", "đơn hàng", "mua hàng", "hàng giao",
    "giá hợp lý", "giá tốt", "giá rẻ", "giá cả", "giá tiền",
    "quà tặng", "tặng quà", "được tặng", "tặng kèm", "khuyến mãi",
    "khuyến mại", "ủng hộ", "hỗ trợ", "nhân viên", "nhiệt tình",
    "thân thiện", "dễ thương", "hài lòng", "yên tâm", "uy tín",
    "thương hiệu", "giới thiệu", "quảng cáo", "trải nghiệm", "cảm giác",
    "cảm thấy", "cảm ơn", "cám ơn", "hy vọng", "hi vọng", "đánh giá",
    "hình ảnh", "minh họa", "minh hoạ", "liên quan", "mang tính",
    "thông thoáng", "thời gian", "tình trạng", "khắc phục",
    "lần đầu", "lần sau", "lần này", "lần trước", "lần nữa", "nhiều lần",
    "hằng ngày", "hàng ngày", "buổi sáng", "buổi tối", "sáng tối",
    "tiếp tục", "mua tiếp", "mua lại", "quay lại", "trở lại", "lâu dài",
    "nói chung", "mọi người", "như thế nào", "như mô tả", "mô tả",
    "có thể", "không rõ", "chưa biết", "chưa dùng", "chưa sử dụng",
    "chưa có", "dùng thử", "dùng tốt", "dùng thích", "dùng ổn",
    "dùng ok", "dùng được", "dùng quen", "dùng hết", "dùng lâu",
    "dùng sữa rửa mặt", "dùng để rửa mặt", "để rửa mặt", "dùng sản phẩm để",
    "dùng sản phẩm này", "mình dùng", "mình thấy", "mình mua", "mình thích",
    "tôi dùng", "tôi thấy", "bao nhiêu", "nên mua", "nên dùng",
    "nên thử", "sẽ mua", "sẽ dùng", "thấy hợp", "thấy sạch", "thấy mịn",
    "rất tốt", "rất sạch", "rất thích", "rất dịu", "rất nhẹ", "rất thơm",
    "rất đẹp", "rất hợp", "rất ok", "rất nhanh", "rất hài lòng",
    "rất đáng tiền", "cực kì", "cực kỳ", "khá tốt", "khá thích",
    "khá hợp", "khá to", "tạm được", "tốt lắm", "thích lắm",
    "sạch lắm", "thơm lắm", "hơi lâu", "hơi khô", "hơi bị",
]))

# Từ đơn ít nhập nhằng trong các ngữ cảnh đã rà. Không ánh xạ đơn lẻ
# da/dung/mat/dau/rat/chat/hang/ban/la/on/it: chúng có nhiều cách hiểu,
# hoặc là từ Việt/Anh hợp lệ. 'dược sĩ', 'xác minh', 'tẩy trang' giữ nguyên.
MISSING_DIACRITICS_WORDS = {
    "khong": "không", "thom": "thơm", "tot": "tốt", "tôt": "tốt",
    "mun": "mụn", "diu": "dịu", "tuyp": "tuýp", "thich": "thích",
    "nhieu": "nhiều", "luon": "luôn", "giup": "giúp", "sieu": "siêu",
    "dep": "đẹp", "mem": "mềm",
}

# Dạng thiếu một phần dấu đã kiểm tra theo cụm, không sinh mọi tổ hợp dấu:
# sinh tự do sẽ đổi sai 'hàng ngày' thành 'hằng ngày', 'dược' thành 'được'.
MISSING_DIACRITICS_PARTIAL_PHRASES = {
    "sản phâm": "sản phẩm", "sản phảm": "sản phẩm", "san phâm": "sản phẩm",
    "chất lương": "chất lượng", "chât lượng": "chất lượng",
    "chát lượng": "chất lượng", "chát luong": "chất lượng",
    "chất luợng": "chất lượng", "chất luọng": "chất lượng",
    "chất lưọng": "chất lượng", "chất luơng": "chất lượng",
    "cẩn thân": "cẩn thận", "cẩn thạn": "cẩn thận", "cân thận": "cẩn thận",
    "cản thận": "cẩn thận", "cản thạn": "cẩn thận",
    "sữa rưa mặt": "sữa rửa mặt", "sữa rủa mặt": "sữa rửa mặt",
    "sữa rửa măt": "sữa rửa mặt", "rửa măt": "rửa mặt",
    "rưa mặt": "rửa mặt", "rủa mặt": "rửa mặt", "sủ dụng": "sử dụng",
    "bọc chống sóc": "bọc chống sốc", "chống sóc": "chống sốc",
    "chống sôc": "chống sốc", "bị kích ưng": "bị kích ứng",
    "dầu nhơn": "dầu nhờn", "duỡng ẩm": "dưỡng ẩm", "mùi huơng": "mùi hương",
}


def _missing_accent_word_pattern(word: str) -> str:
    """Một tiếng đúng dấu hoặc mất toàn bộ dấu, không sửa dấu đã có."""
    ascii_word = "".join(char for char in unicodedata.normalize("NFD", word)
                         if unicodedata.category(char) != "Mn").replace("đ", "d")
    return "(?:" + "|".join(re.escape(form) for form in sorted({word, ascii_word})) + ")"


def _compile_missing_accent_phrases():
    phrases = sorted(MISSING_DIACRITICS_PHRASES,
                     key=lambda phrase: (len(phrase.split()), len(phrase)), reverse=True)
    patterns = []
    targets = {}
    for index, phrase in enumerate(phrases):
        name = f"accent_{index}"
        body = r"\s+".join(_missing_accent_word_pattern(word) for word in phrase.split())
        patterns.append(f"(?P<{name}>{body})")
        targets[name] = phrase
    for index, (source, target) in enumerate(MISSING_DIACRITICS_PARTIAL_PHRASES.items()):
        name = f"partial_{index}"
        body = r"\s+".join(re.escape(word) for word in source.split())
        patterns.insert(0, f"(?P<{name}>{body})")
        targets[name] = target
    return re.compile(r"(?<!\w)(?:" + "|".join(patterns) + r")(?!\w)", re.I), targets


_MISSING_ACCENT_PHRASE_PATTERN, _MISSING_ACCENT_TARGETS = _compile_missing_accent_phrases()
_MISSING_ACCENT_WORD_PATTERN = re.compile(
    r"(?<!\w)(?:" + "|".join(map(re.escape, MISSING_DIACRITICS_WORDS)) + r")(?!\w)", re.I)

# Giá trị ngắn chỉ rõ nghĩa khi nằm trong một trường review tiếng Việt.
# Không dùng các ánh xạ này cho từ đơn ở vị trí bất kỳ trong câu.
MISSING_DIACRITICS_FIELD_VALUES = {
    "làm sạch": {"sach": "sạch", "on": "ổn", "min": "mịn"},
    "công dụng": {"sach": "sạch", "on": "ổn"},
    "lợi ích": {"sach": "sạch", "on": "ổn"},
    "tạo bọt": {"bot": "bọt", "min": "mịn", "on": "ổn", "sach": "sạch"},
    "bọt": {"bot": "bọt", "min": "mịn", "on": "ổn"},
    "hương thơm": {"nhe": "nhẹ", "on": "ổn"},
    "mùi hương": {"nhe": "nhẹ", "on": "ổn"},
    "độ dịu nhẹ": {"nhe": "nhẹ", "on": "ổn", "min": "mịn"},
    "dịu nhẹ": {"nhe": "nhẹ", "on": "ổn"},
    "dưỡng ẩm": {"am": "ẩm", "on": "ổn"},
}
_REVIEW_LABELS = "|".join(map(re.escape, sorted({
    *MISSING_DIACRITICS_FIELD_VALUES, "bao bì", "kết cấu", "làm đẹp", "mẹo",
    "kinh nghiệm sử dụng", "dành cho da", "hiệu quả", "giá cả", "thành phần",
}, key=len, reverse=True)))
_MISSING_ACCENT_FIELD_PATTERN = re.compile(
    r"(?<!\w)(?P<prefix>(?P<label>" + _REVIEW_LABELS + r")\s*:\s*)"
    r"(?P<value>sach|on|min|bot|nhe|am)(?!\w)"
    r"(?=\s*(?:$|[\r\n,.;!?]|(?:" + _REVIEW_LABELS + r")\s*:))", re.I)


def restore_missing_diacritics(text: str, *, changes: list | None = None) -> str:
    """Khôi phục dấu có chọn lọc; changes ghi (cụm gốc, cụm thay thế) để audit.

    Không phải mô hình phục hồi dấu toàn câu. Từ đa nghĩa chỉ sửa khi nằm
    trong cụm đã chọn, không dùng rating hay nội dung tích cực để đoán dấu.
    """
    text = unicodedata.normalize("NFC", text)

    def replace_phrase(match):
        source = match.group()
        target = _MISSING_ACCENT_TARGETS[match.lastgroup]
        # 'sạch sau khi rửa', 'làm sạch sau đó' mô tả thời điểm, không phải 'sâu'.
        if (target.endswith("sâu") and source.lower().split()[-1] == "sau"
                and re.match(r"\s+(?:khi|đó|do|này|nay|rửa|rua|dùng|dung|sử|su|mỗi|moi|một|mot|vài|vai|\d+)\b",
                             match.string[match.end():], re.I)):
            return source
        # 'hàng gia công' không có nghĩa là 'hàng giả'. Giữ cả trường hợp
        # khách dán lặp câu thiếu khoảng trắng: 'côngHàng gia công...'.
        if target == "hàng giả" and re.match(r"\s+(?:công|cong)", match.string[match.end():], re.I):
            return source
        if " ".join(source.lower().split()) == target:
            return source
        if changes is not None:
            changes.append((source, target))
        return target

    def replace_word(match):
        source = match.group()
        target = MISSING_DIACRITICS_WORDS[source.lower()]
        if changes is not None:
            changes.append((source, target))
        return target

    def replace_field(match):
        source = match["value"]
        target = MISSING_DIACRITICS_FIELD_VALUES.get(match["label"].lower(), {}).get(source.lower())
        if target is None:
            return match.group()
        if changes is not None:
            changes.append((source, target))
        return match["prefix"] + target

    text = _MISSING_ACCENT_PHRASE_PATTERN.sub(replace_phrase, text)
    text = _MISSING_ACCENT_WORD_PATTERN.sub(replace_word, text)
    return _MISSING_ACCENT_FIELD_PATTERN.sub(replace_field, text)


# ======================== CÁC HÀM HỖ TRỢ ==========================
def resolve_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    # Ưu tiên CWD (PyCharm thường đặt ở thư mục root của project).
    candidates = [Path.cwd() / p, Path(__file__).resolve().parent / p,
                  Path(__file__).resolve().parent.parent / p]
    return next((x for x in candidates if x.exists()), candidates[0])


def load_teencode_dict(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    result = {}
    with path.open(encoding="utf-8-sig") as file:
        for line in file:
            parts = line.strip().split("\t", maxsplit=1)
            if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                result[parts[0].strip().lower()] = parts[1].strip().lower()
    return result


def make_teencode_mapping() -> dict[str, str]:
    # Các từ riêng do người dùng quản lý có thể đặt trong data/teencode.txt.
    # Các thay thế nguy hiểm của từ điển legacy không được áp dụng nữa.
    extras = load_teencode_dict(resolve_path("data/teencode.txt"))
    blocked = {"in", "long", "pham", "bi", "su", "mat", "r", "date", "mask", "moi", "dung"}
    extras = {k: v for k, v in extras.items() if k not in blocked and len(k) >= 2}
    return {**extras, **CUSTOM_TEENCODE}


def compile_mapping(mapping: dict[str, str]):
    compiled = []
    for src, target in sorted(mapping.items(), key=lambda kv: len(kv[0]), reverse=True):
        # Lookaround xử lý từ có dấu, ngoặc và cụm từ an toàn hơn \b đơn thuần.
        pat = re.compile(r"(?<!\w)" + re.escape(src) + r"(?!\w)", re.IGNORECASE)
        compiled.append((pat, target))
    return compiled


_TEENCODE_PATTERNS = None
_PHRASE_PATTERNS = compile_mapping(PHRASE_UNDERSCORE_DICT)


def normalize_teencode(text: str) -> str:
    global _TEENCODE_PATTERNS
    if _TEENCODE_PATTERNS is None:
        _TEENCODE_PATTERNS = compile_mapping(make_teencode_mapping())
    for pattern, replacement in _TEENCODE_PATTERNS:
        text = pattern.sub(lambda _: replacement, text)
    return text


def remove_coin_hunting_phrases(text: str) -> str:
    """Chỉ xoá câu nhận xu cụ thể; không xoá cả phần review còn lại."""
    patterns = [
        r"(?:hình ảnh|hình|ảnh|video|clip)(?:\s*(?:và|\/|\+)\s*(?:video|hình ảnh))?"
        r"\s*(?:chỉ\s*)?(?:mang\s*(?:tính\s*chất|tính|tc)\s*)?"
        r"(?:minh\s*họa|minh\s*hoạ|tượng\s*trưng|tham\s*khảo)?"
        r"\s*(?:chỉ\s*)?(?:để\s*)?(?:nhận|kiếm|lấy)\s*xu+",
        r"(?:đăng|up)\s*(?:ảnh|hình|video)\s*(?:để\s*)?(?:nhận|lấy)\s*xu+",
    ]
    for pattern in patterns:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip(" .,;:-")


def remove_emoji(text: str) -> str:
    if demoji is not None:
        return demoji.replace(text, " ")
    # fallback khi không cài demoji; raw comment vẫn luôn được giữ nguyên.
    return re.sub(r"[\U0001F000-\U0001FAFF\U00002600-\U000027BF]", " ", text)


def remove_repeated_characters(text: str) -> str:
    # 'đẹpppp' -> 'đẹp'; không thay mọi kí tự lặp đôi (vd 'ee' trong tên brand).
    return re.sub(r"([a-zà-ỹđ])\1{2,}", r"\1", text, flags=re.IGNORECASE)


def normalize_comment(raw: object) -> str:
    """Text đọc được cho ABSA: vẫn giữ phủ định, dấu câu và chữ số (pH, giá...)."""
    if raw is None or pd.isna(raw):
        return ""
    customer_text, _ = strip_seller_reply(str(raw))
    text = unicodedata.normalize("NFC", html.unescape(customer_text)).lower()
    text = re.sub(r"<[^>]*>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = remove_emoji(text)
    text = remove_coin_hunting_phrases(text)
    text = remove_repeated_characters(text)
    text = restore_missing_diacritics(text)
    text = normalize_teencode(text)
    # Giữ cả số và dấu .,!? để không đổi sắc thái phủ định; làm sạch dấu lạ.
    text = re.sub(r"[^\w\s,.!?%:/+\-]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def apply_phrase_underscore(text: str) -> str:
    for pattern, underscored in _PHRASE_PATTERNS:
        text = pattern.sub(lambda _: underscored, text)
    return re.sub(r"\s*_\s*", "_", text)


class WordSegmenter:
    def __init__(self, enabled: bool):
        self.segmenter = None
        self.enabled = enabled
        self._warned = False

    def setup(self) -> None:
        if not self.enabled:
            print("[*] VnCoreNLP tắt theo cấu hình, cleaned_comment không được phân đoạn từ.")
            return
        if py_vncorenlp is None:
            warnings.warn("Chưa cài py_vncorenlp: dùng text chưa word segment. "
                          "Chạy pip install py_vncorenlp để có phân đoạn từ.")
            return
        model_dir = resolve_path(MODEL_DIRECTORY)
        model_dir.mkdir(parents=True, exist_ok=True)
        try:
            # Chỉ tải tài nguyên wseg cần dùng; không phụ thuộc lệnh wget trên Windows.
            resources = ("models/wordsegmenter/vi-vocab",
                         "models/wordsegmenter/wordsegmenter.rdr", "VnCoreNLP-1.2.jar")
            for resource in resources:
                destination = model_dir / resource
                if destination.is_file() and destination.stat().st_size:
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                print(f"[*] Tải tài nguyên VNCoreNLP: {resource}")
                download = destination.with_name(destination.name + ".download")
                try:
                    with urlopen(MODEL_BASE_URL + resource, timeout=120) as response, download.open("wb") as stream:
                        shutil.copyfileobj(response, stream)
                    os.replace(download, destination)
                finally:
                    download.unlink(missing_ok=True)
            self.segmenter = py_vncorenlp.VnCoreNLP(
                annotators=["wseg"], save_dir=str(model_dir)
            )
            print("[+] VNCoreNLP sẵn sàng.")
        except Exception as exc:
            warnings.warn(f"Không khởi tạo được VNCoreNLP ({exc}). "
                          "Tiếp tục xử lý nhưng chưa word segment.")

    def tokenize(self, text: str) -> str:
        if not text:
            return ""
        prepared = apply_phrase_underscore(text)
        # Phân đoạn từ có thể bị ảnh hưởng bởi dấu câu trong review.
        if self.segmenter is not None:
            try:
                segments = self.segmenter.word_segment(prepared)
                prepared = " ".join(segments) if isinstance(segments, list) else str(segments)
            except Exception as exc:
                if not self._warned:
                    warnings.warn(f"Lỗi word_segment ({exc}); dùng text trước phân đoạn.")
                    self._warned = True
        prepared = re.sub(r"\s*_\s*", "_", prepared)
        return re.sub(r"\s+", " ", prepared).strip()


def to_topic_text(cleaned_comment: str) -> str:
    return " ".join(
        word for word in cleaned_comment.split()
        if word.strip(".,!? ") not in TOPIC_STOPWORDS
    )


def clean_and_tokenize(text: object, segmenter: WordSegmenter | None = None) -> str:
    """Tên hàm tương thích ý tưởng pipeline cũ; dùng VnCoreNLP nếu được truyền vào."""
    normalized = normalize_comment(text)
    return (segmenter or WordSegmenter(False)).tokenize(normalized)


def normalize_url(value: object) -> str:
    if pd.isna(value) or not str(value).strip():
        return ""
    url = str(value).strip()
    try:
        parts = urlsplit(url)
        return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), "", ""))
    except ValueError:
        return url


def extract_brand(product_name: object) -> str:
    if pd.isna(product_name):
        return "other"
    name = str(product_name).lower()
    for candidate, brand in sorted(BRAND_ALIASES.items(), key=lambda x: len(x[0]), reverse=True):
        if re.search(r"(?<!\w)" + re.escape(candidate) + r"(?!\w)", name):
            return brand
    return "other"


def coalesce_brand(row: pd.Series) -> str:
    val = row.get("brand", "")
    if pd.notna(val) and str(val).strip().lower() not in {"", "nan", "none", "n/a", "other", "unknown"}:
        return str(val).strip()
    return extract_brand(row.get("product_name", ""))


def is_vietnamese(text: object) -> bool:
    """Heuristic nhẹ, tuỳ chọn (mặc định KHÔNG loại theo ngôn ngữ)."""
    if pd.isna(text) or not str(text).strip():
        return False
    s = str(text).lower()
    if re.search(r"[ăâđêôơưàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ]", s):
        return True
    # Không dấu vẫn có thể là tiếng Việt.
    return bool(re.search(r"\b(khong|ko|sua|rua|mat|da|mun|giao|hang|thom|tot|dung|sach)\b", s))


def parse_price_vnd(price: object):
    """Trả int hoặc pd.NA. Nếu là khoảng giá thì không tự chọn đầu khoảng."""
    if pd.isna(price) or not str(price).strip():
        return pd.NA
    if isinstance(price, numbers.Real):
        return int(price) if float(price).is_integer() else pd.NA
    s = str(price).strip().lower()
    # Không tự lấy cận dưới khi trang hiển thị khoảng giá các biến thể.
    if re.search(r"(?:\d[\d.,]*\s*(?:₫|đ|vnd)?\s*[-–—~]\s*(?:₫|đ|vnd)?\s*\d)", s) or re.search(r"\d\s*đến\s*\d", s):
        return pd.NA
    s = re.sub(r"[^\d,.]", "", s)
    if not s:
        return pd.NA
    # Ưu tiên giá VN dạng "320.000" = 320000, KHÔNG hiểu nhầm thành 320.0.
    if re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", s):
        return int(re.sub(r"[.,]", "", s))
    # Khi pandas chuyển cột số sang chuỗi, "260000.0" vẫn là 260000.
    if re.fullmatch(r"\d+\.0+", s):
        return int(float(s))
    if s.isdigit():
        return int(s)
    return pd.NA


def parse_discount_pct(discount: object):
    if pd.isna(discount) or not str(discount).strip():
        return pd.NA
    s = str(discount).strip().replace(",", ".")
    match = re.search(r"-?(\d+(?:\.\d+)?)\s*%", s)
    if match:
        value = float(match.group(1))
        return value if 0 <= value <= 100 else pd.NA
    # JSON crawler có thể ghi giảm giá dạng số nguyên (không có %).
    if re.fullmatch(r"\d+(?:\.\d+)?", s):
        value = float(s)
        return value if 0 <= value <= 100 else pd.NA
    return pd.NA


def load_inputs(paths: list[str]) -> pd.DataFrame:
    chunks = []
    for path_str in paths:
        path = resolve_path(path_str)
        if not path.exists():
            print(f"[!] Bỏ qua file chưa tồn tại: {path}")
            continue
        df = pd.read_csv(path, encoding="utf-8-sig", dtype={"review_id": "string", "product_id": "string"}, low_memory=False)
        df.columns = [str(c).strip() for c in df.columns]
        for src, dest in COLUMN_ALIASES.items():
            if src in df.columns and dest not in df.columns:
                df = df.rename(columns={src: dest})
        if "comment" not in df.columns:
            print(f"[!] File {path.name} không có cột comment/review_text; bỏ qua.")
            continue
        if "platform" not in df.columns:
            name = path.name.lower()
            guess = "Shopee" if "shopee" in name else "Lazada" if "lazada" in name else "Tiki" if "tiki" in name else "Unknown"
            df["platform"] = guess
        df["source_file"] = path.name
        chunks.append(df)
        print(f"[+] {path.name}: {len(df):,} dòng.")
    if not chunks:
        raise FileNotFoundError("Không tìm thấy file đầu vào hợp lệ. Kiểm tra INPUT_FILES hoặc --input.")
    return pd.concat(chunks, ignore_index=True, sort=False)


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Ưu tiên review_id ổn định; nếu thiếu dùng product+date+rating+raw text.

    Không loại trùng toàn cục theo comment: nhiều khách có thể viết 'dùng tốt'.
    """
    df = df.copy()
    for key in ("review_id", "product_id", "product_url", "product_name", "date", "rating", "platform"):
        if key not in df.columns:
            df[key] = ""
    def txt(col):
        return df[col].fillna("").astype(str).str.strip()
    platform = txt("platform").str.lower()
    product = txt("product_url").map(normalize_url)
    # Tiki có product_id và review_id; dùng làm ID ưu tiên.
    product = product.mask(product.eq(""), txt("product_id"))
    product = product.mask(product.eq(""), txt("product_name").str.lower())
    rid = txt("review_id").str.replace(r"\.0$", "", regex=True)
    use_rid = rid.ne("") & ~rid.str.lower().isin(["nan", "none", "null"])
    raw = txt("comment").str.lower().str.replace(r"\s+", " ", regex=True)
    dedupe_key = (
        platform + "\x1f" + product + "\x1f" +
        use_rid.map({True: "id:", False: "content:"}) +
        rid.where(use_rid, txt("date") + "\x1f" + txt("rating") + "\x1f" + raw)
    )
    before = len(df)
    df = df.loc[~dedupe_key.duplicated(keep="first")].copy()
    print(f"[*] Bỏ {before-len(df):,} dòng trùng trong cùng sản phẩm/platform (hoặc review_id).")
    return df


def process_csv_file(input_file, output_file, text_column="comment", *,
                     use_vncorenlp: bool = USE_VNCORENLP,
                     filter_vietnamese: bool = FILTER_VIETNAMESE) -> pd.DataFrame:
    """Tương thích lối gọi cũ: input_file có thể là str hoặc list[str]."""
    files = [str(input_file)] if isinstance(input_file, (str, Path)) else list(input_file)
    df = load_inputs(files)
    if text_column != "comment" and text_column in df.columns:
        df["comment"] = df[text_column]
    print(f"[*] Tổng số dòng đầu vào: {len(df):,}.")
    df = df.copy()
    # Giữ raw comment, metadata; chỉ bỏ đánh giá thiếu text.
    df["comment"] = df["comment"].fillna("").astype(str).str.strip()
    df = df.loc[df["comment"].ne("")].copy()
    df = remove_duplicates(df)
    if filter_vietnamese:
        before = len(df)
        df = df.loc[df["comment"].map(is_vietnamese)].copy()
        print(f"[*] Lọc theo heuristic tiếng Việt: {before:,} -> {len(df):,} dòng.")
    else:
        print("[*] Không loại review tiếng Việt không dấu (FILTER_VIETNAMESE=False).")

    print("[*] Chuẩn hóa comment, giữ phủ định và số liệu...")
    tqdm.pandas(desc="Chuẩn hóa review")
    df["normalized_comment"] = df["comment"].progress_map(normalize_comment)
    before = len(df)
    df = df.loc[df["normalized_comment"].str.len().ge(MIN_CLEANED_LENGTH)].copy()
    print(f"[*] Bỏ {before-len(df):,} comment rỗng/quá ngắn sau chuẩn hoá.")

    segmenter = WordSegmenter(use_vncorenlp)
    segmenter.setup()
    print("[*] Word segmentation + chuẩn hóa cụm từ sữa rửa mặt...")
    df["cleaned_comment"] = df["normalized_comment"].progress_map(segmenter.tokenize)
    df["topic_comment"] = df["cleaned_comment"].map(to_topic_text)

    # Chỉ điền brand thiếu; KHÔNG ghi đè brand crawler đã thu được.
    if "brand" not in df.columns:
        df["brand"] = ""
    df["brand"] = df.apply(coalesce_brand, axis=1)
    print(f"[*] Brand chưa xác định (other): {(df['brand'] == 'other').sum():,} dòng.")
    df["review_text_length"] = df["comment"].str.len()
    if "price" in df.columns:
        df["price_vnd"] = df["price"].map(parse_price_vnd).astype("Int64")
    if "original_price" in df.columns:
        df["original_price_vnd"] = df["original_price"].map(parse_price_vnd).astype("Int64")
    if "discount" in df.columns:
        df["discount_pct"] = pd.to_numeric(df["discount"].map(parse_discount_pct), errors="coerce")
        if "original_price_vnd" in df and "price_vnd" in df:
            possible = (100 * (1 - df["price_vnd"].astype(float) /
                               df["original_price_vnd"].replace(0, pd.NA).astype(float))).round(2)
            df["discount_pct"] = df["discount_pct"].fillna(possible.where(possible.between(0, 100)))

    # Cột quan trọng đầu file; giữ toàn bộ metadata khác của cả ba sàn ở phía sau.
    leading = ["platform", "product_name", "brand", "rating", "date", "product_url",
               "price", "price_vnd", "original_price", "original_price_vnd",
               "discount", "discount_pct", "comment", "review_text_length",
               "normalized_comment", "cleaned_comment", "topic_comment",
               "attributes", "review_id", "product_id", "source_file"]
    columns = [c for c in leading if c in df.columns]
    columns.extend(c for c in df.columns if c not in columns)
    df = df.loc[:, columns]
    output = resolve_path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"\n[+] Hoàn thành: {len(df):,} review -> {output}")
    print("[+] Các cột NLP: normalized_comment (ABSA/readable), "
          "cleaned_comment (segmented), topic_comment (BERTopic).")
    return df


def append_cleaned_column(input_file, output_file=None, *, text_column="comment",
                          output_column="cleaned_comment",
                          use_vncorenlp: bool = USE_VNCORENLP) -> dict:
    """Chỉ thêm text đã xử lý; giữ nguyên từng giá trị và thứ tự dòng nguồn.

    Không lọc dòng ngắn, dòng trùng hay sửa metadata. Khi cập nhật trực tiếp,
    sao lưu bản gốc và thay thế file sau khi đã kiểm tra CSV kết quả.
    """
    source = resolve_path(input_file).resolve()
    output = resolve_path(output_file).resolve() if output_file else source
    original_bytes = source.read_bytes()
    with source.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = list(reader.fieldnames or [])
        if not columns or len(set(columns)) != len(columns):
            raise ValueError("CSV phải có tên cột hợp lệ, không trùng nhau.")
        if text_column not in columns:
            raise ValueError(f"Không có cột text nguồn: {text_column}")
        if not output_column.strip() or output_column in columns:
            raise ValueError(f"Tên cột mới rỗng hoặc đã tồn tại: {output_column!r}")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("CSV có dòng không khớp số cột; dừng để tránh mất dữ liệu.")

    segmenter = WordSegmenter(use_vncorenlp)
    segmenter.setup()
    if use_vncorenlp and segmenter.segmenter is None:
        raise RuntimeError("VNCoreNLP chưa sẵn sàng. Không cập nhật CSV. "
                           "Dùng --no-vncorenlp nếu chủ động bỏ phân đoạn từ.")
    processed = []
    for row in tqdm(rows, desc="Tiền xử lý comment"):
        text = clean_and_tokenize(row[text_column], segmenter)
        processed.append({**row, output_column: text})
    if use_vncorenlp and segmenter._warned:
        raise RuntimeError("Phân đoạn từ gặp lỗi; không cập nhật CSV với kết quả không đồng nhất.")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging_path = None
    backup = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8-sig", newline="",
                                         dir=output.parent, prefix=output.stem + ".",
                                         suffix=".tmp", delete=False) as stream:
            staging_path = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=[*columns, output_column])
            writer.writeheader()
            writer.writerows(processed)
        with staging_path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != [*columns, output_column]:
                raise RuntimeError("Tên/thứ tự cột kết quả không khớp.")
            verified = list(reader)
        if len(verified) != len(rows):
            raise RuntimeError("Số dòng kết quả không khớp nguồn.")
        for original, saved, expected in zip(rows, verified, processed):
            if any(saved[column] != original[column] for column in columns):
                raise RuntimeError("Có thay đổi ngoài cột text mới; không cập nhật CSV.")
            if saved[output_column] != expected[output_column]:
                raise RuntimeError("Text sau khi ghi CSV không khớp kết quả tiền xử lý.")
        if source.read_bytes() != original_bytes:
            raise RuntimeError("File nguồn vừa bị sửa trong khi xử lý; không ghi đè.")
        if output == source:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            backup = source.with_name(f"{source.stem}.before_preprocessing_{timestamp}{source.suffix}")
            shutil.copy2(source, backup)
            if backup.read_bytes() != original_bytes:
                raise RuntimeError("Bản sao lưu không khớp nguồn; không cập nhật CSV.")
        os.replace(staging_path, output)
        staging_path = None
    finally:
        if staging_path is not None:
            staging_path.unlink(missing_ok=True)
    result = {
        "rows": len(rows), "original_columns": len(columns), "added_column": output_column,
        "empty_processed_rows": sum(not row[output_column] for row in processed),
        "vncorenlp_enabled": segmenter.segmenter is not None,
        "output": str(output), "backup": str(backup) if backup else None,
    }
    print(f"[+] Đã thêm {output_column} cho {len(rows):,} dòng; giữ nguyên các cột nguồn.")
    if backup:
        print(f"[+] Bản sao lưu: {backup}")
    print(f"[+] Kết quả: {output}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Preprocess review sữa rửa mặt trên các sàn TMĐT")
    parser.add_argument("--input", nargs="+", default=DEFAULT_INPUT_FILES,
                        help="Một hoặc nhiều CSV nguồn")
    parser.add_argument("--output", default=None,
                        help="Đường dẫn CSV kết quả; chế độ append mặc định cập nhật file đầu vào")
    parser.add_argument("--append-text-only", action="store_true",
                        help="Chỉ thêm một cột text; giữ nguyên mọi dòng và metadata")
    parser.add_argument("--text-column", default="comment", help="Cột text nguồn")
    parser.add_argument("--output-column", default="cleaned_comment", help="Tên cột text mới ở chế độ append")
    parser.add_argument("--no-vncorenlp", action="store_true",
                        help="Chạy không cần Java/VnCoreNLP (giữ cụm từ underscore)")
    parser.add_argument("--filter-vietnamese", action="store_true",
                        help="Bật lọc ngôn ngữ heuristic (không khuyến nghị cho review không dấu)")
    args = parser.parse_args()
    if args.append_text_only:
        if len(args.input) != 1:
            parser.error("--append-text-only yêu cầu đúng một file --input")
        if args.filter_vietnamese:
            parser.error("Chế độ append không lọc dòng; không dùng --filter-vietnamese")
        append_cleaned_column(args.input[0], args.output, text_column=args.text_column,
                              output_column=args.output_column,
                              use_vncorenlp=not args.no_vncorenlp)
    else:
        process_csv_file(args.input, args.output or DEFAULT_OUTPUT_FILE,
                         text_column=args.text_column, use_vncorenlp=not args.no_vncorenlp,
                         filter_vietnamese=args.filter_vietnamese)


if __name__ == "__main__":
    main()
