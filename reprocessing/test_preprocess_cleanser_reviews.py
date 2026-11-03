import csv
import tempfile
import unittest
import unicodedata
from pathlib import Path

from preprocess_cleanser_reviews import append_cleaned_column, normalize_comment, restore_missing_diacritics


class RestoreMissingDiacriticsTests(unittest.TestCase):
    def test_full_and_partial_unaccented_phrases_with_negation(self):
        self.assertEqual(
            normalize_comment("SAN phẩm ko kich ứng, da dau. dong   goi can than!"),
            "sản phẩm không kích ứng, da dầu. đóng gói cẩn thận!",
        )
        self.assertEqual(restore_missing_diacritics("rat tot; rat da; kho da"),
                         "rất tốt; rát da; khô da")

    def test_valid_vietnamese_english_brands_and_ambiguous_words_are_preserved(self):
        examples = [
            "hàng ngày, tinh chất, xác minh, tẩy trang, do da mình khô",
            "da dùng xong sạch sau khi rửa, làm sạch sau đó",
            "sach sau moi lan su dung",
            "hàng gia công; hàng gia côngHàng gia công",
            "da; dung; mat; ban; on; chat; it; min; rat",
            "La Roche-Posay, Simple, The Face Shop, Hada Labo, CeraVe",
            "Great product, it is on sale, apply for 5 min. Chat with seller.",
            "sản phẩm không tốt, bị rát da, không hợp, không kích ứng",
        ]
        for text in examples:
            with self.subTest(text=text):
                expected = text.replace("su dung", "sử dụng")
                self.assertEqual(restore_missing_diacritics(text), expected)

    def test_context_resolves_usage_and_delivery(self):
        self.assertEqual(
            restore_missing_diacritics("Dung san pham de rua mat; giao dung san pham"),
            "dùng sản phẩm để rửa mặt; giao đúng sản phẩm",
        )
        self.assertEqual(restore_missing_diacritics("dung san pham"), "dung sản phẩm")
        restored = restore_missing_diacritics("danh gia tôt")
        self.assertEqual(restored, "đánh giá tốt")
        self.assertEqual(restore_missing_diacritics(restored), restored)

    def test_only_complete_field_values_resolve_ambiguous_single_words(self):
        self.assertEqual(
            restore_missing_diacritics("Làm sạch:sach Tạo bọt:min Hương thơm:nhe"),
            "Làm sạch:sạch Tạo bọt:mịn Hương thơm:nhẹ",
        )
        self.assertEqual(restore_missing_diacritics("Hương thơm:on sale; sach; nhe"),
                         "Hương thơm:on sale; sach; nhe")

    def test_unicode_boundaries_and_trace(self):
        changes = []
        result = restore_missing_diacritics(unicodedata.normalize("NFD", "san phẩm"), changes=changes)
        self.assertEqual(result, "sản phẩm")
        self.assertEqual(changes, [("san phẩm", "sản phẩm")])
        self.assertEqual(restore_missing_diacritics("totally depots sansan phamx"),
                         "totally depots sansan phamx")
        self.assertEqual(restore_missing_diacritics(result), result)


class AppendCleanedColumnTests(unittest.TestCase):
    def test_in_place_preserves_duplicates_short_empty_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reviews.csv"
            columns = ["brand", "comment", "review_id", "price", "metadata"]
            rows = [
                {"brand": "SIMPLE", "comment": "srm ko kich ung 😊, pH 5.5",
                 "review_id": "000123", "price": "63.500 ₫", "metadata": "NA"},
                {"brand": "SIMPLE", "comment": "srm ko kich ung 😊, pH 5.5",
                 "review_id": "000123", "price": "63.500 ₫", "metadata": "NA"},
                {"brand": "Other", "comment": "ok", "review_id": "", "price": "", "metadata": ""},
                {"brand": "Other", "comment": "", "review_id": "0", "price": "0", "metadata": "null"},
            ]
            with path.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)
            original = path.read_bytes()
            summary = append_cleaned_column(path, use_vncorenlp=False)
            self.assertEqual(Path(summary["backup"]).read_bytes(), original)
            with path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                self.assertEqual(reader.fieldnames, columns + ["cleaned_comment"])
                result = list(reader)
            self.assertEqual([{key: row[key] for key in columns} for row in result], rows)
            self.assertEqual(result[0]["cleaned_comment"], "sữa_rửa_mặt không_kích_ứng , ph 5.5")
            self.assertEqual(result[0]["cleaned_comment"], result[1]["cleaned_comment"])
            self.assertEqual(result[2]["cleaned_comment"], "ok")
            self.assertEqual(result[3]["cleaned_comment"], "")

    def test_existing_column_is_rejected_without_modifying_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reviews.csv"
            path.write_text("comment,cleaned_comment\ngood,previous\n", encoding="utf-8")
            original = path.read_bytes()
            with self.assertRaises(ValueError):
                append_cleaned_column(path, use_vncorenlp=False)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
