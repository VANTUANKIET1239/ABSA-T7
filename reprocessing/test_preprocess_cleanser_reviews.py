import csv
import hashlib
import json
import tempfile
import unittest
import unicodedata
from pathlib import Path

from preprocess_cleanser_reviews import (
    WordSegmenter, append_cleaned_column, normalize_comment, remove_punctuation,
    restore_missing_diacritics,
    audit_review_languages, remove_reviewed_records,
)


class RestoreMissingDiacriticsTests(unittest.TestCase):
    def test_full_and_partial_unaccented_phrases_with_negation(self):
        self.assertEqual(
            normalize_comment("SAN phẩm ko kich ứng, da dau. dong   goi can than!"),
            "sản phẩm không kích ứng da dầu đóng gói cẩn thận",
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


class RemovePunctuationTests(unittest.TestCase):
    def test_sentence_punctuation_is_replaced_without_joining_words(self):
        self.assertEqual(
            normalize_comment("Không sạch...da khô, rát: không hợp!?; (thơm) “dịu nhẹ”…"),
            "không sạch da khô rát không hợp thơm dịu nhẹ",
        )

    def test_preserves_vietnamese_accents_negation_digits_and_word_joiners(self):
        source = unicodedata.normalize("NFD", "không_kích_ứng, dưỡng_ẩm: 120ml! 50% + dịu_nhẹ")
        self.assertEqual(remove_punctuation(source), "không_kích_ứng dưỡng_ẩm 120ml 50 dịu_nhẹ")

    def test_field_diacritics_are_restored_before_colons_are_removed(self):
        self.assertEqual(
            normalize_comment("Làm sạch:sach Tạo bọt:min Hương thơm:nhe"),
            "làm sạch sạch tạo bọt mịn hương thơm nhẹ",
        )

    def test_punctuation_returned_by_segmenter_is_also_removed(self):
        class FakeSegmenter:
            def word_segment(self, text):
                return ["không_kích_ứng . dưỡng_ẩm , tốt !"]

        segmenter = WordSegmenter(False)
        segmenter.segmenter = FakeSegmenter()
        self.assertEqual(segmenter.tokenize("không kích ứng dưỡng ẩm tốt"),
                         "không_kích_ứng dưỡng_ẩm tốt")


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
            self.assertEqual(result[0]["cleaned_comment"], "sữa_rửa_mặt không_kích_ứng ph 5 5")
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


class LanguageReviewTests(unittest.TestCase):
    def test_reviewed_removal_preserves_unaccented_mixed_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            columns = ["comment", "platform", "review_id", "gemini_label"]
            rows = [
                {"comment": "Great product, fast delivery", "platform": "Tiki", "review_id": "001", "gemini_label": "old"},
                {"comment": "san pham tot", "platform": "Tiki", "review_id": "002", "gemini_label": "keep"},
                {"comment": "Great product nhưng hơi khô da", "platform": "Tiki", "review_id": "003", "gemini_label": "keep"},
                {"comment": "Great product, fast delivery", "platform": "Tiki", "review_id": "004", "gemini_label": "old"},
            ]
            with source.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)
            original = source.read_bytes()
            out = root / "audit"
            audit_review_languages(source, out)
            self.assertEqual(source.read_bytes(), original)
            candidates = json.loads((out / "candidates.json").read_text(encoding="utf-8"))
            self.assertEqual({r["source_record"] for r in candidates}, {2, 3, 4, 5})
            decisions = root / "decisions.json"
            digest = hashlib.sha256(original).hexdigest()
            manifest = {"source_sha256": digest, "decisions": {
                "2": {"status": "english", "reason": "English review"},
                "3": {"status": "retained", "reason": "Vietnamese without accents"},
                "4": {"status": "mixed", "reason": "Mixed Vietnamese and English"},
                "5": {"status": "english", "reason": "English review"},
            }}
            decisions.write_text(json.dumps(manifest), encoding="utf-8")
            audit_review_languages(source, out, decisions)
            self.assertEqual(source.read_bytes(), original)
            report = remove_reviewed_records(source, [out / "english_reviews.csv"],
                                             record_column="language_source_record",
                                             removal_label="language", expected_sha256=digest)
            self.assertEqual(report["rows_removed"], 2)
            self.assertEqual(Path(report["backup"]).read_bytes(), original)
            with source.open(encoding="utf-8-sig", newline="") as stream:
                self.assertEqual(list(csv.DictReader(stream)), rows[1:3])
            with self.assertRaises(ValueError):
                audit_review_languages(source, out, decisions)
            with self.assertRaises(ValueError):
                remove_reviewed_records(source, [out / "english_reviews.csv"],
                                        record_column="language_source_record", expected_sha256=digest)

    def test_incomplete_decisions_and_changed_source_row_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            source.write_text('comment,platform\nGreat product,Tiki\n', encoding="utf-8")
            original = source.read_bytes()
            decisions = root / "decisions.json"
            decisions.write_text(json.dumps({"source_sha256": hashlib.sha256(original).hexdigest(),
                                             "decisions": {}}), encoding="utf-8")
            with self.assertRaises(ValueError):
                audit_review_languages(source, root / "audit", decisions)
            flagged = root / "flagged.csv"
            flagged.write_text('comment,platform,language_source_record\nChanged,Tiki,2\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                remove_reviewed_records(source, [flagged], record_column="language_source_record")
            self.assertEqual(source.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
