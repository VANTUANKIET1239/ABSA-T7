import unittest
import unicodedata

from preprocess_cleanser_reviews import normalize_comment
from remove_seller_replies import has_customer_content
from seller_reply_rules import strip_seller_reply


class SellerReplyTests(unittest.TestCase):
    def test_keep_negative_customer_review_before_cerave_response(self):
        customer = "Mua lần đầu, không sạch, không hợp da. 😔"
        reply = ("Chào bạn, cám ơn bạn đã quan tâm và ủng hộ sản phẩm của Cerave Lazada Mall. "
                 "Shop hi vọng vẫn sẽ tiếp tục nhận được sự tin yêu từ bạn. Chúc bạn luôn xinh đẹp ❤")
        retained, evidence = strip_seller_reply(customer + " " + reply)
        self.assertEqual(retained, customer)
        self.assertEqual((retained + evidence["removed_text"]), customer + " " + reply)
        normalized = normalize_comment(customer + " " + reply)
        self.assertIn("không sạch", normalized)
        self.assertIn("không hợp da", normalized)
        self.assertNotIn("cerave", normalized)
        self.assertNotIn("tin yêu", normalized)

    def test_offset_mapping_preserves_raw_unicode_and_line_breaks(self):
        customer = unicodedata.normalize("NFD", "Đóng gói tệ, da bị rát! 😢")
        text = customer + "\r\nRohto   Mentholatum Việt Nam xin chào quý khách! Rất xin lỗi quý khách."
        kept, evidence = strip_seller_reply(text)
        self.assertEqual(kept, customer)
        self.assertEqual(kept + evidence["removed_text"], text)

    def test_customer_gratitude_and_experiences_are_preserved(self):
        reviews = [
            "Cảm ơn shop, cảm ơn bạn shipper đã giao hàng nhanh. Chúc shop đắt hàng!",
            "Tôi rất vui vì được tặng quà, chúng mình đã mua nhiều lần.",
            "Không hợp da, chúng tôi bị mụn sau khi dùng. Mong shop phản hồi cho mình!",
            "Shop còn gửi thiệp cảm ơn viết tay, dễ thương quá.",
            "Chào mọi người, mình dùng thấy rất tốt, sẽ mua lại.",
        ]
        for text in reviews:
            with self.subTest(text=text):
                self.assertEqual(strip_seller_reply(text), (text, None))

    def test_earliest_boundary_includes_sender_and_full_reply(self):
        cases = [
            ("tệ", "Cetaphil Skincare rất tiếc khi bạn chưa có trải nghiệm hài lòng. Chúc bạn một ngày tốt lành!"),
            ("sạch", "Cỏ rất trân trọng đánh giá mà bạn dành cho sản phẩm ạ. Chúc bạn vui vẻ!"),
            ("ok", "Cỏ mềm xin được gửi tới bạn thật nhiều yêu thương. Chúng mình luôn trân trọng bạn."),
            ("khá ổn", "Cảm ơn phản hồi của bạn về sản phẩm của Cỏ Mềm. Cỏ xin ghi nhận phản hồi."),
            ("tốt", "Thật vui khi chúng mình nhận được phản hồi của bạn sau khi sử dụng sản phẩm của Cỏ ạ."),
            ("hợp", "ShopmyphamNga Mockba rất vui được phục vụ quý khách các lần sau."),
            ("được", "Cám ơn bạn đã chọn mua hàng tại SunLady shop!."),
        ]
        for customer, reply in cases:
            with self.subTest(reply=reply):
                kept, evidence = strip_seller_reply(customer + " " + reply)
                self.assertEqual(kept, customer)
                self.assertIsNotNone(evidence)
                self.assertEqual(strip_seller_reply(kept), (kept, None))

    def test_seller_only_records_do_not_become_fake_customer_reviews(self):
        for text in ["", "❤️", "🤹😍😍", "Hiệu quả：", "Chào,", "Bao bì: Mẹo:"]:
            with self.subTest(text=text):
                self.assertFalse(has_customer_content(text))
        for text in ["ok", "đc", "Bình thường", "Chất lượng: không tốt", "5 sao"]:
            with self.subTest(text=text):
                self.assertTrue(has_customer_content(text))


if __name__ == "__main__":
    unittest.main()
