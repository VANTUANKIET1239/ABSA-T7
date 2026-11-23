from dataclasses import replace
from pathlib import Path
import sys
import unittest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.config import Settings, ASPECTS, POLARITIES
from backend.main import create_app
from backend.model_service import ModelUnavailable, decode_probabilities
from backend.preprocessing import parse_segmented_output, PreprocessingError
from backend.schemas import ReviewResult, summarize


class FakeService:
    """Deterministic fixture used only by tests, never selected by the running API."""
    def __init__(self, fail=False):
        self.loaded = False
        self.device = "cpu"
        self.version = "UNIT-TEST-FIXTURE"
        self.loads = 0
        self.calls = 0
        self.fail = fail

    def load(self):
        self.loads += 1
        if self.fail:
            raise ModelUnavailable("Thiếu checkpoint kiểm thử.")
        self.loaded = True

    def analyze(self, texts):
        self.calls += 1
        results = []
        for i, text in enumerate(texts):
            values = [[0.01, 0.01, 0.01, 0.97] for _ in ASPECTS]
            if text.strip():
                values[0] = [0.97, 0.01, 0.01, 0.01]
                values[6] = [0.01, 0.97, 0.01, 0.01]
            results.append(ReviewResult(review_id=i, text=text, aspects=decode_probabilities(values),
                                        status="analyzed" if text.strip() else "empty"))
        return results


class MappingTests(unittest.TestCase):
    def test_exact_class_and_aspect_order_and_none_omitted(self):
        self.assertEqual(POLARITIES, ("Positive", "Negative", "Neutral", "None"))
        values = [[float(j == i % 4) for j in range(4)] for i in range(11)]
        result = decode_probabilities(values)
        self.assertEqual(result[0].aspect, "CleansingEffectiveness")
        self.assertEqual(result[1].sentiment, "Negative")
        self.assertEqual(result[2].sentiment, "Neutral")
        self.assertNotIn("A4", [row.code for row in result])
        self.assertEqual(result[-1].code, "A11")
        self.assertEqual(result[-1].aspect, "General")

    def test_bad_shape_fails_without_remapping(self):
        with self.assertRaises(ValueError):
            decode_probabilities([[1, 0, 0]] * 11)

    def test_summary_counts_unique_pairs_and_review_mentions_separately(self):
        values = [[0, 0, 0, 1] for _ in ASPECTS]
        values[0] = [1, 0, 0, 0]
        positive = decode_probabilities(values)[0]
        negative = positive.model_copy(update={"sentiment": "Negative"})
        result = ReviewResult(review_id=0, text="fixture", aspects=[positive, positive, negative])
        rows = summarize([result])
        self.assertEqual((rows["A1"].positive, rows["A1"].negative, rows["A1"].mentions), (1, 1, 1))
        self.assertEqual(rows["A2"].neutral, 0)
        self.assertEqual(len(rows), 11)

    def test_segment_boundaries_and_underscore_repair(self):
        self.assertEqual(parse_segmented_output("1\trửa\t_\n2\tmặt\t_\n\n1\tkhô\t_\n2\t_\t_\n3\tda\t_\n", 2), ["rửa mặt", "khô_da"])

    def test_misaligned_segmentation_rejected(self):
        with self.assertRaises(PreprocessingError):
            parse_segmented_output("1\tword\t_\n", 2)


class APITests(unittest.TestCase):
    def setUp(self):
        self.service = FakeService()
        self.app = create_app(replace(Settings(), max_reviews=2, max_review_chars=20, max_total_chars=30), self.service)
        self.client = TestClient(self.app)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)

    def test_model_loaded_once_and_health_has_limits(self):
        self.assertTrue(self.client.get("/health").json()["model_loaded"])
        self.client.post("/analyze", json={"reviews": ["sạch"]})
        self.client.post("/analyze", json={"reviews": ["khô"]})
        self.assertEqual(self.service.loads, 1)

    def test_batch_summary_and_indices(self):
        response = self.client.post("/analyze", json={"reviews": ["sạch", "khô"]})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total_reviews"], 2)
        self.assertEqual([r["review_id"] for r in data["results"]], [0, 1])
        self.assertEqual(data["summary"]["A1"]["positive"], 2)
        self.assertEqual(data["summary"]["A7"]["negative"], 2)
        self.assertEqual(data["summary"]["A2"]["mentions"], 0)

    def test_empty_list_is_valid_and_not_sent_to_model(self):
        data = self.client.post("/analyze", json={"reviews": []}).json()
        self.assertEqual(data["results"], [])
        self.assertEqual(self.service.calls, 0)
        self.assertEqual(len(data["summary"]), 11)

    def test_empty_text_never_creates_neutral(self):
        data = self.client.post("/analyze", json={"reviews": ["  "]}).json()
        self.assertEqual(data["results"][0]["aspects"], [])
        self.assertTrue(all(row["neutral"] == 0 for row in data["summary"].values()))

    def test_request_limits_before_inference(self):
        for reviews in (["a"] * 3, ["a" * 21], ["a" * 16] * 2):
            self.assertEqual(self.client.post("/analyze", json={"reviews": reviews}).status_code, 413)
        self.assertEqual(self.service.calls, 0)

    def test_validation_does_not_echo_review_text(self):
        response = self.client.post("/analyze", json={"reviews": ["PRIVATE REVIEW", 123]})
        self.assertEqual(response.status_code, 422)
        self.assertNotIn("PRIVATE REVIEW", response.text)

    def test_unexpected_fields_rejected(self):
        self.assertEqual(self.client.post("/analyze", json={"reviews": [], "url": "https://example.com"}).status_code, 422)

    def test_only_extension_cors_and_local_host(self):
        good = self.client.options("/analyze", headers={"Origin": "chrome-extension://" + "a" * 32,
                                  "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type"})
        self.assertEqual(good.status_code, 200)
        bad = self.client.get("/health", headers={"Origin": "https://shopee.vn"})
        self.assertNotIn("access-control-allow-origin", bad.headers)
        self.assertEqual(self.client.get("/health", headers={"Host": "evil.example"}).status_code, 400)

    def test_missing_model_reports_error_without_fake_predictions(self):
        with TestClient(create_app(Settings(), FakeService(fail=True))) as client:
            self.assertFalse(client.get("/health").json()["model_loaded"])
            response = client.post("/analyze", json={"reviews": ["review"]})
            self.assertEqual(response.status_code, 503)
            self.assertNotIn("results", response.json())

    def test_public_api_requires_key_before_inference(self):
        service = FakeService()
        settings = replace(Settings(), api_key="UNIT-TEST-SECRET", allowed_hosts=("testserver", "*.trycloudflare.com"))
        with TestClient(create_app(settings, service)) as client:
            for headers in ({}, {"X-API-Key": "wrong"}):
                response = client.post("/analyze", json={"reviews": ["sạch"]}, headers=headers)
                self.assertEqual(response.status_code, 401)
                self.assertEqual(service.calls, 0)
                self.assertNotIn(settings.api_key, response.text)
            response = client.post("/analyze", json={"reviews": ["sạch"]}, headers={"X-API-Key": settings.api_key})
            self.assertEqual(response.status_code, 200)
            self.assertTrue(client.get("/health", headers={"Host": "demo.trycloudflare.com"}).json()["model_loaded"])
            self.assertNotIn(settings.api_key, client.get("/health").text)
            self.assertEqual(client.get("/health", headers={"Host": "evil.example"}).status_code, 400)
            preflight = client.options("/analyze", headers={"Origin": "chrome-extension://" + "a" * 32,
                "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type,x-api-key"})
            self.assertEqual(preflight.status_code, 200)


if __name__ == "__main__":
    unittest.main()
