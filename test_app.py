import io
import json
import os
import unittest
from unittest.mock import Mock, patch
from zipfile import ZipFile

import app as carrot


class CarrotApiTests(unittest.TestCase):
    def setUp(self):
        carrot.app.config["TESTING"] = True
        self.client = carrot.app.test_client()
        with carrot._rate_lock:
            carrot._rate_events.clear()

    def test_health_does_not_expose_secrets(self):
        with patch.dict(os.environ, {"AI_API_KEY": "not-a-real-key", "AI_MODEL": "openai/gpt-oss-120b:fastest"}):
            response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["ok"])
        self.assertTrue(data["ai_configured"])
        self.assertNotIn("not-a-real-key", response.get_data(as_text=True))

    def test_chat_requires_message(self):
        response = self.client.post("/api/chat/stream", json={"message": "  "})
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())

    def test_chat_missing_key_returns_actionable_error(self):
        with patch.dict(os.environ, {"AI_API_KEY": "", "HF_TOKEN": ""}, clear=False):
            response = self.client.post("/api/chat/stream", json={"message": "hello", "history": []})
        self.assertEqual(response.status_code, 503)
        self.assertIn("AI_API_KEY", response.get_json()["error"])

    def test_unsupported_study_tool(self):
        response = self.client.post("/api/study/not-a-tool", data={"text": "hello"})
        self.assertEqual(response.status_code, 404)

    def test_summary_requires_input(self):
        with patch.dict(os.environ, {"AI_API_KEY": "dummy"}):
            response = self.client.post("/api/study/summarize", data={})
        self.assertEqual(response.status_code, 400)

    @patch.object(carrot, "InferenceClient")
    def test_image_generation_uses_token_parameter(self, inference_client):
        image = Mock()
        image.save.side_effect = lambda output, format: output.write(b"image")
        inference_client.return_value.text_to_image.return_value = image
        with patch.dict(os.environ, {"IMAGE_API_KEY": "image-token", "AI_API_KEY": ""}):
            response = self.client.post("/api/image", json={"prompt": "A carrot in space"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "image/png")
        inference_client.assert_called_once_with(token="image-token", timeout=180)

    def test_extract_text_file(self):
        from werkzeug.datastructures import FileStorage
        f = FileStorage(stream=io.BytesIO(b"Hello\nCarrot AI"), filename="notes.txt")
        self.assertIn("Hello", carrot.extract_upload(f))

    def test_extract_docx_without_optional_parser(self):
        from werkzeug.datastructures import FileStorage
        raw = io.BytesIO()
        document_xml = '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>DOCX notes</w:t></w:r></w:p></w:body></w:document>'
        with ZipFile(raw, "w") as archive:
            archive.writestr("word/document.xml", document_xml)
        upload = FileStorage(stream=io.BytesIO(raw.getvalue()), filename="notes.docx")
        self.assertIn("DOCX notes", carrot.extract_upload(upload))

    def test_unsafe_upload_extension_rejected(self):
        from werkzeug.datastructures import FileStorage
        f = FileStorage(stream=io.BytesIO(b"not allowed"), filename="payload.exe")
        with self.assertRaises(ValueError):
            carrot.extract_upload(f)

    def test_security_headers_present(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.headers.get("X-Frame-Options"), "DENY")
        self.assertEqual(response.headers.get("X-Content-Type-Options"), "nosniff")
        self.assertIn("default-src 'self'", response.headers.get("Content-Security-Policy", ""))

    def test_json_parser_accepts_fenced_json(self):
        parsed = carrot.parse_json_object('```json\n{"cards":[{"front":"Q","back":"A"}]}\n```')
        self.assertEqual(parsed["cards"][0]["front"], "Q")

    @patch.object(carrot.requests, "post")
    def test_search_unwraps_duckduckgo_result_links(self, post):
        response = Mock()
        response.text = '<div class="result"><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Farticle">Example</a><a class="result__snippet">Useful result</a></div>'
        response.raise_for_status.return_value = None
        post.return_value = response
        results = carrot.search_web("example query")
        self.assertEqual(results[0]["url"], "https://example.com/article")


if __name__ == "__main__":
    unittest.main()
