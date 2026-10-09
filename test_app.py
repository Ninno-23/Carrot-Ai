import io
import app as carrot


def test_home_page():
    response = carrot.app.test_client().get("/")
    assert response.status_code == 200
    assert b"Carrot AI" in response.data


def test_health_does_not_expose_secrets(monkeypatch):
    monkeypatch.setattr(carrot, "AI_API_KEY", "secret-test-value")
    response = carrot.app.test_client().get("/api/health")
    assert response.status_code == 200
    assert response.json["ai_configured"] is True
    assert b"secret-test-value" not in response.data
    assert "features" in response.json


def test_config_lists_study_tools():
    response = carrot.app.test_client().get("/api/config")
    assert response.status_code == 200
    tools = response.json["study_tools"]
    assert "flashcards" in tools
    assert "quiz" in tools


def test_chat_rejects_blank_message():
    response = carrot.app.test_client().post("/api/chat", json={"message": "   "})
    assert response.status_code == 400
    assert "error" in response.json


def test_chat_returns_answer_when_provider_mocked(monkeypatch):
    monkeypatch.setattr(carrot, "ask_ai", lambda messages, **kwargs: "Hello from mock AI")
    response = carrot.app.test_client().post(
        "/api/chat", json={"message": "hello", "history": []}
    )
    assert response.status_code == 200
    assert response.json["answer"] == "Hello from mock AI"


def test_chat_handles_missing_provider_configuration(monkeypatch):
    monkeypatch.setattr(carrot, "AI_API_KEY", "")
    response = carrot.app.test_client().post("/api/chat", json={"message": "hello"})
    assert response.status_code in (502, 503)
    assert "not configured" in response.json["error"].lower()


def test_unknown_study_tool_is_not_found():
    response = carrot.app.test_client().post("/api/study/unknown", data={"text": "abc"})
    assert response.status_code == 404


def test_study_tool_uses_mock_ai(monkeypatch):
    monkeypatch.setattr(carrot, "ask_ai", lambda messages, **kwargs: "Q: What?\nA: This.")
    response = carrot.app.test_client().post(
        "/api/study/flashcards", data={"text": "A source passage."}
    )
    assert response.status_code == 200
    assert response.json["tool"] == "flashcards"
    assert "Q: What?" in response.json["answer"]


def test_document_rejects_unsupported_extension():
    response = carrot.app.test_client().post(
        "/api/document",
        data={"file": (io.BytesIO(b"abc"), "bad.exe")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "supported" in response.json["error"].lower()


def test_document_accepts_txt():
    response = carrot.app.test_client().post(
        "/api/document",
        data={"file": (io.BytesIO(b"Hello study text"), "notes.txt")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    assert response.json["chars"] > 0
    assert "Hello" in response.json["text"]


def test_image_requires_token_when_missing(monkeypatch):
    monkeypatch.setattr(carrot, "IMAGE_API_URL", "")
    monkeypatch.setattr(carrot, "IMAGE_API_KEY", "")
    monkeypatch.setattr(carrot, "HF_TOKEN", "")
    response = carrot.app.test_client().post(
        "/api/image", json={"prompt": "A carrot in space"}
    )
    assert response.status_code == 503
    assert "huggingface" in response.json["error"].lower() or "token" in response.json["error"].lower()


def test_image_configured_with_hf_token(monkeypatch):
    monkeypatch.setattr(carrot, "IMAGE_API_URL", "")
    monkeypatch.setattr(carrot, "IMAGE_API_KEY", "hf_test_token")
    monkeypatch.setattr(carrot, "IMAGE_MODEL", "black-forest-labs/FLUX.1-schnell")
    assert carrot.image_configured() is True


def test_image_rejects_blank_prompt(monkeypatch):
    monkeypatch.setattr(carrot, "IMAGE_API_KEY", "hf_test_token")
    response = carrot.app.test_client().post("/api/image", json={"prompt": "  "})
    assert response.status_code == 400


def test_search_rejects_empty_query():
    response = carrot.app.test_client().post("/api/search", json={"query": "  "})
    assert response.status_code == 400


def test_search_with_mock(monkeypatch):
    monkeypatch.setattr(
        carrot,
        "web_search",
        lambda query, max_results=5: [
            {"title": "T", "url": "https://example.com", "snippet": "S"}
        ],
    )
    response = carrot.app.test_client().post(
        "/api/search", json={"query": "carrots", "summarize": False}
    )
    assert response.status_code == 200
    assert len(response.json["results"]) == 1
