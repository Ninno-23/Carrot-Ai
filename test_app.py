import io
import app as carrot


def test_home_page():
    response = carrot.app.test_client().get('/')
    assert response.status_code == 200
    assert b'Carrot AI' in response.data


def test_health_does_not_expose_secrets(monkeypatch):
    monkeypatch.setattr(carrot, 'AI_API_KEY', 'secret-test-value')
    response = carrot.app.test_client().get('/api/health')
    assert response.status_code == 200
    assert response.json['ai_configured'] is True
    assert b'secret-test-value' not in response.data


def test_chat_rejects_blank_message():
    response = carrot.app.test_client().post('/api/chat', json={'message': '   '})
    assert response.status_code == 400
    assert 'error' in response.json


def test_chat_returns_answer_when_provider_mocked(monkeypatch):
    monkeypatch.setattr(carrot, 'ask_ai', lambda messages, **kwargs: 'Hello from mock AI')
    response = carrot.app.test_client().post('/api/chat', json={'message': 'hello', 'history': []})
    assert response.status_code == 200
    assert response.json['answer'] == 'Hello from mock AI'


def test_chat_handles_missing_provider_configuration(monkeypatch):
    monkeypatch.setattr(carrot, 'AI_API_KEY', '')
    response = carrot.app.test_client().post('/api/chat', json={'message': 'hello'})
    assert response.status_code == 502
    assert 'not configured' in response.json['error'].lower()


def test_unknown_study_tool_is_not_found():
    response = carrot.app.test_client().post('/api/study/unknown', data={'text': 'abc'})
    assert response.status_code == 404


def test_study_tool_uses_mock_ai(monkeypatch):
    monkeypatch.setattr(carrot, 'ask_ai', lambda messages, **kwargs: 'Q: What?\nA: This.')
    response = carrot.app.test_client().post('/api/study/flashcards', data={'text': 'A source passage.'})
    assert response.status_code == 200
    assert response.json['tool'] == 'flashcards'
    assert 'Q: What?' in response.json['answer']


def test_document_rejects_unsupported_extension():
    response = carrot.app.test_client().post('/api/document', data={'file': (io.BytesIO(b'abc'), 'bad.exe')}, content_type='multipart/form-data')
    assert response.status_code == 400
    assert 'supported document formats' in response.json['error'].lower()


def test_image_is_explicitly_optional(monkeypatch):
    monkeypatch.setattr(carrot, 'IMAGE_API_URL', '')
    monkeypatch.setattr(carrot, 'IMAGE_API_KEY', '')
    response = carrot.app.test_client().post('/api/image', json={'prompt': 'A carrot in space'})
    assert response.status_code == 503
