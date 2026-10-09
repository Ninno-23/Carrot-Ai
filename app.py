"""Carrot AI: Render-ready Flask backend for chat, study tools, search, and images."""
from __future__ import annotations

import io
import json
import logging
import os
import re
import threading
import time
from collections import defaultdict, deque
from typing import Any, Iterable
from urllib.parse import parse_qs, urljoin, urlparse
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request, send_file
from huggingface_hub import InferenceClient
from pypdf import PdfReader

APP_VERSION = "3.0.0"
CHAT_URL_DEFAULT = "https://router.huggingface.co/v1/chat/completions"
MAX_TEXT_CHARS = 90_000
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
ALLOWED_EXTENSIONS = {"pdf", "docx", "txt", "md", "py", "js", "ts", "json", "csv", "html", "css", "sql", "yaml", "yml"}

load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("carrot-ai")
app = Flask(__name__, static_folder="static", static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES + 1024 * 1024

# Lightweight per-process rate limiter. For multiple workers, use a shared store such as Redis.
_rate_lock = threading.Lock()
_rate_events: dict[str, deque[float]] = defaultdict(deque)
RATE_LIMITS = {"chat": (18, 60), "study": (12, 60), "search": (12, 60), "image": (4, 60), "document": (12, 60)}

SYSTEM_PROMPT = """You are Carrot AI, a capable, careful, friendly assistant. Answer the user's actual request directly, using the requested format and an appropriate level of detail. If a reasonable assumption lets you proceed, state it briefly instead of stalling with unnecessary questions. Be accurate and transparent about uncertainty; never invent sources, facts, or actions. For study help, prioritize supplied material, preserve its terminology, and flag missing information rather than silently inventing details. For code, provide complete, practical, secure solutions and explain important assumptions. Use clear headings or steps when they improve readability, not by default."""


def env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def client_ip() -> str:
    # Do not trust arbitrary X-Forwarded-For values; Render terminates the proxy layer.
    return request.remote_addr or "unknown"


def limited(bucket: str):
    limit, period = RATE_LIMITS[bucket]
    now = time.monotonic()
    key = f"{bucket}:{client_ip()}"
    with _rate_lock:
        events = _rate_events[key]
        while events and now - events[0] > period:
            events.popleft()
        if len(events) >= limit:
            return jsonify(error="Too many requests. Please wait a moment and try again."), 429
        events.append(now)
    return None


def extract_docx_xml(raw: bytes) -> str:
    try:
        with ZipFile(io.BytesIO(raw)) as archive:
            document = archive.getinfo("word/document.xml")
            if document.file_size > MAX_UPLOAD_BYTES:
                raise ValueError("DOCX contents exceed the supported extraction limit.")
            root = ElementTree.fromstring(archive.read(document))
    except (BadZipFile, KeyError, ElementTree.ParseError) as exc:
        raise ValueError("The DOCX file is invalid or could not be read.") from exc

    word_namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    body = root.find(f"{word_namespace}body")
    if body is None:
        raise ValueError("The DOCX file does not contain readable document content.")
    paragraphs = []
    for paragraph in body.iter(f"{word_namespace}p"):
        paragraph_text = "".join(node.text or "" for node in paragraph.iter(f"{word_namespace}t"))
        if paragraph_text.strip():
            paragraphs.append(paragraph_text)
    return "\n".join(paragraphs)


def ai_key() -> str:
    return env("AI_API_KEY") or env("HF_TOKEN")


def ai_model() -> str:
    return env("AI_MODEL", "openai/gpt-oss-120b:fastest")


def chat_url() -> str:
    return env("AI_API_URL", CHAT_URL_DEFAULT)


def safe_messages(message: str, history: Any) -> list[dict[str, str]]:
    result = [{"role": "system", "content": SYSTEM_PROMPT}]
    if isinstance(history, list):
        for item in history[-20:]:
            if not isinstance(item, dict) or item.get("role") not in ("user", "assistant"):
                continue
            content = item.get("content")
            if isinstance(content, str) and content.strip():
                result.append({"role": item["role"], "content": content[:12_000]})
    result.append({"role": "user", "content": message[:MAX_TEXT_CHARS]})
    return result


def chat_request(messages: list[dict[str, str]], *, stream: bool = False, max_tokens: int = 1600):
    key = ai_key()
    if not key:
        raise RuntimeError("AI is not configured yet. Add AI_API_KEY in your Render environment variables.")
    url = chat_url()
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise RuntimeError("AI_API_URL must be a valid HTTPS endpoint.")
    payload = {"model": ai_model(), "messages": messages, "stream": stream, "max_tokens": max_tokens, "temperature": 0.4}
    try:
        response = requests.post(url, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json", "Accept": "text/event-stream" if stream else "application/json"}, json=payload, stream=stream, timeout=(15, 180))
    except requests.RequestException as exc:
        log.warning("AI provider connection failed: %s", type(exc).__name__)
        raise RuntimeError("Could not connect to the AI provider. Please try again shortly.") from exc
    if not response.ok:
        detail = ""
        try:
            body = response.json()
            detail = body.get("error", {}).get("message", "") if isinstance(body.get("error"), dict) else str(body.get("error", ""))
        except Exception:
            detail = response.text[:300]
        response.close()
        if response.status_code in (401, 403):
            raise RuntimeError("The AI provider rejected the API key or permissions. Check AI_API_KEY in Render.")
        if response.status_code == 429:
            raise RuntimeError("The AI provider rate limit or quota was reached. Try again later or check your provider usage.")
        log.warning("AI provider returned HTTP %s: %s", response.status_code, detail[:240])
        raise RuntimeError(f"AI provider request failed (HTTP {response.status_code}). Check model availability and provider access.")
    return response


def extract_completion(data: dict[str, Any]) -> str:
    try:
        content = data["choices"][0]["message"]["content"]
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            return "".join(str(part.get("text", "")) for part in content if isinstance(part, dict)).strip()
    except (KeyError, IndexError, TypeError):
        pass
    raise RuntimeError("The AI provider returned an unexpected response format.")


def complete(messages: list[dict[str, str]], *, max_tokens: int = 1600) -> str:
    response = chat_request(messages, max_tokens=max_tokens)
    try:
        return extract_completion(response.json())
    except (ValueError, requests.JSONDecodeError) as exc:
        raise RuntimeError("The AI provider returned invalid JSON.") from exc
    finally:
        response.close()


def extract_upload(file_storage) -> str:
    if not file_storage or not file_storage.filename:
        raise ValueError("Choose a document first.")
    filename = file_storage.filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    if "." not in filename or filename.rsplit(".", 1)[-1].lower() not in ALLOWED_EXTENSIONS:
        raise ValueError("Unsupported file type. Upload a PDF, DOCX, text, Markdown, or supported code/data file.")
    raw = file_storage.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError("File exceeds the 15 MB limit.")
    ext = filename.rsplit(".", 1)[-1].lower()
    try:
        if ext == "pdf":
            reader = PdfReader(io.BytesIO(raw))
            if reader.is_encrypted:
                try:
                    reader.decrypt("")
                except Exception as exc:
                    raise ValueError("This PDF is password-protected and cannot be read.") from exc
            text = "\n\n".join((page.extract_text() or "") for page in reader.pages)
        elif ext == "docx":
            try:
                from docx import Document
            except ImportError:
                text = extract_docx_xml(raw)
            else:
                doc = Document(io.BytesIO(raw))
                pieces = [p.text for p in doc.paragraphs if p.text.strip()]
                for table in doc.tables:
                    for row in table.rows:
                        pieces.append(" | ".join(cell.text.strip() for cell in row.cells))
                text = "\n".join(pieces)
        else:
            text = raw.decode("utf-8-sig", errors="replace")
    except ValueError:
        raise
    except Exception as exc:
        log.info("Document parsing failed for .%s: %s", ext, type(exc).__name__)
        raise ValueError("The file could not be read. Try exporting it as a text-based PDF, DOCX, or UTF-8 text file.") from exc
    text = text.strip()
    if not text:
        raise ValueError("No readable text was found. Scanned/image-only PDFs need OCR before analysis.")
    return f"Document: {filename}\n\n{text[:MAX_TEXT_CHARS]}"


def request_text() -> str:
    value = request.form.get("text", "").strip()
    if value:
        return value[:MAX_TEXT_CHARS]
    if "file" in request.files:
        return extract_upload(request.files["file"])
    raise ValueError("Paste text or upload a document first.")


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip(), flags=re.I)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise RuntimeError("The AI could not return structured study content. Please try again.")
        try:
            value = json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError as exc:
            raise RuntimeError("The AI returned malformed study content. Please try again.") from exc
    if not isinstance(value, dict):
        raise RuntimeError("The AI returned an unexpected study format.")
    return value


def search_web(query: str) -> list[dict[str, str]]:
    try:
        response = requests.post("https://html.duckduckgo.com/html/", data={"q": query}, headers={"User-Agent": "Mozilla/5.0 (compatible; CarrotAI/3.0; +https://render.com)"}, timeout=(8, 15))
        response.raise_for_status()
    except requests.RequestException as exc:
        log.warning("Search provider failed: %s", type(exc).__name__)
        raise RuntimeError("Web search is temporarily unavailable. Please try again later.") from exc
    soup = BeautifulSoup(response.text, "html.parser")
    results = []
    for result in soup.select(".result"):
        link = result.select_one("a.result__a")
        snippet = result.select_one(".result__snippet")
        if not link or not link.get("href"):
            continue
        url = urljoin("https://duckduckgo.com", link.get("href", ""))
        parsed = urlparse(url)
        if parsed.hostname == "duckduckgo.com" and parsed.path.startswith("/l/"):
            targets = parse_qs(parsed.query).get("uddg", [])
            if not targets:
                continue
            url = targets[0]
            parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            continue
        results.append({"title": link.get_text(" ", strip=True)[:240], "url": url, "snippet": snippet.get_text(" ", strip=True)[:700] if snippet else ""})
        if len(results) >= 8:
            break
    return results


@app.after_request
def security_headers(response):
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    response.headers.setdefault("Cache-Control", "no-store")
    # Keep the policy compatible with the single-file frontend's existing CDN libraries and inline UI.
    response.headers.setdefault("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com; style-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com data:; img-src 'self' data: blob: https:; connect-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'; form-action 'self'")
    return response


@app.get("/")
def index():
    return send_file(os.path.join(app.root_path, "index.html"))


@app.get("/api/health")
def health():
    return jsonify(ok=True, version=APP_VERSION, ai_configured=bool(ai_key()), image_configured=bool(env("IMAGE_API_KEY") or ai_key()), model=ai_model() if ai_key() else None)


@app.post("/api/chat/stream")
def chat_stream():
    denied = limited("chat")
    if denied: return denied
    data = request.get_json(silent=True) or {}
    message = data.get("message")
    if not isinstance(message, str) or not message.strip():
        return jsonify(error="Enter a message first."), 400
    if len(message) > MAX_TEXT_CHARS:
        return jsonify(error="Your message is too long. Please shorten it."), 413
    try:
        upstream = chat_request(safe_messages(message.strip(), data.get("history")), stream=True, max_tokens=2200)
    except RuntimeError as exc:
        return jsonify(error=str(exc)), 503

    def generate():
        try:
            for raw_line in upstream.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                if isinstance(raw_line, bytes):
                    raw_line = raw_line.decode("utf-8", errors="replace")
                line = raw_line.strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if obj.get("error"):
                    err = obj["error"]
                    message_text = err.get("message", "AI provider error") if isinstance(err, dict) else str(err)
                    yield "data: " + json.dumps({"error": message_text[:500]}) + "\n\n"
                    return
                choices = obj.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    content = delta.get("content")
                    if isinstance(content, str) and content:
                        yield "data: " + json.dumps({"text": content}, ensure_ascii=False) + "\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:
            log.warning("Streaming interrupted: %s", type(exc).__name__)
            yield "data: " + json.dumps({"error": "The AI response was interrupted. Please try again."}) + "\n\n"
        finally:
            upstream.close()
    return Response(generate(), mimetype="text/event-stream", headers={"X-Accel-Buffering": "no", "Connection": "keep-alive"})


@app.post("/api/document")
def document_analysis():
    denied = limited("document")
    if denied: return denied
    try:
        text = extract_upload(request.files.get("file"))
        answer = complete([{ "role": "system", "content": SYSTEM_PROMPT + "\nAnalyze uploaded documents carefully. Distinguish direct content from interpretation, preserve key names/numbers, and mention if the document may be incomplete."}, {"role": "user", "content": "Analyze this document. Start with a concise overview, then key points, important terms/numbers, and useful next steps. Do not invent information not present in the source.\n\n" + text}], max_tokens=2200)
        return jsonify(answer=answer)
    except (ValueError, RuntimeError) as exc:
        return jsonify(error=str(exc)), 400 if isinstance(exc, ValueError) else 503


@app.post("/api/study/<kind>")
def study_tool(kind: str):
    denied = limited("study")
    if denied: return denied
    if kind not in {"summarize", "flashcards", "quiz"}:
        return jsonify(error="Unknown study tool."), 404
    try:
        source = request_text()
        if kind == "summarize":
            prompt = "Summarize the supplied material for studying. Use: (1) overview, (2) key ideas in source order, (3) definitions/names/dates/formulas, (4) common confusions only when supported, and (5) a short self-check. Do not add unsupported facts.\n\n" + source
            answer = complete([{ "role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}], max_tokens=2600)
            return jsonify(answer=answer)
        if kind == "flashcards":
            prompt = "Create 12 to 24 high-quality study flashcards from ONLY the supplied material (fewer if the source is short). Return ONLY valid JSON: {\"cards\":[{\"front\":\"question or term\",\"back\":\"accurate answer\"}]}. Cover the important concepts without duplicates. Keep answers concise and never invent facts.\n\n" + source
            data = parse_json_object(complete([{ "role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}], max_tokens=3000))
            cards = data.get("cards", data.get("flashcards", []))
            if not isinstance(cards, list): raise RuntimeError("The AI did not return a flashcard list.")
            cleaned = []
            for c in cards[:40]:
                if isinstance(c, dict):
                    front = str(c.get("front", c.get("q", ""))).strip()[:500]
                    back = str(c.get("back", c.get("a", ""))).strip()[:1200]
                    if front and back: cleaned.append({"front": front, "back": back})
            if not cleaned: raise RuntimeError("No valid flashcards were returned. Try a longer source passage.")
            return jsonify(answer=f"Created {len(cleaned)} flashcards from the supplied material.", structured=cleaned)
        prompt = "Create a multiple-choice quiz based ONLY on the supplied material. Return ONLY valid JSON: {\"questions\":[{\"question\":\"...\",\"options\":[\"A\",\"B\",\"C\",\"D\"],\"answer\":0,\"explanation\":\"...\"}]}. Create 8 to 12 questions (fewer for short sources). Each question must have exactly 4 options, answer must be integer 0-3, only one option is correct, explanations must be grounded in the source.\n\n" + source
        data = parse_json_object(complete([{ "role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}], max_tokens=3200))
        questions = data.get("questions", data.get("quiz", []))
        if not isinstance(questions, list): raise RuntimeError("The AI did not return a quiz list.")
        cleaned = []
        for q in questions[:20]:
            if not isinstance(q, dict): continue
            question = str(q.get("question", "")).strip()[:800]
            options = q.get("options")
            try: answer = int(q.get("answer", -1))
            except (ValueError, TypeError): answer = -1
            if question and isinstance(options, list) and len(options) == 4 and 0 <= answer < 4:
                cleaned.append({"question": question, "options": [str(o)[:500] for o in options], "answer": answer, "explanation": str(q.get("explanation", ""))[:1200]})
        if not cleaned: raise RuntimeError("No valid quiz questions were returned. Try a longer source passage.")
        return jsonify(answer=f"Created a {len(cleaned)}-question quiz from the supplied material.", structured=cleaned)
    except (ValueError, RuntimeError) as exc:
        return jsonify(error=str(exc)), 400 if isinstance(exc, ValueError) else 503


@app.post("/api/search")
def web_search():
    denied = limited("search")
    if denied: return denied
    data = request.get_json(silent=True) or {}
    query = data.get("query")
    if not isinstance(query, str) or not query.strip():
        return jsonify(error="Enter a search query first."), 400
    query = query.strip()[:300]
    try:
        results = search_web(query)
        synthesis = ""
        if data.get("synthesize") and results and ai_key():
            sources = "\n\n".join(f"{i+1}. {r['title']}\nURL: {r['url']}\nSnippet: {r['snippet']}" for i, r in enumerate(results))
            try:
                synthesis = complete([{ "role": "system", "content": SYSTEM_PROMPT + "\nWhen using search results, only claim what the snippets support and explicitly note that snippets may be incomplete. Do not fabricate citations."}, {"role": "user", "content": f"Summarize the useful findings for this query and cite source titles inline. Query: {query}\n\nSearch results:\n{sources}"}], max_tokens=1400)
            except RuntimeError:
                synthesis = "Search results are ready below. AI synthesis was unavailable, so review the source links directly."
        elif not results:
            synthesis = "No search results were returned. Try a more specific query."
        return jsonify(query=query, results=results, synthesis=synthesis)
    except RuntimeError as exc:
        return jsonify(error=str(exc)), 502


@app.post("/api/image")
def generate_image():
    denied = limited("image")
    if denied: return denied
    data = request.get_json(silent=True) or {}
    prompt = data.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return jsonify(error="Describe the image you want to create."), 400
    if len(prompt) > 3000:
        return jsonify(error="The image prompt is too long. Please shorten it."), 413
    key = env("IMAGE_API_KEY") or ai_key()
    if not key:
        return jsonify(error="Image generation is not configured. Add IMAGE_API_KEY in Render environment variables."), 503
    model = env("IMAGE_MODEL", "black-forest-labs/FLUX.1-dev")
    try:
        # An optional dedicated image endpoint can be configured, otherwise use Hugging Face's image client.
        image_url = env("IMAGE_API_URL")
        if image_url:
            parsed = urlparse(image_url)
            if parsed.scheme != "https" or not parsed.netloc:
                return jsonify(error="IMAGE_API_URL must be a valid HTTPS endpoint."), 400
            upstream = requests.post(image_url, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json", "Accept": "image/*, application/json"}, json={"inputs": prompt.strip()}, timeout=(15, 180))
            if not upstream.ok:
                log.warning("Configured image endpoint returned HTTP %s", upstream.status_code)
                return jsonify(error="The configured image provider rejected the request. Check the endpoint, token permissions, and quota."), 502
            content_type = upstream.headers.get("content-type", "")
            if content_type.startswith("image/"):
                mime = content_type.split(";", 1)[0]
                return send_file(io.BytesIO(upstream.content), mimetype=mime, as_attachment=False, download_name="carrot-ai-image.png", max_age=0)
            return jsonify(error="The configured image endpoint did not return an image. Use an image-generation endpoint, not a chat endpoint."), 502
        # The OpenAI-compatible chat endpoint is not used for images.
        client = InferenceClient(token=key, timeout=180)
        image = client.text_to_image(prompt.strip(), model=model)
        output = io.BytesIO()
        image.save(output, format="PNG")
        output.seek(0)
        return send_file(output, mimetype="image/png", as_attachment=False, download_name="carrot-ai-image.png", max_age=0)
    except Exception as exc:
        log.warning("Image provider failed: %s", type(exc).__name__)
        msg = "Image generation failed. Check IMAGE_API_KEY, model access, provider quota, and Hugging Face Inference Provider availability."
        # Avoid returning provider payloads because they may contain internal metadata.
        return jsonify(error=msg), 502


@app.errorhandler(413)
def too_large(_error):
    return jsonify(error="Upload is too large. The maximum request size is 15 MB per file."), 413


@app.errorhandler(404)
def not_found(_error):
    if request.path.startswith("/api/"):
        return jsonify(error="API endpoint not found."), 404
    return jsonify(error="Page not found."), 404


@app.errorhandler(500)
def internal_error(_error):
    log.exception("Unexpected server error")
    return jsonify(error="An unexpected server error occurred. Please try again."), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "10000")), debug=False)
