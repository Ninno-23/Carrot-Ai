# 🥕 Carrot AI v2

Full-stack study assistant: chat, PDF/Word analysis, flashcards, quizzes, image generation, and web search.

API keys stay on the **server**. Never put provider keys in HTML or commit them to GitHub.

## Features

| Area | What works |
|------|------------|
| **Chat** | Questions, code help, conversation history (browser localStorage) |
| **Documents** | PDF, DOCX, TXT, MD upload (15 MB). Answers can use document context |
| **Study** | Summary, reviewer, flashcards, quiz from uploaded text or paste |
| **Images** | Real image generation via Hugging Face (FLUX.1-schnell by default); download button |
| **Search** | DuckDuckGo web search + optional AI summary with source links |
| **Quality** | Key protection, upload validation, health/config APIs, pytest suite |
| **Mobile** | Responsive sidebar, touch-friendly controls |

### Limits (honest)

- ChatGPT *subscriptions* do not automatically include API credits — you need a provider API key.
- Image generation uses a free Hugging Face token (`HF_TOKEN`) and FLUX.1-schnell by default. First request may take ~20s while the model loads.
- Scanned PDFs need OCR first; this server extracts *existing* text only.
- Render free tier may sleep; chat history is stored in the browser, not a durable server DB.

## Run locally

Python 3.11 or 3.12 recommended.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
# Edit .env — put your real AI_API_KEY (never commit .env)

python app.py
```

Open http://127.0.0.1:5000

### Free provider examples

| Provider | `AI_API_URL` | Notes |
|----------|--------------|--------|
| [Groq](https://console.groq.com) | `https://api.groq.com/openai/v1/chat/completions` | Fast free tier; model e.g. `llama-3.3-70b-versatile` |
| OpenRouter | `https://openrouter.ai/api/v1/chat/completions` | Free models available |
| OpenAI | `https://api.openai.com/v1/chat/completions` | Paid |

## Image generation (accurate photos / art)

1. Create a **free** token: https://huggingface.co/settings/tokens (Read access is enough).
2. Set `HF_TOKEN` in `.env` (local) or Render → Environment.
3. Default model: `black-forest-labs/FLUX.1-schnell` — fast and high quality on the free tier.
4. In the app: type a prompt → press **🎨 Image**. Use download to save the picture.

Works on Wi‑Fi or mobile data as long as the server can reach Hugging Face.

If the first request says the model is loading, wait ~20 seconds and try again.

## Your Hugging Face setup (recommended)

On **Render → Environment** set:

| Key | Value |
|-----|--------|
| `AI_API_URL` | `https://router.huggingface.co/v1/chat/completions` |
| `AI_API_KEY` | your HF token (same as on huggingface.co/settings/tokens) |
| `AI_MODEL` | `Qwen/Qwen2.5-7B-Instruct` |
| `HF_TOKEN` | same HF token (enables 🎨 image generation) |

Do **not** put the real token in GitHub files. Only in Render (or local `.env`).

## Deploy to GitHub + Render

1. Extract this ZIP so files are at the **repository root** (no nested folder).
2. Push to GitHub (`app.py`, `requirements.txt`, `render.yaml`, `templates/` at root).
3. In Render: **New → Blueprint** and connect the repo, **or** create a Python web service:
   - Build: `pip install -r requirements.txt`
   - Start: `gunicorn app:app --workers 2 --threads 4 --timeout 120 --bind 0.0.0.0:$PORT`
4. Set environment variables (same names as `.env.example`):
   - `AI_API_URL`, `AI_API_KEY`, `AI_MODEL` (required for chat/study)
   - Optional: `IMAGE_API_URL`, `IMAGE_API_KEY`, `IMAGE_MODEL`
5. Deploy. Open `/api/health` — you want `"status":"healthy"` and `"ai_configured":true`.

Do **not** paste real keys into chat, source code, or GitHub. If you are under 18, ask a parent/guardian before enabling paid APIs.

## Repository layout

```text
.
├── app.py
├── requirements.txt
├── render.yaml
├── .env.example
├── .gitignore
├── README.md
├── data/
│   └── .gitkeep
├── static/
├── tests/
│   ├── conftest.py
│   └── test_app.py
└── templates/
    └── index.html
```

## API overview

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/health` | Status + feature flags (no secrets) |
| GET | `/api/config` | Capability list for the UI |
| POST | `/api/chat` | `{ "message", "history?", "context?" }` |
| POST | `/api/document` | multipart file → extracted text |
| POST | `/api/study/<summary\|reviewer\|flashcards\|quiz>` | file or `{ "text" }` |
| POST | `/api/image` | `{ "prompt" }` → b64 or url |
| POST | `/api/search` | `{ "query", "summarize?" }` |

## Tests

```bash
pytest -q
```

## Security notes

- `.gitignore` blocks `.env`, caches, and databases.
- Health/config endpoints never echo API keys.
- Uploads are limited to 15 MB and known document types.
- Document text is treated as untrusted data in prompts (not as system instructions).
