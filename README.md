# 🥕 Carrot AI — Full-stack starter

Carrot AI is a Flask web application with a responsive browser interface, server-side AI requests, document extraction, study tools, and optional image generation. The backend keeps API keys on the server; **never put provider keys in HTML or commit them to GitHub**.

## Features

- AI chat with browser-side conversation history
- PDF, DOCX and TXT upload and document analysis (15 MB limit)
- Study tools: summary, reviewer, flashcards, and quiz
- Optional image generation through an OpenAI-compatible image endpoint
- Health/capability endpoints at `/api/health` and `/api/config`
- Input validation, upload limits, helpful provider errors, and tests
- Render deployment configuration

### Important limits

- A provider API key and available quota are required for AI chat and study tools. ChatGPT subscriptions do not automatically include API credits.
- Image generation is **optional**, not magically enabled by the chat model. It only works after configuring a compatible image-generation endpoint and key.
- Text extraction does not perform OCR on scanned PDFs. Convert the scan to searchable text first.
- Render's free web service may sleep, and its filesystem is not a durable database. This starter saves conversations in the user's browser; it does not promise cross-device/server-side chat syncing.

## Run locally

Python 3.11 or 3.12 is recommended.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env   # Windows; on macOS/Linux use: cp .env.example .env
python app.py
```

Open `http://127.0.0.1:5000`. Edit `.env` with your provider's actual endpoint, model and key. Never share `.env` or screenshots that reveal keys.

## Deploy to GitHub + Render

1. Extract this ZIP. Upload all files to the root of your GitHub repository (no nested folder). `app.py`, `requirements.txt`, `render.yaml`, and `templates/` must be at the repository root level as shown below.
2. In Render, choose **New → Blueprint** and connect the repository, or create a Python web service with build command `pip install -r requirements.txt` and start command `gunicorn app:app --workers 2 --threads 4 --timeout 120 --bind 0.0.0.0:$PORT`.
3. In Render → Environment, set these server-side variables using values from your chosen provider:
   - `AI_API_URL`: the provider's OpenAI-compatible **chat completions** endpoint
   - `AI_API_KEY`: your secret provider key/token
   - `AI_MODEL`: a model that your provider account can actually access
4. Save and redeploy. Visit `/api/health`; it should report `"status":"healthy"` and `"ai_configured":true` when all three values are present. This checks configuration, not provider quota or live model access.
5. Optional image generation: configure `IMAGE_API_URL`, `IMAGE_API_KEY`, and, if required by that provider, `IMAGE_MODEL`. The endpoint must accept an OpenAI-compatible images request and return `data[0].b64_json` or an HTTPS `data[0].url`. If your provider uses another format, this adapter needs to be adjusted for it.

Do not use the sample placeholder `AI_API_KEY` value. Do not paste tokens into chat, source code, or GitHub files. If you are under 18, ask a parent/guardian before enabling any paid API or billing.

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
├── tests/
│   ├── conftest.py
│   └── test_app.py
└── templates/
    └── index.html
```

## Tests

```bash
pytest -q
```
