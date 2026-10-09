# Carrot AI — Next Level Full Stack (Render-only)

A single-origin Flask + HTML application. The frontend and API run together on **one Render Web Service**; GitHub stores the source code only. Do not deploy the HTML as a separate Static Site.

## Included
- `index.html` — upgraded responsive frontend with dark mode, safer Markdown rendering, chat export, stop-generation control, saved chats, study tools, document upload, web search, and image generation UI.
- `app.py` — Flask API for streaming chat, document extraction/analysis, summaries, flashcards, quizzes, web search with optional AI synthesis, image generation, health checks, basic per-process rate limiting, upload limits, and security headers.
- `requirements.txt`, `render.yaml` — Render deployment setup.
- `tests/test_app.py` — backend tests with mocked provider calls.
- `smoke-check.js` — dependency-free frontend static checks.

## Render setup
1. Push this folder's contents to a GitHub repository (the files must be at repository root).
2. In Render, create **New → Web Service**, connect that repository, and choose the Python runtime. You may use the included `render.yaml` blueprint instead.
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 240 --access-logfile - --error-logfile -`
5. Add Environment variables in Render (do not add real keys to GitHub):
   - `AI_API_KEY` — a Hugging Face token with Inference Providers permission.
   - `AI_API_URL` — `https://router.huggingface.co/v1/chat/completions`
   - `AI_MODEL` — `openai/gpt-oss-120b:fastest`
   - `IMAGE_API_KEY` — token permitted to use image inference (may be the same token if its permissions/quotas support it).
   - `IMAGE_MODEL` — `black-forest-labs/FLUX.1-dev`
   - `IMAGE_API_URL` — optional; leave blank to use the Hugging Face image client. If set, it must be an HTTPS image-generation endpoint returning an image response.
6. Deploy and open the Render URL. `/api/health` should show `ok: true`; `ai_configured` should be true after the chat token is set.

## Security — important
API keys are secrets. The keys posted in the chat should be treated as exposed: revoke/rotate them in Hugging Face, then add the replacement values directly in Render's Environment settings. Never paste them into source files, frontend HTML, GitHub commits, screenshots, or logs. This project deliberately does not contain the supplied key strings.

## Features and limits
- Chat uses the Hugging Face OpenAI-compatible chat-completions endpoint and Server-Sent Events for streaming.
- Image generation uses `huggingface_hub.InferenceClient.text_to_image`; the OpenAI-compatible chat endpoint is not used for images.
- PDF/DOCX/text/code files up to 15 MB are supported. Scanned image-only PDFs need OCR; this backend explicitly reports when no extractable text is found.
- Search uses DuckDuckGo's HTML results; provider availability and markup may change. Search results should be verified at their source links.
- In-memory rate limiting is per process and resets on restart. A production multi-worker/multi-instance deployment should use a shared rate-limit store.
- Free Render services may sleep and may have resource limits. Hugging Face model access, provider availability, permissions, and quotas depend on the Hugging Face account.
- Conversations are stored in the browser's local storage and are not synced across devices.

## Local checks
- Frontend static checks: `node smoke-check.js`
- Backend tests: `python -m unittest discover -s tests -v`
- Run locally after installing requirements: copy `.env.example` to `.env`, add provider settings, then run `python app.py`. `.env` is ignored by Git.
