const fs = require('fs');
const html = fs.readFileSync(__dirname + '/index.html', 'utf8');
const app = fs.readFileSync(__dirname + '/app.py', 'utf8');
const cancellableRequests = (html.match(/signal: activeController\.signal/g) || []).length;
const checks = [
  ['doctype', /<!DOCTYPE html>/i.test(html)],
  ['main chat stream endpoint preserved', html.includes('/api/chat/stream')],
  ['study endpoints preserved', html.includes('/api/study/')],
  ['document endpoint preserved', html.includes('/api/document')],
  ['search endpoint preserved', html.includes('/api/search')],
  ['image endpoint preserved', html.includes('/api/image')],
  ['health endpoint preserved', html.includes('/api/health')],
  ['stop generation control', html.includes('id="stop"') && html.includes('activeController.abort()')],
  ['conversation export', html.includes('function exportCurrentChat()')],
  ['safe external URL validation', html.includes('function safeUrl(value)')],
  ['study widgets persisted', html.includes('type: "flashcards"') && html.includes('type: "quiz"')],
  ['reduced motion support', html.includes('prefers-reduced-motion:reduce')],
  ['provider setup guidance', html.includes('id="setupDialog"') && html.includes('AI_API_KEY')],
  ['image and search do not depend on native prompts', !html.includes('window.prompt(') && !html.includes('prompt("What do you want to search for?")')],
  ['chat, study, search, document, and image requests cancellable', cancellableRequests >= 5],
  ['local .env settings loaded', app.includes('load_dotenv()')],
  ['Hugging Face image token passed correctly', /InferenceClient\(token=key/.test(app)],
  ['DuckDuckGo redirect links unwrapped', app.includes('parse_qs(parsed.query)')],
];
let fail = false;
for (const [name, ok] of checks) { console.log(`${ok ? 'PASS' : 'FAIL'} ${name}`); if (!ok) fail = true; }
if (fail) process.exit(1);
