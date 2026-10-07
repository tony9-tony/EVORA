"""
EVORA Chat Server.

Serves a dark futuristic chat UI and bridges it to EVORA's existing
model, identity, memory, and reasoning infrastructure.
Uses a dedicated persistent event loop for the chat session to avoid
"Event loop is closed" errors across multiple requests.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import socket
import socketserver
import threading
import time
import webbrowser
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional

from evora import voice as voice_mod
from evora.chat import ChatSession


CHAT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EVORA</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg-primary: #0a0a0f;
    --bg-surface: rgba(255,255,255,0.03);
    --bg-elevated: rgba(255,255,255,0.06);
    --border-subtle: rgba(255,255,255,0.08);
    --border-strong: rgba(255,255,255,0.12);
    --text-primary: #e2e8f0;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --accent-indigo: #6366f1;
    --accent-indigo-dim: rgba(99,102,241,0.15);
    --accent-emerald: #10b981;
    --accent-emerald-dim: rgba(16,185,129,0.15);
    --accent-rose: #f43f5e;
    --accent-amber: #f59e0b;
    --font-mono: ui-monospace, 'Cascadia Code', 'Fira Code', 'JetBrains Mono', monospace;
    --font-sans: ui-sans-serif, system-ui, -apple-system, sans-serif;
  }
  html, body { height: 100%; }
  body {
    background: var(--bg-primary);
    background-image:
      radial-gradient(ellipse at 20% 0%, rgba(99,102,241,0.08) 0%, transparent 50%),
      radial-gradient(ellipse at 80% 100%, rgba(16,185,129,0.06) 0%, transparent 50%);
    color: var(--text-primary);
    font-family: var(--font-sans);
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  /* Header */
  .header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.75rem 1.5rem;
    background: var(--bg-surface);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid var(--border-subtle);
    flex-shrink: 0;
  }
  .brand {
    display: flex;
    align-items: center;
    gap: 0.75rem;
  }
  .logo {
    width: 32px; height: 32px;
    border-radius: 8px;
    background: linear-gradient(135deg, var(--accent-indigo), #8b5cf6);
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 14px; color: #fff;
    box-shadow: 0 0 20px rgba(99,102,241,0.3);
  }
  .brand-text h1 {
    font-size: 14px; font-weight: 600; letter-spacing: 0.5px; line-height: 1.2;
  }
  .brand-text .subtitle {
    font-size: 11px; color: var(--text-muted); font-family: var(--font-mono); letter-spacing: 0.3px;
  }
  .status-pills {
    display: flex; gap: 0.5rem; align-items: center;
  }
  .pill {
    font-size: 11px; padding: 0.2rem 0.6rem; border-radius: 9999px;
    background: var(--bg-elevated); border: 1px solid var(--border-subtle);
    color: var(--text-secondary); font-family: var(--font-mono); white-space: nowrap;
  }
  .pill .dot {
    display: inline-block; width: 6px; height: 6px; border-radius: 50%;
    margin-right: 4px; vertical-align: middle;
  }
  .dot.online { background: var(--accent-emerald); box-shadow: 0 0 6px var(--accent-emerald); }
  .dot.thinking { background: var(--accent-amber); animation: pulse 1s infinite; }
  .dot.error { background: var(--accent-rose); }
  @keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.8); }
  }

  /* Chat area */
  .chat-area {
    flex: 1; overflow-y: auto; padding: 1.5rem; display: flex; flex-direction: column; gap: 0.75rem;
    scroll-behavior: smooth;
  }
  .chat-area::-webkit-scrollbar { width: 6px; }
  .chat-area::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 3px; }
  .chat-area::-webkit-scrollbar-track { background: transparent; }

  /* Messages */
  .message-row { display: flex; gap: 0.75rem; animation: fadeIn 0.2s ease-out; }
  .message-row.user { flex-direction: row-reverse; }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }

  .avatar {
    width: 28px; height: 28px; border-radius: 50%; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    font-size: 11px; font-weight: 600; color: #fff;
  }
  .avatar.evora { background: linear-gradient(135deg, var(--accent-indigo), #8b5cf6); }
  .avatar.user { background: linear-gradient(135deg, var(--accent-emerald), #059669); }

  .bubble {
    max-width: 75%; padding: 0.65rem 1rem; border-radius: 1rem; line-height: 1.5;
    font-size: 14px; word-break: break-word; white-space: pre-wrap;
  }
  .bubble.evora {
    background: var(--accent-indigo-dim);
    border: 1px solid rgba(99,102,241,0.2);
    border-bottom-left-radius: 0.25rem;
  }
  .bubble.user {
    background: var(--accent-emerald-dim);
    border: 1px solid rgba(16,185,129,0.2);
    border-bottom-right-radius: 0.25rem;
  }
  .bubble.error {
    background: rgba(244,63,94,0.1);
    border: 1px solid rgba(244,63,94,0.2);
    color: #fda4af;
  }
  .bubble.system {
    background: var(--bg-elevated);
    border: 1px solid var(--border-subtle);
    color: var(--text-muted);
    font-size: 12px;
    text-align: center;
    max-width: 90%;
    align-self: center;
  }
  .bubble .meta {
    font-size: 10px; color: var(--text-muted); margin-top: 0.3rem;
    font-family: var(--font-mono);
  }

  /* Thinking indicator */
  .thinking-row { display: flex; gap: 0.75rem; align-items: center; padding: 0.25rem 0; }
  .thinking-dots { display: flex; gap: 4px; }
  .thinking-dots span {
    width: 6px; height: 6px; border-radius: 50%; background: var(--accent-indigo);
    animation: bounce 1.4s infinite ease-in-out both;
  }
  .thinking-dots span:nth-child(1) { animation-delay: -0.32s; }
  .thinking-dots span:nth-child(2) { animation-delay: -0.16s; }
  @keyframes bounce {
    0%, 80%, 100% { transform: scale(0); }
    40% { transform: scale(1); }
  }

  /* Tool action indicator */
  .tool-action {
    background: var(--bg-elevated);
    border: 1px solid var(--border-subtle);
    border-radius: 0.5rem;
    padding: 0.4rem 0.6rem;
    margin: 0.3rem 0;
    font-size: 12px;
    font-family: var(--font-mono);
    color: var(--text-secondary);
    border-left: 2px solid var(--accent-indigo);
  }

  /* Input area */
  .input-area {
    padding: 1rem 1.5rem; background: var(--bg-surface);
    backdrop-filter: blur(12px); border-top: 1px solid var(--border-subtle);
    display: flex; gap: 0.5rem; align-items: flex-end; flex-shrink: 0;
  }
  .input-wrap {
    flex: 1; position: relative; background: var(--bg-elevated);
    border: 1px solid var(--border-subtle); border-radius: 0.75rem;
    transition: border-color 0.15s;
  }
  .input-wrap:focus-within { border-color: rgba(99,102,241,0.4); }
  textarea {
    width: 100%; background: transparent; border: none; color: var(--text-primary);
    padding: 0.65rem 0.9rem; font-family: var(--font-sans); font-size: 14px;
    resize: none; outline: none; min-height: 42px; max-height: 160px; line-height: 1.5;
  }
  textarea::placeholder { color: var(--text-muted); }

  .icon-btn {
    width: 38px; height: 38px; border-radius: 0.6rem; border: 1px solid var(--border-subtle);
    background: var(--bg-elevated); color: var(--text-secondary); cursor: pointer;
    display: flex; align-items: center; justify-content: center; transition: all 0.15s;
    font-size: 16px; flex-shrink: 0;
  }
  .icon-btn:hover { background: rgba(255,255,255,0.1); color: var(--text-primary); }
  .icon-btn.primary {
    background: linear-gradient(135deg, var(--accent-indigo), #8b5cf6);
    border-color: transparent; color: #fff;
  }
  .icon-btn.primary:hover { opacity: 0.9; }
  .icon-btn.recording {
    background: rgba(244,63,94,0.2); border-color: rgba(244,63,94,0.4);
    color: var(--accent-rose); animation: pulse 1s infinite;
  }
  .icon-btn:disabled { opacity: 0.4; cursor: not-allowed; }

  .input-hint {
    font-size: 10px; color: var(--text-muted); padding: 0.15rem 0.6rem 0;
    font-family: var(--font-mono);
  }

  /* Welcome */
  .welcome {
    text-align: center; padding: 3rem 1rem; color: var(--text-muted);
  }
  .welcome .logo-large {
    width: 64px; height: 64px; border-radius: 16px; margin: 0 auto 1rem;
    background: linear-gradient(135deg, var(--accent-indigo), #8b5cf6);
    display: flex; align-items: center; justify-content: center;
    font-size: 28px; font-weight: 700; color: #fff;
    box-shadow: 0 0 40px rgba(99,102,241,0.3);
  }
  .welcome h2 { color: var(--text-primary); font-size: 18px; margin-bottom: 0.5rem; }
  .welcome p { font-size: 13px; max-width: 360px; margin: 0 auto; line-height: 1.5; }

  /* Loading spinner overlay */
  .loading-overlay {
    position: fixed; top: 0; left: 0; right: 0; bottom: 0;
    background: var(--bg-primary);
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    z-index: 9999;
  }
  .spinner {
    width: 48px; height: 48px;
    border: 3px solid var(--border-subtle);
    border-top-color: var(--accent-indigo);
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }
  @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
</style>
</head>
<body>
  <div class="loading-overlay" id="loadingOverlay">
    <div class="spinner"></div>
  </div>

  <header class="header">
    <div class="brand">
      <div class="logo">E</div>
      <div class="brand-text">
        <h1>EVORA</h1>
        <div class="subtitle" id="modelStatus">Initializing...</div>
      </div>
    </div>
    <div class="status-pills">
      <span class="pill"><span class="dot online" id="statusDot"></span><span id="statusText">Online</span></span>
      <span class="pill" id="identityPill">guest</span>
      <span class="pill" id="memoryPill">memory: 0</span>
    </div>
  </header>

  <div class="chat-area" id="chatArea">
    <div class="welcome" id="welcome">
      <div class="logo-large">E</div>
      <h2>EVORA Chat</h2>
      <p>Connected to your local Ollama instance. Type a message to begin.</p>
    </div>
  </div>

  <div class="input-area">
    <button class="icon-btn" id="micBtn" title="Voice input">🎤</button>
    <div class="input-wrap">
      <textarea id="userInput" rows="1" placeholder="Message EVORA..."></textarea>
      <div class="input-hint">Enter = send &nbsp;|&nbsp; Shift+Enter = newline</div>
    </div>
    <button class="icon-btn" id="clearBtn" title="Clear conversation">🗑</button>
    <button class="icon-btn primary" id="sendBtn" title="Send">➤</button>
  </div>

<script>
(function() {
  const chatArea = document.getElementById('chatArea');
  const userInput = document.getElementById('userInput');
  const sendBtn = document.getElementById('sendBtn');
  const clearBtn = document.getElementById('clearBtn');
  const micBtn = document.getElementById('micBtn');
  const modelStatus = document.getElementById('modelStatus');
  const identityPill = document.getElementById('identityPill');
  const memoryPill = document.getElementById('memoryPill');
  const statusDot = document.getElementById('statusDot');
  const statusText = document.getElementById('statusText');
  const welcome = document.getElementById('welcome');
  const loadingOverlay = document.getElementById('loadingOverlay');

  // Hide loading overlay after initial render
  setTimeout(() => {
    loadingOverlay.style.opacity = '0';
    loadingOverlay.style.transition = 'opacity 0.3s ease';
    setTimeout(() => { loadingOverlay.style.display = 'none'; }, 300);
  }, 500);

  let isProcessing = false;
  let recognition = null;
  let isRecording = false;

  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  function scrollToBottom() {
    requestAnimationFrame(() => { chatArea.scrollTop = chatArea.scrollHeight; });
  }

  function appendBubble(role, content, meta, isStreaming) {
    let row;
    if (isStreaming) {
      row = document.getElementById('evoraStreamingRow');
      if (row) {
        row.querySelector('.bubble').textContent += content;
        return;
      }
    }
    if (welcome) welcome.remove();
    row = document.createElement('div');
    row.className = 'message-row ' + role;
    if (role === 'evora' && isStreaming) row.id = 'evoraStreamingRow';
    const avatar = document.createElement('div');
    avatar.className = 'avatar ' + role;
    avatar.textContent = role === 'evora' ? 'E' : 'U';
    const bubble = document.createElement('div');
    bubble.className = 'bubble ' + role;
    bubble.textContent = content;
    if (meta) {
      const m = document.createElement('div');
      m.className = 'meta';
      m.textContent = meta;
      bubble.appendChild(m);
    }
    row.appendChild(avatar);
    row.appendChild(bubble);
    chatArea.appendChild(row);
    scrollToBottom();
  }

  function finalizeStreamingBubble(meta) {
    const row = document.getElementById('evoraStreamingRow');
    if (row) {
      row.id = '';
      if (meta) {
        const m = document.createElement('div');
        m.className = 'meta';
        m.textContent = meta;
        row.querySelector('.bubble').appendChild(m);
      }
    }
  }

  function appendSystem(text) {
    if (welcome) welcome.remove();
    const bubble = document.createElement('div');
    bubble.className = 'bubble system';
    bubble.textContent = text;
    chatArea.appendChild(bubble);
    scrollToBottom();
  }

  function appendToolAction(content) {
    if (welcome) welcome.remove();
    const el = document.createElement('div');
    el.className = 'tool-action';
    el.textContent = '🔧 ' + content;
    chatArea.appendChild(el);
    scrollToBottom();
  }

  function showThinking() {
    if (welcome) welcome.remove();
    const row = document.createElement('div');
    row.className = 'thinking-row';
    row.id = 'thinkingIndicator';
    const avatar = document.createElement('div');
    avatar.className = 'avatar evora';
    avatar.textContent = 'E';
    const dots = document.createElement('div');
    dots.className = 'thinking-dots';
    dots.innerHTML = '<span></span><span></span><span></span>';
    row.appendChild(avatar);
    row.appendChild(dots);
    chatArea.appendChild(row);
    scrollToBottom();
  }

  function hideThinking() {
    const el = document.getElementById('thinkingIndicator');
    if (el) el.remove();
  }

  function setProcessing(val) {
    isProcessing = val;
    sendBtn.disabled = val;
    userInput.disabled = val;
    if (val) {
      statusDot.className = 'dot thinking';
      statusText.textContent = 'Thinking...';
    } else {
      statusDot.className = 'dot online';
      statusText.textContent = 'Online';
    }
  }

  async function sendMessage(text) {
    if (!text.trim() || isProcessing) return;
    const msg = text.trim();
    userInput.value = '';
    autoResize();
    appendBubble('user', msg);
    setProcessing(true);
    showThinking();

    const evtSource = new EventSource('/api/chat/stream?message=' + encodeURIComponent(msg));
    let currentBubble = false;
    let hasError = false;

    evtSource.addEventListener('content', function(e) {
      hideThinking();
      const data = JSON.parse(e.data);
      if (!currentBubble) {
        appendBubble('evora', data.content || '', null, true);
        currentBubble = true;
      } else {
        appendBubble('evora', data.content || '', null, true);
      }
    });

    evtSource.addEventListener('tool', function(e) {
      const data = JSON.parse(e.data);
      appendToolAction(data.name + ': ' + (data.output || data.error || ''));
    });

    evtSource.addEventListener('error', function(e) {
      hasError = true;
      hideThinking();
      const data = JSON.parse(e.data);
      appendBubble('error', data.error || 'Unknown error', 'error');
    });

    evtSource.addEventListener('done', function(e) {
      hideThinking();
      const data = JSON.parse(e.data);
      if (!hasError) {
        finalizeStreamingBubble(data.model ? ' • ' + data.model + ' • ' + (data.response_time || '?') + 's' : null);
      }
      evtSource.close();
      setProcessing(false);
      userInput.focus();
    });

    evtSource.onerror = function() {
      if (!hasError) {
        hideThinking();
        appendBubble('error', 'Connection error: streaming failed. Try again.', 'error');
        hasError = true;
      }
      evtSource.close();
      setProcessing(false);
    };
  }

  async function sendMessageLegacy(text) {
    if (!text.trim() || isProcessing) return;
    const msg = text.trim();
    userInput.value = '';
    autoResize();
    appendBubble('user', msg);
    setProcessing(true);
    showThinking();
    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: msg }),
      });
      const data = await res.json();
      hideThinking();
      if (data.error) {
        appendBubble('error', data.error, 'error');
      } else {
        appendBubble('evora', data.response, data.provider + ' • ' + data.model);
      }
    } catch (e) {
      hideThinking();
      appendBubble('error', 'Connection error: ' + e.message, 'error');
    } finally {
      setProcessing(false);
      userInput.focus();
    }
  }

  function autoResize() {
    userInput.style.height = 'auto';
    userInput.style.height = Math.min(userInput.scrollHeight, 160) + 'px';
  }

  userInput.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage(userInput.value);
    }
  });
  userInput.addEventListener('input', autoResize);

  sendBtn.addEventListener('click', () => sendMessage(userInput.value));

  clearBtn.addEventListener('click', async () => {
    if (isProcessing) return;
    await fetch('/api/clear', { method: 'POST' });
    chatArea.innerHTML = '';
    appendSystem('Conversation cleared.');
  });

  // Voice input via Web Speech API
  function initVoice() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      micBtn.style.display = 'none';
      return;
    }
    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = 'en-US';
    recognition.onstart = () => {
      isRecording = true;
      micBtn.classList.add('recording');
      micBtn.textContent = '⏹';
    };
    recognition.onend = () => {
      isRecording = false;
      micBtn.classList.remove('recording');
      micBtn.textContent = '🎤';
    };
    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      userInput.value = transcript;
      autoResize();
      sendMessage(transcript);
    };
    recognition.onerror = (event) => {
      console.error('Speech error:', event.error);
      isRecording = false;
      micBtn.classList.remove('recording');
      micBtn.textContent = '🎤';
      appendSystem('Voice input error: ' + event.error);
    };
  }
  micBtn.addEventListener('click', () => {
    if (!recognition) { initVoice(); }
    if (isRecording) { recognition.stop(); }
    else { recognition.start(); }
  });

  // Status
  async function loadStatus() {
    try {
      const res = await fetch('/api/status');
      const data = await res.json();
      modelStatus.textContent = data.provider + ' • ' + data.model;
      identityPill.textContent = data.display_name || (data.identity + ' (' + data.authority + ')');
      memoryPill.textContent = 'memory: ' + (data.memory_count || 0);
    } catch (e) {
      modelStatus.textContent = 'Disconnected';
      statusDot.className = 'dot error';
      statusText.textContent = 'Error';
    }
  }
  loadStatus();
  setInterval(loadStatus, 30000);

  userInput.focus();
})();
</script>
</body>
</html>"""


async def _run_chat_message(message: str) -> dict:
    """Process a chat message within the shared event loop."""
    return await chat_session.process_message(message)


async def _stream_chat_message(message: str, agent: bool = True):
    """Stream one chat turn as event dicts (content, tool_start, approval_request, ...)."""
    try:
        source = chat_session.stream_agent(message) if agent else chat_session.stream_message(message)
        async for event in source:
            yield event
    except Exception as e:
        yield {"type": "error", "error": str(e)}


# ---------------------------------------------------------------- access control

_LOCAL_HOSTS = ("localhost", "127.0.0.1", "[::1]", "::1")
_failed: dict = {}
MAX_BODY = 20 * 1024 * 1024


def _evora_home() -> Path:
    return Path(os.environ.get("EVORA_HOME") or (Path.home() / ".evora"))


def get_token() -> str:
    """Owner token: env EVORA_TOKEN, else ~/.evora/web_token (created on first run)."""
    env = os.environ.get("EVORA_TOKEN")
    if env:
        return env
    path = _evora_home() / "web_token"
    try:
        if path.exists():
            value = path.read_text(encoding="utf-8").strip()
            if value:
                return value
        path.parent.mkdir(parents=True, exist_ok=True)
        value = secrets.token_urlsafe(24)
        path.write_text(value, encoding="utf-8")
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        return value
    except OSError:
        return secrets.token_urlsafe(24)


class ChatHandler(BaseHTTPRequestHandler):
    """HTTP handler for the EVORA owner console."""

    # ---- helpers
    def _client_ip(self) -> str:
        return self.client_address[0] if self.client_address else ""

    def _is_local(self) -> bool:
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].lower() if not (self.headers.get("Host") or "").startswith("[") \
            else (self.headers.get("Host") or "").split("]")[0] + "]"
        return (host in _LOCAL_HOSTS and not self.headers.get("X-Forwarded-For")
                and self._client_ip() in ("127.0.0.1", "::1", "localhost"))

    def _presented_token(self) -> str:
        from urllib.parse import urlparse, parse_qs
        tok = self.headers.get("X-Evora-Token", "")
        if not tok:
            for part in (self.headers.get("Cookie") or "").split(";"):
                k, _, v = part.strip().partition("=")
                if k == "evora_token":
                    tok = v
        if not tok:
            tok = (parse_qs(urlparse(self.path).query).get("token") or [""])[0]
        return tok

    def _locked_out(self) -> bool:
        info = _failed.get(self._client_ip())
        return bool(info and info[0] >= 10 and time.time() - info[1] < 300)

    def _token_ok(self) -> bool:
        if self._locked_out():
            return False
        tok = self._presented_token()
        ok = bool(tok) and secrets.compare_digest(tok, get_token())
        if not ok and tok:
            fails, first = _failed.get(self._client_ip(), (0, time.time()))
            _failed[self._client_ip()] = (fails + 1, first if fails else time.time())
        return ok

    def _authorize(self, owner_only: bool = False) -> bool:
        """Remote callers always need the token. Local callers need it for owner-only actions."""
        if owner_only or not self._is_local():
            if not self._token_ok():
                self._send_json({"error": "unauthorized"}, status=401)
                return False
        return True

    def _send_json(self, data, status: int = 200):
        payload = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _query(self) -> dict:
        from urllib.parse import urlparse, parse_qs
        return {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}

    def _route(self) -> str:
        return self.path.split("?", 1)[0]

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length > MAX_BODY:
            raise ValueError("body too large")
        return self.rfile.read(length) if length else b""

    # ---- GET
    def do_GET(self):
        route = self._route()
        if route in ("/", "/index.html"):
            if "token" in self._query():
                if self._token_ok():
                    self.send_response(302)
                    secure = "; Secure" if self.headers.get("X-Forwarded-Proto") == "https" else ""
                    self.send_header("Set-Cookie", f"evora_token={self._presented_token()}; Path=/; HttpOnly; SameSite=Strict{secure}")
                    self.send_header("Location", "/")
                    self.end_headers()
                    return
            if not self._is_local() and not self._token_ok():
                self._send_login()
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(_load_ui().encode("utf-8"))
        elif route == "/api/status":
            if not self._authorize():
                return
            try:
                data = chat_session.status()
                data["voice_server"] = voice_mod.available()
                data["owner"] = self._token_ok()
            except Exception as e:
                data = {"error": str(e), "provider": "none", "model": "none"}
            self._send_json(data)
        elif route == "/api/approvals":
            if self._authorize():
                self._send_json({"pending": chat_session.governance.broker.list_pending()})
        elif route == "/api/autonomy":
            if self._authorize():
                self._send_json(chat_session.governance.describe())
        elif route == "/api/audit":
            if self._authorize(owner_only=True):
                limit = int(self._query().get("limit", 100))
                ok, count = chat_session.governance.audit.verify()
                self._send_json({"intact": ok, "records": count,
                                 "entries": chat_session.governance.audit.tail(min(limit, 500))})
        elif route == "/api/trash":
            if self._authorize(owner_only=True):
                self._send_json({"items": chat_session.governance.trash.list()})
        elif route == "/api/vault":
            if self._authorize(owner_only=True):
                self._send_json(chat_session.governance.vault.status())
        elif route == "/api/weaknesses":
            if self._authorize(owner_only=True):
                self._send_json({"items": chat_session.governance.tracker.top(30)})
        elif route.startswith("/api/chat/stream"):
            if self._authorize():
                self._handle_stream()
        else:
            self.send_error(404, "Not found")

    def _send_login(self):
        page = ("<!doctype html><meta name=viewport content='width=device-width,initial-scale=1'>"
                "<title>EVORA</title><body style='background:#07080d;color:#e2e8f0;font-family:system-ui;"
                "display:grid;place-items:center;height:100vh;margin:0'><form onsubmit=\"location='/?token='+"
                "encodeURIComponent(t.value);return false\" style='text-align:center'><h2>EVORA</h2>"
                "<input id=t type=password placeholder='Owner token' autofocus style='padding:12px;border-radius:10px;"
                "border:1px solid #334;background:#0e1018;color:#fff;width:260px'><br><br>"
                "<button style='padding:10px 24px;border-radius:10px;border:0;background:#6366f1;color:#fff'>Enter</button>"
                "</form></body>")
        data = page.encode("utf-8")
        self.send_response(401)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # ---- POST
    def do_POST(self):
        route = self._route()
        try:
            raw = self._read_body()
        except ValueError:
            self._send_json({"error": "body too large"}, status=413)
            return
        if route == "/api/transcribe":
            if not self._authorize():
                return
            try:
                result = voice_mod.transcribe(raw, language=self._query().get("lang"),
                                              suffix=self._query().get("ext", ".webm"))
                self._send_json(result)
            except voice_mod.VoiceUnavailable as e:
                self._send_json({"error": str(e), "fallback": "browser"}, status=501)
            except Exception as e:
                self._send_json({"error": f"transcription failed: {e}", "fallback": "browser"}, status=500)
            return

        try:
            body = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            self._send_json({"error": "invalid JSON"}, status=400)
            return

        if route == "/api/chat":
            if not self._authorize():
                return
            message = body.get("message", "")
            if not message:
                self._send_json({"error": "Empty message"})
                return
            try:
                response = asyncio.run_coroutine_threadsafe(
                    _run_chat_message(message), _get_event_loop()
                ).result()
                self._send_json(response)
            except Exception as e:
                self._send_json({"error": str(e)})
        elif route == "/api/clear":
            if not self._authorize():
                return
            try:
                chat_session.clear()
                self._send_json({"status": "ok"})
            except Exception as e:
                self._send_json({"error": str(e)})
        elif route == "/api/approve":
            # Owner-only: the agent can never approve its own request.
            if not self._authorize(owner_only=True):
                return
            ok = chat_session.governance.broker.resolve(str(body.get("id", "")), str(body.get("decision", "")))
            self._send_json({"ok": ok}, status=200 if ok else 404)
        elif route == "/api/autonomy":
            if not self._authorize(owner_only=True):
                return
            try:
                chat_session.governance.set_level(int(body.get("level")))
                self._send_json(chat_session.governance.describe())
            except (TypeError, ValueError):
                self._send_json({"error": "level must be 0-3"}, status=400)
        elif route in ("/api/vault/unlock", "/api/vault/lock", "/api/vault/set", "/api/vault/delete"):
            if not self._authorize(owner_only=True):
                return
            vault = chat_session.governance.vault
            try:
                if route.endswith("/unlock"):
                    if not vault.unlock(str(body.get("passphrase", ""))):
                        self._send_json({"error": "wrong passphrase"}, status=403)
                        return
                elif route.endswith("/lock"):
                    vault.lock()
                elif route.endswith("/set"):
                    vault.set(str(body.get("name", "")).strip(), str(body.get("value", "")))
                    chat_session.governance.audit.record("vault_set", name=body.get("name"), by="owner")
                else:
                    vault.delete(str(body.get("name", "")))
                    chat_session.governance.audit.record("vault_delete", name=body.get("name"), by="owner")
                self._send_json(vault.status())
            except Exception as e:
                self._send_json({"error": str(e)}, status=400)
        elif route == "/api/weaknesses/mark":
            if not self._authorize(owner_only=True):
                return
            ok = chat_session.governance.tracker.mark(str(body.get("id", "")), str(body.get("status", "")))
            self._send_json({"ok": ok}, status=200 if ok else 404)
        elif route == "/api/trash/restore":
            if not self._authorize(owner_only=True):
                return
            try:
                path = chat_session.governance.trash.restore(str(body.get("id", "")))
                chat_session.governance.audit.record("trash_restored", id=body.get("id"), path=path, by="owner")
                self._send_json({"restored": path})
            except Exception as e:
                self._send_json({"error": str(e)}, status=400)
        else:
            self.send_error(404, "Not found")

    def _handle_stream(self):
        """SSE: /api/chat/stream?message=...&mode=agent|plain"""
        from queue import Queue
        params = self._query()
        message = params.get("message")
        if not message:
            self._send_json({"error": "Empty message"})
            return
        agent_mode = params.get("mode", "agent") != "plain"

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        loop = _get_event_loop()
        event_queue: Queue = Queue()

        async def collect_events():
            try:
                async for event in _stream_chat_message(message, agent=agent_mode):
                    event_queue.put(event)
            except Exception as e:
                event_queue.put({"type": "error", "error": str(e)})

        asyncio.run_coroutine_threadsafe(collect_events(), loop)

        while True:
            event = event_queue.get()
            event_type = event.get("type", "content")
            data = {k: v for k, v in event.items() if k != "type"}
            line = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
            try:
                self.wfile.write(line.encode("utf-8"))
                self.wfile.flush()
            except Exception:
                break
            if event_type in ("done", "error"):
                break

    def log_message(self, format, *args):
        pass  # suppress default HTTP logging


def _load_ui() -> str:
    path = Path(__file__).with_name("web") / "index.html"
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return CHAT_HTML


class ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


chat_session: Optional[ChatSession] = None
_event_loop: Optional[asyncio.AbstractEventLoop] = None
_event_loop_thread: Optional[threading.Thread] = None


def _get_event_loop() -> asyncio.AbstractEventLoop:
    """Get or create the persistent event loop for the chat session."""
    global _event_loop, _event_loop_thread
    if _event_loop is None or _event_loop.is_closed():
        _event_loop = asyncio.new_event_loop()
        _event_loop_thread = threading.Thread(
            target=_event_loop.run_forever, daemon=True
        )
        _event_loop_thread.start()
    return _event_loop


def _find_free_port(start=8080, max_tries=10, host="127.0.0.1"):
    for port in range(start, start + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue
    raise OSError(f"No free port found in range {start}-{start + max_tries - 1}")


def start_chat_server(config=None, logger=None, port: int = 8080, provider_override: Optional[str] = None,
                      host: str = "127.0.0.1", open_browser: bool = True):
    """Start the EVORA owner console.

    host="127.0.0.1" keeps it on this computer. Use host="0.0.0.0" (or an ngrok tunnel to the port)
    to reach it from other devices: every remote request then needs the owner token.
    """
    global chat_session
    chat_session = ChatSession(config=config, logger=logger, provider_override=provider_override)

    actual_port = _find_free_port(port, host="127.0.0.1" if host in ("0.0.0.0", "") else host)
    server = ThreadedHTTPServer((host, actual_port), ChatHandler)
    token = get_token()
    url = f"http://127.0.0.1:{actual_port}"

    print(f"EVORA Console running at {url}")
    if host not in ("127.0.0.1", "localhost"):
        print(f"Remote access is ON ({host}). Owner token (keep secret): {token}")
        print("Remote devices open the site and enter this token once.")
    print("Press Ctrl+C to stop.")

    if open_browser:
        try:
            webbrowser.open(f"{url}/?token={token}")
        except Exception:
            pass

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    try:
        while True:
            server_thread.join(timeout=1.0)
            if not server_thread.is_alive():
                break
    except KeyboardInterrupt:
        pass
    finally:
        _shutdown()

    return server


def _shutdown():
    """Clean shutdown of the event loop and provider clients."""
    global _event_loop, _event_loop_thread
    if chat_session is not None:
        chat_session.close()
    if _event_loop is not None and not _event_loop.is_closed():
        _event_loop.call_soon_threadsafe(_event_loop.stop)
        if _event_loop_thread is not None:
            _event_loop_thread.join(timeout=5.0)
        try:
            _event_loop.close()
        except Exception:
            pass
        _event_loop = None
        _event_loop_thread = None
