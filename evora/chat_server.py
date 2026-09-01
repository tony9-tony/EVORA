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
import socket
import socketserver
import threading
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional

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
      if (!hasError) {
        finalizeStreamingBubble(null);
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
      identityPill.textContent = data.identity + ' (' + data.authority + ')';
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


async def _stream_chat_message(message: str):
    """Process a chat message and stream the response token-by-token.

    Yields event dicts of the form:
        {"type": "content", "content": "chunk"}
        {"type": "tool", "name": "...", "output": "..."}
        {"type": "error", "error": "..."}
    """
    try:
        async for event in chat_session.stream_message(message):
            yield event
    except Exception as e:
        yield {"type": "error", "error": str(e)}


class ChatHandler(BaseHTTPRequestHandler):
    """HTTP handler for the EVORA chat server."""

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(CHAT_HTML.encode("utf-8"))
        elif self.path == "/api/status":
            try:
                data = chat_session.status()
            except Exception as e:
                data = {"error": str(e), "provider": "none", "model": "none"}
            self._send_json(data)
        elif self.path.startswith("/api/chat/stream"):
            self._handle_stream()
        else:
            self.send_error(404, "Not found")

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        if self.path == "/api/chat":
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
        elif self.path == "/api/clear":
            try:
                chat_session.clear()
                self._send_json({"status": "ok"})
            except Exception as e:
                self._send_json({"error": str(e)})
        else:
            self.send_error(404, "Not found")

    def _handle_stream(self):
        """Handle SSE streaming request at /api/chat/stream?message=..."""
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        message = params.get("message", [None])[0]
        if not message:
            self._send_json({"error": "Empty message"})
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        loop = _get_event_loop()

        async def stream_to_response():
            try:
                async for event in _stream_chat_message(message):
                    event_type = event.get("type", "content")
                    data = json.dumps({k: v for k, v in event.items() if k != "type"})
                    line = f"event: {event_type}\ndata: {data}\n\n"
                    self.wfile.write(line.encode("utf-8"))
                    self.wfile.flush()
            except Exception as e:
                err_data = {"error": str(e)}
                line = f"event: error\ndata: {json.dumps(err_data)}\n\n"
                self.wfile.write(line.encode("utf-8"))
                self.wfile.flush()
            finally:
                line = "event: done\ndata: {}\n\n"
                self.wfile.write(line.encode("utf-8"))
                self.wfile.flush()

        def run_stream():
            try:
                asyncio.run_coroutine_threadsafe(
                    stream_to_response(), loop
                ).result()
            except Exception:
                pass

        threading.Thread(target=run_stream, daemon=True).start()

    def _send_json(self, data):
        payload = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        pass  # suppress default HTTP logging


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


def _find_free_port(start=8080, max_tries=10):
    for port in range(start, start + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise OSError(f"No free port found in range {start}-{start + max_tries - 1}")


def start_chat_server(config=None, logger=None, port: int = 8080, provider_override: Optional[str] = None):
    """Start the EVORA chat web server and open the browser.

    Returns the server instance so callers can shut it down cleanly.
    """
    global chat_session
    chat_session = ChatSession(config=config, logger=logger, provider_override=provider_override)

    actual_port = _find_free_port(port)
    server = ThreadedHTTPServer(("127.0.0.1", actual_port), ChatHandler)
    url = f"http://127.0.0.1:{actual_port}"

    print(f"EVORA Chat running at {url}")
    print("Press Ctrl+C to stop.")

    try:
        webbrowser.open(url)
    except Exception:
        pass

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    # Keep the main thread alive so the server can handle requests
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
