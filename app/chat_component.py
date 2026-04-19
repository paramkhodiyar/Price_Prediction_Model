# Chat panel component — self-contained HTML/CSS/JS
# Rendered via st.components.v1.html() inside an iframe.
# POST /chat endpoint built by Param; __ENDPOINT__ is replaced at render time.
# Falls back to rotating mock replies when the endpoint is unavailable.

_CHAT_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
  @import url('https://fonts.googleapis.com/css2?family=Public+Sans:wght@300;400;500;600;700&display=swap');

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: 'Public Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    background: #F8F9FB;
    height: 100vh;
    padding: 0;
    overflow: hidden;
  }

  /* ── Outer shell ─────────────────────────────────────── */
  .chat-shell {
    display: flex;
    flex-direction: column;
    height: 100vh;
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    overflow: hidden;
    box-shadow: 0 4px 24px rgba(0,0,0,0.06);
  }

  /* ── Header bar ──────────────────────────────────────── */
  .chat-header {
    background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%);
    color: #FFFFFF;
    padding: 1rem 1.4rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    flex-shrink: 0;
    user-select: none;
  }
  .chat-header .avatar {
    width: 36px;
    height: 36px;
    background: rgba(255,255,255,0.2);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    flex-shrink: 0;
  }
  .chat-header .info { flex: 1; }
  .chat-header .info .name {
    font-weight: 700;
    font-size: 0.95rem;
    line-height: 1.2;
  }
  .chat-header .info .sub {
    font-size: 0.72rem;
    opacity: 0.85;
    font-weight: 400;
  }
  .status-dot {
    width: 8px;
    height: 8px;
    background: #A7F3D0;
    border-radius: 50%;
    flex-shrink: 0;
    animation: pulse-dot 2.5s infinite;
  }
  @keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.5; transform: scale(0.85); }
  }

  /* ── Messages area ───────────────────────────────────── */
  .messages-area {
    flex: 1;
    overflow-y: auto;
    padding: 1.25rem;
    display: flex;
    flex-direction: column;
    gap: 1rem;
    background: #F8F9FB;
    scroll-behavior: smooth;
  }
  .messages-area::-webkit-scrollbar { width: 4px; }
  .messages-area::-webkit-scrollbar-track { background: transparent; }
  .messages-area::-webkit-scrollbar-thumb {
    background: #CBD5E1;
    border-radius: 2px;
  }

  /* ── Message rows ────────────────────────────────────── */
  .msg-row {
    display: flex;
    flex-direction: column;
    max-width: 82%;
    animation: slideUp 0.22s ease-out;
  }
  @keyframes slideUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  .msg-row.user { align-self: flex-end; align-items: flex-end; }
  .msg-row.bot  { align-self: flex-start; align-items: flex-start; }

  .msg-label {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: #94A3B8;
    margin-bottom: 0.3rem;
  }

  /* ── Bubble ──────────────────────────────────────────── */
  .bubble {
    padding: 0.75rem 1rem;
    border-radius: 14px;
    font-size: 0.875rem;
    line-height: 1.65;
    word-break: break-word;
  }
  .user .bubble {
    background: #1E293B;
    color: #F1F5F9;
    border-bottom-right-radius: 4px;
  }
  .bot .bubble {
    background: #FFFFFF;
    color: #1E293B;
    border: 1px solid #E2E8F0;
    border-bottom-left-radius: 4px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
  }

  /* HTML report bubble ─────────────────────────────────── */
  .bubble.is-report {
    padding: 0;
    overflow: hidden;
    border: 1.5px solid #F59E0B;
    max-width: 100%;
    width: 100%;
  }
  .bubble.is-report .report-body { padding: 1rem; }
  .copy-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.4rem;
    width: 100%;
    padding: 0.55rem;
    background: #FEF9C3;
    border: none;
    border-top: 1px solid #FDE68A;
    color: #92400E;
    font-size: 0.78rem;
    font-weight: 700;
    cursor: pointer;
    font-family: inherit;
    transition: background 0.15s;
    letter-spacing: 0.3px;
  }
  .copy-btn:hover { background: #FDE68A; }

  /* ── Typing indicator ────────────────────────────────── */
  .typing-row {
    display: none;
    align-self: flex-start;
    animation: slideUp 0.18s ease-out;
  }
  .typing-row.visible { display: flex; }
  .typing-dots {
    display: flex;
    align-items: center;
    gap: 5px;
    padding: 0.7rem 1rem;
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    border-bottom-left-radius: 4px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
  }
  .dot {
    width: 7px;
    height: 7px;
    background: #F59E0B;
    border-radius: 50%;
    animation: bounce-dot 1.3s infinite ease-in-out;
  }
  .dot:nth-child(2) { animation-delay: 0.18s; }
  .dot:nth-child(3) { animation-delay: 0.36s; }
  @keyframes bounce-dot {
    0%, 80%, 100% { transform: translateY(0);    opacity: 0.35; }
    40%            { transform: translateY(-7px); opacity: 1;    }
  }

  /* ── Input area ──────────────────────────────────────── */
  .input-area {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.85rem 1rem;
    border-top: 1px solid #E2E8F0;
    background: #FFFFFF;
    flex-shrink: 0;
  }
  #userInput {
    flex: 1;
    padding: 0.65rem 1rem;
    border: 1.5px solid #E2E8F0;
    border-radius: 10px;
    font-size: 0.875rem;
    color: #1E293B;
    background: #F8F9FB;
    outline: none;
    font-family: inherit;
    transition: border-color 0.15s, box-shadow 0.15s, background 0.15s;
    resize: none;
  }
  #userInput:focus {
    border-color: #F59E0B;
    background: #FFFFFF;
    box-shadow: 0 0 0 3px rgba(245,158,11,0.12);
  }
  #userInput:disabled { opacity: 0.5; cursor: not-allowed; }

  #sendBtn {
    padding: 0.65rem 1.15rem;
    background: #F59E0B;
    color: #FFFFFF;
    border: none;
    border-radius: 10px;
    font-size: 0.875rem;
    font-weight: 700;
    cursor: pointer;
    font-family: inherit;
    display: flex;
    align-items: center;
    gap: 0.35rem;
    transition: background 0.15s, transform 0.1s, box-shadow 0.15s;
    white-space: nowrap;
    flex-shrink: 0;
    box-shadow: 0 2px 8px rgba(245,158,11,0.3);
  }
  #sendBtn:hover   { background: #D97706; box-shadow: 0 2px 12px rgba(217,119,6,0.4); }
  #sendBtn:active  { transform: scale(0.96); }
  #sendBtn:disabled {
    background: #CBD5E1;
    cursor: not-allowed;
    transform: none;
    box-shadow: none;
  }

  /* ── Toast ───────────────────────────────────────────── */
  .toast {
    position: fixed;
    bottom: 72px;
    left: 50%;
    transform: translateX(-50%) translateY(6px);
    background: #1E293B;
    color: #F1F5F9;
    padding: 0.4rem 1rem;
    border-radius: 8px;
    font-size: 0.78rem;
    font-weight: 600;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.25s, transform 0.25s;
    z-index: 100;
  }
  .toast.show {
    opacity: 1;
    transform: translateX(-50%) translateY(0);
  }
</style>
</head>
<body>

<div class="chat-shell">

  <!-- Header -->
  <div class="chat-header">
    <div class="avatar">🏠</div>
    <div class="info">
      <div class="name">Valora Chat</div>
      <div class="sub">Powered by Llama 3.3 70B · RAG-grounded market data</div>
    </div>
    <div class="status-dot"></div>
  </div>

  <!-- Messages -->
  <div class="messages-area" id="messages">

    <!-- Dummy: bot welcome -->
    <div class="msg-row bot">
      <div class="msg-label">Valora</div>
      <div class="bubble">
        Welcome to <strong>ValoraAI</strong> 🏡 I can help you with property prices, market trends, investment analysis, and full advisory reports for any Indian metro. What would you like to know?
      </div>
    </div>

    <!-- Dummy: user question -->
    <div class="msg-row user">
      <div class="msg-label">You</div>
      <div class="bubble">What's the average price for a 2BHK in Bandra?</div>
    </div>

    <!-- Dummy: bot reply -->
    <div class="msg-row bot">
      <div class="msg-label">Valora</div>
      <div class="bubble">
        In <strong>Mumbai Bandra West</strong>, 2BHK apartments typically average <strong>₹2.1 Cr</strong> for 750–900 sqft units in 2023–24. Bandra is one of Mumbai's most premium micro-markets with consistent 8–10% annual appreciation and strong rental demand from professionals. Would you like a full BUY/HOLD/SELL advisory report for this locality?
      </div>
    </div>

    <!-- Typing indicator (hidden by default) -->
    <div class="typing-row" id="typingRow">
      <div class="typing-dots">
        <div class="dot"></div>
        <div class="dot"></div>
        <div class="dot"></div>
      </div>
    </div>

  </div><!-- /messages-area -->

  <!-- Input -->
  <div class="input-area">
    <input
      type="text"
      id="userInput"
      placeholder="Ask about prices, market trends, investment advice…"
      autocomplete="off"
      spellcheck="false"
    />
    <button id="sendBtn" onclick="sendMessage()">
      Send
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
        <line x1="22" y1="2" x2="11" y2="13"></line>
        <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
      </svg>
    </button>
  </div>

</div><!-- /chat-shell -->

<div class="toast" id="toast">Copied to clipboard!</div>

<script>
  /* ── Config ───────────────────────────────────────────────── */
  const CHAT_ENDPOINT = '__ENDPOINT__';

  /* ── Mock replies (rotated when backend is unavailable) ─────── */
  const MOCK_REPLIES = [
    "Based on current market data, this locality shows strong appreciation potential with 8–12% annual growth. Would you like me to generate a full advisory report?",
    "Great question! Properties near metro corridors and IT hubs continue to outperform in 2024. Key factors include connectivity, social infrastructure, and upcoming developments.",
    "For investment purposes, 2BHK and 3BHK apartments in tier-1 metros like Mumbai, Bangalore, and Gurgaon offer the best liquidity and rental yields of 3–4%.",
    "The premium segment (above ₹2 Cr) is seeing increased NRI and tech-executive demand, while affordable housing under ₹60 Lac is thriving in peripheral zones.",
    "I can generate a detailed HTML advisory report with a BUY / HOLD / SELL verdict for any specific property. Just fill in the form above and click <strong>Generate AI Advisory Report</strong>!",
    "Properties below 5 years of age command a 10–20% premium over older stock. Furnished units also deliver 8–12% higher rental yields. Floor-level premiums apply above the 10th floor.",
  ];
  let mockIdx = 0;

  /* ── Helpers ─────────────────────────────────────────────── */
  function scrollToBottom() {
    const area = document.getElementById('messages');
    area.scrollTo({ top: area.scrollHeight, behavior: 'smooth' });
  }

  function setTyping(visible) {
    const row = document.getElementById('typingRow');
    row.classList.toggle('visible', visible);
    if (visible) scrollToBottom();
  }

  function setLocked(locked) {
    document.getElementById('userInput').disabled = locked;
    document.getElementById('sendBtn').disabled   = locked;
  }

  function showToast(msg) {
    const t = document.getElementById('toast');
    t.textContent = msg;
    t.classList.add('show');
    setTimeout(() => t.classList.remove('show'), 2200);
  }

  /* ── Append a message bubble ─────────────────────────────── */
  function appendMessage(content, role, isHtml) {
    isHtml = isHtml || (typeof content === 'string' && content.trim().startsWith('<div'));

    const area     = document.getElementById('typingRow');   // insert before typing indicator
    const messages = document.getElementById('messages');

    const row = document.createElement('div');
    row.className = 'msg-row ' + role;

    const label = document.createElement('div');
    label.className = 'msg-label';
    label.textContent = (role === 'user') ? 'You' : 'Valora';
    row.appendChild(label);

    const bubble = document.createElement('div');
    bubble.className = 'bubble';

    if (isHtml) {
      bubble.classList.add('is-report');

      const body = document.createElement('div');
      body.className = 'report-body';
      body.innerHTML = content;
      bubble.appendChild(body);

      const copyBtn = document.createElement('button');
      copyBtn.className = 'copy-btn';
      copyBtn.innerHTML = '&#128203; Copy Report';
      copyBtn.onclick = function() {
        navigator.clipboard.writeText(content)
          .then(() => showToast('Report copied to clipboard!'))
          .catch(() => showToast('Copy failed — please select & copy manually.'));
      };
      bubble.appendChild(copyBtn);
    } else {
      bubble.innerHTML = content;
    }

    row.appendChild(bubble);
    messages.insertBefore(row, area);   // keep typing indicator at end
    scrollToBottom();
  }

  /* ── Send message ─────────────────────────────────────────── */
  async function sendMessage() {
    const input = document.getElementById('userInput');
    const text  = input.value.trim();
    if (!text) return;

    appendMessage(text, 'user', false);
    input.value = '';
    setTyping(true);
    setLocked(true);

    try {
      const resp = await fetch(CHAT_ENDPOINT, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ message: text }),
      });

      if (!resp.ok) throw new Error('HTTP ' + resp.status);
      const data = await resp.json();

      const isHtml = data.is_html ||
        (typeof data.reply === 'string' && data.reply.trim().startsWith('<'));
      appendMessage(data.reply, 'bot', isHtml);

    } catch (err) {
      // /chat not yet live — graceful mock fallback
      console.info('[ValoraAI Chat] Backend not available, using mock reply.', err.message);
      const delay = 1000 + Math.random() * 900;
      await new Promise(r => setTimeout(r, delay));
      appendMessage(MOCK_REPLIES[mockIdx++ % MOCK_REPLIES.length], 'bot', false);
    } finally {
      setTyping(false);
      setLocked(false);
      input.focus();
    }
  }

  /* ── Keyboard: Enter to send ─────────────────────────────── */
  document.getElementById('userInput').addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  /* Initial scroll */
  scrollToBottom();
</script>

</body>
</html>
"""


def get_chat_html(chat_endpoint: str = "http://localhost:8000/chat") -> str:
    """Return the chat panel HTML with the endpoint URL injected."""
    return _CHAT_TEMPLATE.replace("__ENDPOINT__", chat_endpoint)
