/* 前端逻辑：调用后端 /ask，渲染问答与来源。 */
const chat = document.getElementById('chat');
const input = document.getElementById('input');
const sendBtn = document.getElementById('send');
const topkSel = document.getElementById('topk');
const statusText = document.getElementById('status-text');
const statusDot = document.getElementById('status-dot');

/* ---- 探活 ---- */
function setStatus(ok, text) {
  statusText.textContent = text;
  statusDot.classList.toggle('off', !ok);
}

async function checkHealth() {
  try {
    const r = await fetch('/health');
    setStatus(r.ok, r.ok ? '服务在线' : '服务异常');
  } catch {
    setStatus(false, '服务离线');
  }
}

/* ---- 渲染辅助 ---- */
function hideWelcome() {
  const w = document.getElementById('welcome');
  if (w) w.style.display = 'none';
}

function scrollBottom() {
  chat.scrollTop = chat.scrollHeight;
}

function addMessage(role, node) {
  const wrap = document.createElement('div');
  wrap.className = 'msg ' + role;
  wrap.appendChild(node);
  chat.appendChild(wrap);
  scrollBottom();
}

function addUser(text) {
  const div = document.createElement('div');
  div.className = 'bubble user';
  div.textContent = text;
  addMessage('user', div);
  hideWelcome();
}

function sourceCard(i, s, ctx) {
  const card = document.createElement('details');
  card.className = 'source-card';
  const summary = document.createElement('summary');

  const num = document.createElement('span');
  num.className = 'src-num';
  num.textContent = '来源' + (i + 1);

  const path = document.createElement('span');
  path.className = 'src-path';
  path.textContent = s.section ? `${s.source} · ${s.section}` : s.source;

  const dist = document.createElement('span');
  dist.className = 'src-dist';
  dist.textContent = '距离 ' + Number(s.distance).toFixed(3);

  summary.append(num, path, dist);
  card.appendChild(summary);

  if (ctx) {
    const body = document.createElement('div');
    body.className = 'src-body';
    body.textContent = ctx;
    card.appendChild(body);
  }
  return card;
}

function addAssistant(ans, sources, contexts) {
  const body = document.createElement('div');
  body.className = 'assistant-body';

  if (!ans) {
    const note = document.createElement('div');
    note.className = 'bubble assistant note';
    note.textContent = '（未配置 LLM_API_KEY，以下仅为检索到的资料片段）';
    body.appendChild(note);
    (sources || []).forEach((s, i) => body.appendChild(sourceCard(i, s, contexts && contexts[i])));
  } else {
    const bubble = document.createElement('div');
    bubble.className = 'bubble assistant markdown';
    bubble.innerHTML = renderMarkdown(ans);
    body.appendChild(bubble);

    if (sources && sources.length) {
      const wrap = document.createElement('div');
      wrap.className = 'sources';
      const title = document.createElement('div');
      title.className = 'sources-title';
      title.textContent = `参考来源（${sources.length}）`;
      wrap.appendChild(title);
      sources.forEach((s, i) => wrap.appendChild(sourceCard(i, s, contexts && contexts[i])));
      body.appendChild(wrap);
    }
  }

  addMessage('assistant', body);
}

/* ---- Markdown（安全转义后做轻量渲染）---- */
function escapeHtml(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function renderMarkdown(md) {
  const lines = escapeHtml(md).split('\n');
  let html = '';
  let listType = null;
  let inCode = false;
  const codeBuf = [];

  const closeList = () => {
    if (listType) { html += '</' + listType + '>'; listType = null; }
  };
  const inline = (t) => t
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/\*([^*]+)\*/g, '<em>$1</em>');

  for (const line of lines) {
    if (line.trim().startsWith('```')) {
      if (inCode) {
        html += '<pre><code>' + codeBuf.join('\n') + '</code></pre>';
        codeBuf.length = 0;
        inCode = false;
      } else {
        closeList();
        inCode = true;
      }
      continue;
    }
    if (inCode) { codeBuf.push(line); continue; }

    const h = line.match(/^(#{1,4})\s+(.*)$/);
    if (h) {
      closeList();
      const lvl = h[1].length + 2; // # -> h3
      html += `<h${lvl}>${inline(h[2])}</h${lvl}>`;
      continue;
    }
    const ul = line.match(/^\s*[-*+]\s+(.*)$/);
    if (ul) {
      if (listType !== 'ul') { closeList(); html += '<ul>'; listType = 'ul'; }
      html += '<li>' + inline(ul[1]) + '</li>';
      continue;
    }
    const ol = line.match(/^\s*\d+[.)]\s+(.*)$/);
    if (ol) {
      if (listType !== 'ol') { closeList(); html += '<ol>'; listType = 'ol'; }
      html += '<li>' + inline(ol[1]) + '</li>';
      continue;
    }
    if (line.trim() === '') { closeList(); continue; }
    closeList();
    html += '<p>' + inline(line) + '</p>';
  }
  if (inCode) html += '<pre><code>' + codeBuf.join('\n') + '</code></pre>';
  closeList();
  return html;
}

function showError(text) {
  const err = document.createElement('div');
  err.className = 'bubble assistant error';
  err.textContent = text;
  addMessage('assistant', err);
}

/* ---- 发送 ---- */
function autoGrow() {
  input.style.height = 'auto';
  input.style.height = Math.min(input.scrollHeight, 160) + 'px';
}

/* 流式回答：先出气泡，边收边渲染，末尾补来源卡片。 */
async function send() {
  const q = input.value.trim();
  if (!q) return;

  addUser(q);
  input.value = '';
  autoGrow();

  const t0 = performance.now();
  const pending = document.createElement('div');
  pending.className = 'bubble assistant pending';
  const elapsed = document.createElement('span');
  elapsed.className = 'pending-elapsed';
  pending.append('正在检索', elapsed);
  addMessage('assistant', pending);
  const timer = setInterval(() => {
    elapsed.textContent = ' ' + ((performance.now() - t0) / 1000).toFixed(1) + 's';
  }, 100);

  const body = document.createElement('div');
  body.className = 'assistant-body';
  const bubble = document.createElement('div');
  bubble.className = 'bubble assistant markdown';
  let sources = [];
  let contexts = [];
  let text = '';
  let frame = null;

  const paint = () => {
    frame = null;
    bubble.innerHTML = renderMarkdown(text);
    scrollBottom();
  };

  const flush = () => {
    if (frame) return;
    frame = requestAnimationFrame(paint);
  };

  const finish = () => {
    clearInterval(timer);
    if (frame) cancelAnimationFrame(frame);
    paint();
    if (sources.length) {
      const wrap = document.createElement('div');
      wrap.className = 'sources';
      const title = document.createElement('div');
      title.className = 'sources-title';
      title.textContent = `参考来源（${sources.length}）`;
      wrap.appendChild(title);
      sources.forEach((s, i) => wrap.appendChild(sourceCard(i, s, contexts && contexts[i])));
      body.appendChild(wrap);
    }
    scrollBottom();
  };

  const topk = parseInt(topkSel.value, 10) || 4;
  try {
    const r = await fetch('/ask/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q, top_k: topk }),
    });
    if (!r.ok) {
      let data = null;
      try { data = await r.json(); } catch { /* 非 JSON 响应 */ }
      clearInterval(timer);
      pending.closest('.msg').remove();
      const msg = (data && data.detail) || ('HTTP ' + r.status);
      showError('请求失败：' + (typeof msg === 'string' ? msg : JSON.stringify(msg)));
      return;
    }

    const reader = r.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    let started = false;

    const onEvent = (payload) => {
      if (payload.type === 'sources') {
        sources = payload.sources || [];
        contexts = payload.contexts || [];
        if (!started) {
          started = true;
          pending.closest('.msg').remove();
          clearInterval(timer);
          body.appendChild(bubble);
          addMessage('assistant', body);
        }
      } else if (payload.type === 'delta') {
        if (!started) {
          started = true;
          pending.closest('.msg').remove();
          clearInterval(timer);
          body.appendChild(bubble);
          addMessage('assistant', body);
        }
        text += payload.text || '';
        flush();
      } else if (payload.type === 'error') {
        text += `\n\n（${payload.message}）`;
        flush();
      }
    };

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf('\n\n')) >= 0) {
        const raw = buf.slice(0, idx).trim();
        buf = buf.slice(idx + 2);
        if (!raw.startsWith('data: ')) continue;
        try { onEvent(JSON.parse(raw.slice(6))); } catch { /* 跳过坏帧 */ }
      }
    }

    if (!started) {
      pending.closest('.msg').remove();
      showError('请求失败：服务未返回内容');
      return;
    }
    finish();
  } catch (e) {
    clearInterval(timer);
    pending.closest('.msg').remove();
    showError('请求失败：' + e);
  }
}

/* ---- 事件绑定 ---- */
sendBtn.addEventListener('click', send);
input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});
input.addEventListener('input', autoGrow);
document.querySelectorAll('.chip').forEach((c) => {
  c.addEventListener('click', () => { input.value = c.dataset.q; send(); });
});

checkHealth();
setInterval(checkHealth, 30000);
