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

/* ---- 发送 ---- */
function autoGrow() {
  input.style.height = 'auto';
  input.style.height = Math.min(input.scrollHeight, 160) + 'px';
}

async function send() {
  const q = input.value.trim();
  if (!q) return;

  addUser(q);
  input.value = '';
  autoGrow();

  const pending = document.createElement('div');
  pending.className = 'bubble assistant pending';
  pending.textContent = '正在检索并生成回答';
  addMessage('assistant', pending);

  const topk = parseInt(topkSel.value, 10) || 4;
  try {
    const r = await fetch('/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q, top_k: topk }),
    });
    const data = await r.json();
    pending.closest('.msg').remove();
    addAssistant(data.answer, data.sources, data.contexts);
  } catch (e) {
    pending.closest('.msg').remove();
    const err = document.createElement('div');
    err.className = 'bubble assistant error';
    err.textContent = '请求失败：' + e;
    addMessage('assistant', err);
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
