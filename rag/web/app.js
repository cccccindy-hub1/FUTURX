/* 前端逻辑：调用统一入口 /auto/ask/stream（LLM 路由 → 课程 / 论文 / 政策 / 判例 / 名言），流式渲染回答与来源。
   事件：route →（课程等需要作答的库）stage / materials / delta → done；出错时 error。
   正文里的引用是序号 [1] [2]，渲染成上标，悬停显示来源路径；done 到达后用 done.answer 覆盖正文，保证与最终结果一致。 */
const chat = document.getElementById('chat');
const input = document.getElementById('input');
const sendBtn = document.getElementById('send');
const topkSel = document.getElementById('topk');
const keyInput = document.getElementById('apikey');
const statusText = document.getElementById('status-text');
const statusDot = document.getElementById('status-dot');

const ROUTE_LABELS = { course: '课程', thesis: '论文', policy: '政策', case: '判例', quote: '名言', exam: '试卷', meta: '助手介绍', off_topic: '库外' };
const KIND_LABELS = { course: '教材', case: '判例', quote: '讲话', policy: '政策' };
const KEY_STORE = 'futurx_api_key';
const CONV_STORE = 'futurx_conv_by_key';   // { "<api key>": "conv_xxx" }，会话按 key 隔离

/* ---- 当前会话：跟着 API key 走，换 key 就换一套历史 ---- */
let conversationId = null;

function currentKey() {
  return (keyInput && keyInput.value.trim()) || loadKey() || '';
}

function readConvMap() {
  try { return JSON.parse(localStorage.getItem(CONV_STORE) || '{}') || {}; } catch { return {}; }
}

function loadConvId() {
  return readConvMap()[currentKey()] || null;
}

function saveConvId(id) {
  try {
    const m = readConvMap();
    if (id) m[currentKey()] = id; else delete m[currentKey()];
    localStorage.setItem(CONV_STORE, JSON.stringify(m));
  } catch { /* 隐私模式等 */ }
}

function forgetConvId() {
  conversationId = null;
  saveConvId(null);
}

function authHeaders(extra) {
  const h = extra || {};
  const key = currentKey();
  if (key) h.Authorization = 'Bearer ' + key;
  return h;
}

/* ---- 访问密钥：只存在本浏览器 ---- */
function loadKey() {
  try { return localStorage.getItem(KEY_STORE) || ''; } catch { return ''; }
}
function saveKey(v) {
  try { v ? localStorage.setItem(KEY_STORE, v) : localStorage.removeItem(KEY_STORE); } catch { /* 隐私模式等 */ }
}
if (keyInput) {
  keyInput.value = loadKey();
  keyInput.addEventListener('change', () => {
    saveKey(keyInput.value.trim());
    // 换 key = 换一个人，历史与当前会话都要跟着换
    conversationId = loadConvId();
    clearChat();
    refreshConversations();
  });
}

/* ---- 用户标识：随机 ID 存在本浏览器（长期），随请求放在 X-User-Id 头里，只用于永久问答记录（rag.qa_store）。多轮对话由 conversation_id 负责 ---- */
const UID_STORE = 'futurx_user_id';
let sessionUid = '';
function randomId(prefix) {
  let s = '';
  try {
    const a = new Uint8Array(12);
    crypto.getRandomValues(a);
    s = Array.from(a, (b) => b.toString(16).padStart(2, '0')).join('');
  } catch { s = Math.random().toString(16).slice(2) + Date.now().toString(16); }
  return `${prefix}-${s}`;
}
function userId() {
  try {
    let v = localStorage.getItem(UID_STORE);
    if (!v || !/^[A-Za-z0-9_-]{1,64}$/.test(v)) { v = randomId('u'); localStorage.setItem(UID_STORE, v); }
    return v;
  } catch { return sessionUid || (sessionUid = randomId('u')); }   // 隐私模式：本页面生命周期内稳定
}

/* ---- 探活 ---- */
function setStatus(ok, text) {
  statusText.textContent = text;
  const st = document.getElementById('status');
  if (st) st.title = text;                      // 窄屏只显示圆点时，悬停 / 长按仍能看到文字
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

function showWelcome() {
  const w = document.getElementById('welcome');
  if (w) w.style.display = '';
}

/* 清空对话区（保留欢迎页节点本身，否则“新对话”后无法再显示引导） */
function clearChat() {
  [...chat.children].forEach((n) => { if (n.id !== 'welcome') n.remove(); });
  showWelcome();
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
  return wrap;
}

function addUser(text) {
  const div = document.createElement('div');
  div.className = 'bubble user';
  div.textContent = text;
  addMessage('user', div);
  hideWelcome();
}

/* 论文库来源（原 /ask 形状：source / section / distance，可带原文） */
function sourceCard(i, s, ctx) {
  const card = document.createElement('details');
  card.className = 'source-card';
  const summary = document.createElement('summary');

  const num = document.createElement('span');
  num.className = 'src-num';
  num.textContent = `[${i + 1}]`;

  const path = document.createElement('span');
  path.className = 'src-path';
  path.textContent = s.section ? `${s.source} · ${s.section}` : s.source;

  summary.append(num, path);
  if (s.distance !== undefined && s.distance !== null) {
    const dist = document.createElement('span');
    dist.className = 'src-dist';
    dist.textContent = '距离 ' + Number(s.distance).toFixed(3);
    summary.append(dist);
  }
  card.appendChild(summary);

  if (ctx) {
    const body = document.createElement('div');
    body.className = 'src-body';
    body.textContent = ctx;
    card.appendChild(body);
  }
  return card;
}

/* 带序号的来源（课程 / 判例 / 名言）：[n] 路径；有 kind 且不止一种时按库分区 */
// 一道题：题干 + 选项（选择题）+ 默认收起的“查看答案”（答案与评分要点 / 解析由后端原样给出，不经过模型）
function questionBlock(it, i, withMeta) {
  const box = document.createElement('div');
  box.className = 'exam-q';
  const head = document.createElement('div');
  head.className = 'src-row';
  const n = document.createElement('span');
  n.className = 'src-n';
  n.textContent = `${i + 1}.`;
  const t = document.createElement('span');
  t.className = 'src-row-path';
  const meta = withMeta && it.chapter ? `·第${it.chapter}章${it.chapter_title ? ' ' + it.chapter_title : ''}` : '';
  t.textContent = `【${it.qtype_cn || '题目'}${meta}】${it.stem || ''}`;
  head.append(n, t);
  box.appendChild(head);
  const opts = it.options && typeof it.options === 'object' ? Object.entries(it.options) : [];
  if (opts.length) {
    const ul = document.createElement('div');
    ul.className = 'exam-options';
    opts.forEach(([k, v]) => {
      const o = document.createElement('div');
      o.className = 'exam-option';
      o.textContent = `${k}. ${v}`;
      ul.appendChild(o);
    });
    box.appendChild(ul);
  }
  if (it.answer || it.analysis) {
    const d = document.createElement('details');
    d.className = 'exam-answer';
    const sum = document.createElement('summary');
    sum.textContent = '查看答案';
    d.appendChild(sum);
    const add = (label, text) => {
      if (!text) return;
      const p = document.createElement('div');
      p.className = 'exam-answer-row';
      p.textContent = `${label}${text}`;
      d.appendChild(p);
    };
    add('参考答案：', it.answer);
    add('评分要点与解析：', it.analysis);
    box.appendChild(d);
  }
  return box;
}

function examList(items, title, withMeta) {
  const wrap = document.createElement('div');
  wrap.className = 'sources numbered exam-list';
  if (title) {
    const h = document.createElement('div');
    h.className = 'sources-kind';
    h.textContent = title;
    wrap.appendChild(h);
  }
  items.forEach((it, i) => wrap.appendChild(questionBlock(it, i, withMeta)));
  return wrap;
}

// 课程回答后附的试卷库基础题（判断题 / 名词解释，答案默认折叠）
function practiceList(items) {
  return examList(items, '巩固练习', false);
}

function numberedSources(sources) {
  const wrap = document.createElement('div');
  wrap.className = 'sources numbered';
  const title = document.createElement('div');
  title.className = 'sources-title';
  title.textContent = `来源（${sources.length}）`;
  wrap.appendChild(title);

  const kinds = [...new Set(sources.map((s) => s.kind || 'course'))];
  for (const kind of kinds) {
    if (kinds.length > 1) {
      const h = document.createElement('div');
      h.className = 'sources-kind';
      h.textContent = KIND_LABELS[kind] || kind;
      wrap.appendChild(h);
    }
    sources.filter((s) => (s.kind || 'course') === kind).forEach((s) => {
      const row = document.createElement('div');
      row.className = 'src-row';
      const num = document.createElement('span');
      num.className = 'src-n';
      num.textContent = `[${s.n}]`;
      const path = document.createElement('span');
      path.className = 'src-row-path';
      path.textContent = s.path || s.title || '';
      row.append(num, path);
      wrap.appendChild(row);
    });
  }
  return wrap;
}

/* ---- Markdown（安全转义后做轻量渲染；[n] 渲染为带来源提示的上标）---- */
function escapeHtml(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function renderMarkdown(md, cites) {
  const lines = escapeHtml(md).split('\n');
  let html = '';
  let listType = null;
  let inCode = false;
  const codeBuf = [];

  const closeList = () => {
    if (listType) { html += '</' + listType + '>'; listType = null; }
  };
  const cite = (t) => (cites ? t.replace(/\[(\d{1,3})\]/g, (m, n) => {
    const p = cites.get(Number(n));
    return `<sup class="cite"${p ? ` title="${escapeHtml(p)}"` : ''}>[${n}]</sup>`;
  }) : t);
  const inline = (t) => cite(t
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/\*([^*]+)\*/g, '<em>$1</em>'));

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

/* ---- 回答内容的渲染（实时与历史回填共用，避免两套逻辑走偏）---- */

/* 把 [n] -> 来源路径 填进 cites，正文渲染前必须先调用，否则引用没有悬停提示 */
function fillCites(sources, cites) {
  (sources || []).forEach((s, i) => {
    if (!s) return;
    if (s.n !== undefined) cites.set(Number(s.n), s.path || s.title || s.source || '');
    else cites.set(i + 1, s.section ? `${s.source} · ${s.section}` : (s.source || ''));   // 论文库来源没有 n：按顺序编号，与正文 [n] 对应
  });
  return cites;
}

/* 来源卡片 / 试卷题 / 巩固练习。试卷题的正文提示由调用方给，这里只管结构化部分 */
function appendAnswerParts(container, ev, cites) {
  const sources = ev.sources || [];
  const examQs = ev.exam && ev.exam.questions ? ev.exam.questions : [];
  if (examQs.length) container.appendChild(examList(examQs, '', true));
  if (sources.length) {
    if (sources[0].n !== undefined) {
      container.appendChild(numberedSources(sources));
    } else {
      const wrap = document.createElement('div');
      wrap.className = 'sources';
      const title = document.createElement('div');
      title.className = 'sources-title';
      title.textContent = `参考来源（${sources.length}）`;
      wrap.appendChild(title);
      sources.forEach((s, i) => wrap.appendChild(sourceCard(i, s, ev.contexts && ev.contexts[i])));
      container.appendChild(wrap);
    }
  }
  if (ev.practice && ev.practice.length) container.appendChild(practiceList(ev.practice));
  return examQs;
}

/* ---- 发送 ---- */
function autoGrow() {
  input.style.height = 'auto';
  input.style.height = Math.min(input.scrollHeight, 160) + 'px';
}

function routeTag(routes) {
  const tag = document.createElement('div');
  tag.className = 'route-tag';
  tag.textContent = routes.map((r) => ROUTE_LABELS[r] || r).join(' + ');
  return tag;
}

/* 追问胶囊：点一下填入输入框并自动发送（沿用 .chip 的胶囊风格，尺寸略小以区别于欢迎页） */
function followupCapsules(items) {
  const wrap = document.createElement('div');
  wrap.className = 'followups';
  const label = document.createElement('div');
  label.className = 'followups-label';
  label.textContent = '继续追问';
  wrap.appendChild(label);
  items.forEach((text) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'chip followup';
    b.textContent = text;
    b.addEventListener('click', () => {
      if (sendBtn.disabled) return;          // 上一轮还在生成：不打断
      input.value = text;
      autoGrow();
      send();
    });
    wrap.appendChild(b);
  });
  return wrap;
}

async function send() {
  const q = input.value.trim();
  if (!q) return;

  addUser(q);
  input.value = '';
  autoGrow();
  sendBtn.disabled = true;

  const t0 = performance.now();
  let firstAt = null;
  const pending = document.createElement('div');
  pending.className = 'bubble assistant pending';
  const pendingText = document.createElement('span');
  pendingText.textContent = '正在判断问题类型';
  const elapsed = document.createElement('span');
  elapsed.className = 'pending-elapsed';
  pending.append(pendingText, elapsed);
  const pendingMsg = addMessage('assistant', pending);
  const timer = setInterval(() => {
    elapsed.textContent = ' ' + ((performance.now() - t0) / 1000).toFixed(1) + 's';
  }, 100);

  const body = document.createElement('div');
  body.className = 'assistant-body';
  const bubble = document.createElement('div');
  bubble.className = 'bubble assistant markdown';
  const cites = new Map();        // 序号 -> 来源路径（materials 事件陆续填入）
  let text = '';
  let frame = null;
  let started = false;
  let finished = false;

  const paint = () => {
    frame = null;
    bubble.innerHTML = renderMarkdown(text, cites);
    scrollBottom();
  };
  const flush = () => { if (!frame) frame = requestAnimationFrame(paint); };

  const start = () => {
    if (started) return;
    started = true;
    clearInterval(timer);
    pendingMsg.remove();
    body.appendChild(bubble);
    addMessage('assistant', body);
  };

  const meta = (note) => {
    const m = document.createElement('div');
    m.className = 'answer-meta';
    const total = ((performance.now() - t0) / 1000).toFixed(1);
    const first = firstAt === null ? '' : `首字 ${((firstAt - t0) / 1000).toFixed(1)}s · `;
    m.textContent = `${first}总耗时 ${total}s${note ? ' · ' + note : ''}`;
    return m;
  };

  const onEvent = (ev) => {
    if (ev.type === 'route') {
      const routes = ev.routes || [ev.route];
      body.appendChild(routeTag(routes));
      pendingText.textContent = routes.includes('off_topic') ? '正在生成回复' : '正在检索资料';
    } else if (ev.type === 'stage') {
      if (ev.stage === 'select') {
        const names = (ev.concepts || []).slice(0, 4).join('、');
        pendingText.textContent = names ? `已定位概念：${names}，正在召回教材` : '正在召回教材';
      } else if (ev.stage === 'filter') {
        pendingText.textContent = ev.skipped ? `召回 ${ev.recalled} 块，正在作答`
          : `召回 ${ev.recalled} 块、保留 ${ev.kept} 块，正在作答`;
      }
    } else if (ev.type === 'materials') {
      cites.set(Number(ev.n), ev.path || ev.title || '');
      if (started) flush();
    } else if (ev.type === 'delta') {
      if (firstAt === null) firstAt = performance.now();
      start();
      text += ev.text || '';
      flush();
    } else if (ev.type === 'done') {
      finished = true;
      start();
      if (frame) { cancelAnimationFrame(frame); frame = null; }
      const sources = ev.sources || [];
      fillCites(sources, cites);
      const examQs = ev.exam && ev.exam.questions ? ev.exam.questions : [];
      // 试卷库结果：题目、选项和默认折叠的答案由下面的结构化列表渲染，气泡里只留一句提示，避免题干重复显示
      // 题库命中且要讲解（explain）：正文是课程讲解，题目与参考答案仍由下面的折叠块给出
      text = examQs.length && !ev.explain ? (ev.exam.solve ? '题库里有这道题，答案如下：' : `为你找到 ${examQs.length} 道题：`) : (ev.answer || '（没有返回内容）');
      if (ev.status && ev.status !== 'ok') bubble.classList.add('note');
      paint();
      appendAnswerParts(body, ev, cites);
      body.appendChild(meta());
      scrollBottom();
    } else if (ev.type === 'followups') {
      // 追问在 done 之后单独一帧到达（后端不阻塞主答案）。点击填入输入框并自动发送。
      const items = (ev.items || []).filter((s) => typeof s === 'string' && s.trim());
      if (items.length) body.appendChild(followupCapsules(items));
      scrollBottom();
    } else if (ev.type === 'error') {
      finished = true;
      start();
      text += (text ? '\n\n' : '') + '（回答出错，请重试）';
      bubble.classList.add('note');
      paint();
      body.appendChild(meta());
    }
  };

  try {
    const headers = authHeaders({ 'Content-Type': 'application/json' });
    headers['X-User-Id'] = userId();                                  // 浏览器标识（同一浏览器长期不变）
    headers['X-User-Uuid'] = userId();                                // 会话归属也用它：共享 key 时按人隔离历史
    const payload = { question: q, top_k: parseInt(topkSel.value, 10) || 4, include_trace: false };
    if (conversationId) payload.conversation_id = conversationId;   // 不传就会每轮新建一个会话
    const r = await fetch('/auto/ask/stream', {
      method: 'POST',
      headers,
      body: JSON.stringify(payload),
    });
    if (!r.ok) {
      clearInterval(timer);
      pendingMsg.remove();
      if (r.status === 401) showError('需要有效的访问密钥：请在下方“访问密钥”里填写后重试。');
      else if (r.status === 404) {
        // 会话被删或换了 key：丢掉旧 id，重新开一个，别让用户卡住
        forgetConvId();
        showError('这个会话已不存在，已为你开始新对话。请重新发送。');
      } else if (r.status === 503) showError('服务繁忙或未就绪，请稍后重试。');
      else showError('请求失败（HTTP ' + r.status + '），请重试。');
      return;
    }
    // 响应头里就有会话 id，早于任何事件到达，先存下来
    const headCid = r.headers.get('X-Conversation-Id');
    if (headCid && headCid !== conversationId) {
      conversationId = headCid;
      saveConvId(headCid);
    }

    const reader = r.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf('\n\n')) >= 0) {
        const raw = buf.slice(0, idx).trim();
        buf = buf.slice(idx + 2);
        if (!raw.startsWith('data: ')) continue;
        let ev;
        try { ev = JSON.parse(raw.slice(6)); } catch { continue; /* 跳过坏帧 */ }
        onEvent(ev);
      }
    }
    if (!finished) {
      if (!started) { clearInterval(timer); pendingMsg.remove(); }
      showError('连接中断，回答不完整，请重试。');
    }
    refreshConversations();
  } catch {
    clearInterval(timer);
    if (!started) pendingMsg.remove();
    showError('网络错误，请重试。');
  } finally {
    sendBtn.disabled = false;
  }
}

/* ---- 历史会话：列表 + 回填 ---- */

async function convHeaders() {
  const h = authHeaders();
  h['X-User-Uuid'] = userId();          // 与问答请求同一个标识，才能看到「自己」的会话
  return h;
}

async function refreshConversations() {
  const list = document.getElementById('conv-list');
  const empty = document.getElementById('conv-empty');
  if (!list) return;
  // 页面加载时 key 还没填：不发请求，等输入 key 的 change 再拉
  const key = currentKey();
  if (!key) {
    if (empty) empty.textContent = '填写访问密钥后显示历史';
    return;
  }
  try {
    const r = await fetch('/v1/conversations?limit=30', { headers: convHeaders() });
    if (!r.ok) {
      // 鉴权失败：侧栏保持空态，不打扰主流程
      if (empty) empty.textContent = r.status === 401 ? '访问密钥无效，无法查看历史' : '历史加载失败';
      return;
    }
    const data = (await r.json()).data || [];
    [...list.querySelectorAll('.conv-item')].forEach((n) => n.remove());
    if (empty) empty.style.display = data.length ? 'none' : '';
    data.forEach((c) => list.appendChild(convItem(c)));
  } catch {
    if (empty) empty.textContent = '历史加载失败';
  }
}

function convItem(c) {
  const b = document.createElement('button');
  b.type = 'button';
  b.className = 'conv-item' + (c.id === conversationId ? ' active' : '');
  b.textContent = c.title || '（未命名会话）';
  b.title = `${c.title || ''}\n${c.turn_count} 轮 · ${c.updated_at_iso || ''}`;
  b.addEventListener('click', () => openConversation(c.id));
  return b;
}

async function openConversation(cid) {
  if (sendBtn.disabled) return;              // 正在生成，不打断
  try {
    const r = await fetch(`/v1/conversations/${cid}?limit=500`, { headers: convHeaders() });
    if (!r.ok) {
      if (r.status === 404) {
        // 会话已被删除（或换了 key）：丢掉本地 id，直接开始新对话。
        // 否则界面停在旧会话上，下一句发出去时会带着这个已失效的 id。
        forgetConvId();
        clearChat();
        refreshConversations();
      }
      return;
    }
    const conv = await r.json();
    conversationId = cid;
    saveConvId(cid);
    clearChat();
    const msgs = conv.messages || [];
    if (msgs.length) hideWelcome();
    msgs.forEach(renderHistoryMessage);
    refreshConversations();      // 重画列表以标出当前会话（conversationId 已更新）
    scrollBottom();
    closeSidebar();
  } catch { /* 网络错误：保持当前界面 */ }
}

/* 回填一条历史消息。只用库里存的字段重建气泡，不带耗时行与追问胶囊 */
function renderHistoryMessage(m) {
  if (!m || !(m.content || '').trim()) return;
  if (m.role === 'user') {
    addUser(m.content);
    return;
  }
  const body = document.createElement('div');
  body.className = 'assistant-body';
  const bubble = document.createElement('div');
  bubble.className = 'bubble assistant markdown';
  if (m.status && m.status !== 'ok') bubble.classList.add('note');
  const cites = fillCites(m.sources, new Map());
  bubble.innerHTML = renderMarkdown(m.content, cites);
  body.appendChild(bubble);
  // 历史里没有 exam / practice 的结构化数据，只有正文与来源，够还原阅读
  appendAnswerParts(body, { sources: m.sources || [] }, cites);
  addMessage('assistant', body).classList.add('history');
}

/* ---- 侧栏开关（窄屏抽屉）---- */
function openSidebar() {
  const s = document.getElementById('sidebar');
  const mask = document.getElementById('sidebar-mask');
  if (s) s.classList.add('open');
  if (mask) mask.classList.add('open');
}
function closeSidebar() {
  const s = document.getElementById('sidebar');
  const mask = document.getElementById('sidebar-mask');
  if (s) s.classList.remove('open');
  if (mask) mask.classList.remove('open');
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

const newChatBtn = document.getElementById('new-chat');
if (newChatBtn) {
  newChatBtn.addEventListener('click', () => {
    if (sendBtn.disabled) return;
    forgetConvId();          // 只丢“当前会话”，历史列表照旧存在
    clearChat();
    refreshConversations();
    closeSidebar();
    input.focus();
  });
}
const toggleBtn = document.getElementById('toggle-sidebar');
if (toggleBtn) toggleBtn.addEventListener('click', () => {
  const s = document.getElementById('sidebar');
  if (s && s.classList.contains('open')) closeSidebar(); else openSidebar();
});
const mask = document.getElementById('sidebar-mask');
if (mask) mask.addEventListener('click', closeSidebar);

conversationId = loadConvId();
refreshConversations();
// 刷新页面后把当前会话的历史直接铺开：否则输入框发出去的是「接着上一轮」的新问题，
// 而屏幕上只有这一轮，用户看不出在跟谁说话。
if (conversationId) openConversation(conversationId);

checkHealth();
setInterval(checkHealth, 30000);
