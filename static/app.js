/* RAG 知识库问答前端逻辑：上传、文档管理、SSE 流式问答 */

const els = {
  fileInput: document.getElementById('file-input'),
  dropzone: document.getElementById('dropzone'),
  uploadStatus: document.getElementById('upload-status'),
  docList: document.getElementById('doc-list'),
  docCount: document.getElementById('doc-count'),
  docEmpty: document.getElementById('doc-empty'),
  messages: document.getElementById('messages'),
  welcome: document.getElementById('welcome'),
  composer: document.getElementById('composer'),
  question: document.getElementById('question'),
  sendBtn: document.getElementById('send-btn'),
  clearChat: document.getElementById('clear-chat'),
  statusPill: document.getElementById('status-pill'),
  modelLine: document.getElementById('model-line'),
};

let busy = false;
let welcomeHtml = '';

const ICONS = {
  bot: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"
        stroke-linecap="round" stroke-linejoin="round">
        <rect x="4" y="7" width="16" height="12" rx="3" />
        <path d="M12 3v4M9 13h.01M15 13h.01M9.5 16.5h5" />
      </svg>`,
  trash: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"
        stroke-linecap="round" stroke-linejoin="round">
        <path d="M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6" />
      </svg>`,
  quote: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"
        stroke-linecap="round" stroke-linejoin="round">
        <path d="M4 6h16M4 12h10M4 18h13" />
      </svg>`,
};

/* ---------------- 通用工具 ---------------- */

function escapeHtml(text) {
  return String(text)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function inlineFormat(text) {
  return text
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/\[(\d{1,2})\]/g, '<span class="cite-ref">[$1]</span>');
}

function renderMarkdown(source) {
  const lines = escapeHtml(source || '').split('\n');
  const html = [];
  let paragraph = [];
  let i = 0;

  const flush = () => {
    if (paragraph.length) {
      const text = inlineFormat(paragraph.join('\n')).replace(/\n/g, '<br>');
      html.push(`<p>${text}</p>`);
      paragraph = [];
    }
  };

  while (i < lines.length) {
    const line = lines[i];
    const trimmed = line.trim();

    if (trimmed.startsWith('```')) {
      flush();
      const code = [];
      i += 1;
      while (i < lines.length && !lines[i].trim().startsWith('```')) {
        code.push(lines[i]);
        i += 1;
      }
      i += 1;
      html.push(`<pre><code>${code.join('\n')}</code></pre>`);
      continue;
    }

    if (!trimmed) {
      flush();
      i += 1;
      continue;
    }

    if (/^#{1,6}\s+/.test(trimmed)) {
      flush();
      const level = (trimmed.match(/^#+/) || ['#'])[0].length <= 2 ? 3 : 4;
      html.push(`<h${level}>${inlineFormat(trimmed.replace(/^#{1,6}\s+/, ''))}</h${level}>`);
      i += 1;
      continue;
    }

    if (/^[-*+]\s+/.test(trimmed)) {
      flush();
      const items = [];
      while (i < lines.length && /^\s*[-*+]\s+/.test(lines[i])) {
        items.push(`<li>${inlineFormat(lines[i].replace(/^\s*[-*+]\s+/, ''))}</li>`);
        i += 1;
      }
      html.push(`<ul>${items.join('')}</ul>`);
      continue;
    }

    if (/^\d+[.)]\s+/.test(trimmed)) {
      flush();
      const items = [];
      while (i < lines.length && /^\s*\d+[.)]\s+/.test(lines[i])) {
        items.push(`<li>${inlineFormat(lines[i].replace(/^\s*\d+[.)]\s+/, ''))}</li>`);
        i += 1;
      }
      html.push(`<ol>${items.join('')}</ol>`);
      continue;
    }

    if (trimmed.startsWith('&gt;')) {
      flush();
      const quotes = [];
      while (i < lines.length && lines[i].trim().startsWith('&gt;')) {
        quotes.push(lines[i].replace(/^\s*&gt;\s?/, ''));
        i += 1;
      }
      html.push(`<blockquote>${inlineFormat(quotes.join('<br>'))}</blockquote>`);
      continue;
    }

    paragraph.push(line);
    i += 1;
  }

  flush();
  return html.join('');
}

function formatSize(bytes) {
  if (!Number.isFinite(bytes)) return '';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function scrollToBottom() {
  els.messages.scrollTop = els.messages.scrollHeight;
}

async function parseJson(response) {
  try {
    return await response.json();
  } catch (_) {
    return {};
  }
}

/* ---------------- 服务状态 ---------------- */

async function loadHealth() {
  try {
    const response = await fetch('/api/health');
    if (!response.ok) throw new Error('bad status');
    const data = await response.json();
    els.modelLine.textContent = `${data.embedding_model} · ${data.llm_model}`;
    if (data.deepseek_configured) {
      setStatus('ok', '服务正常');
    } else {
      setStatus('warn', '未配置 API Key');
    }
  } catch (_) {
    setStatus('error', '服务不可用');
    els.modelLine.textContent = '无法连接后端服务';
  }
}

function setStatus(kind, text) {
  els.statusPill.className = `status ${kind}`;
  const label = els.statusPill.querySelector('span');
  if (label) label.textContent = text;
}

/* ---------------- 文档管理 ---------------- */

async function loadDocuments() {
  try {
    const response = await fetch('/api/documents');
    if (!response.ok) throw new Error('加载文档列表失败');
    const data = await response.json();
    renderDocuments(data.items || []);
  } catch (error) {
    els.docList.innerHTML = '';
    els.docCount.textContent = '0';
    els.docEmpty.hidden = false;
    els.docEmpty.textContent = error.message;
  }
}

function renderDocuments(items) {
  els.docCount.textContent = String(items.length);
  els.docEmpty.hidden = items.length > 0;
  els.docEmpty.textContent = '暂无文档，先上传一份资料';
  els.docList.innerHTML = '';

  for (const doc of items) {
    const item = document.createElement('li');
    item.className = 'doc-item';

    const type = document.createElement('span');
    type.className = 'doc-type';
    type.textContent = (doc.filename.split('.').pop() || 'doc').slice(0, 4).toUpperCase();

    const info = document.createElement('div');
    info.className = 'doc-info';

    const name = document.createElement('div');
    name.className = 'doc-name';
    name.textContent = doc.filename;
    name.title = doc.filename;

    const meta = document.createElement('div');
    meta.className = 'doc-meta';
    meta.textContent = `${doc.chunk_count} 个片段 · ${formatSize(doc.size)}`;

    info.append(name, meta);

    const del = document.createElement('button');
    del.className = 'doc-delete';
    del.type = 'button';
    del.title = '删除文档';
    del.innerHTML = ICONS.trash;
    del.addEventListener('click', () => deleteDocument(doc));

    item.append(type, info, del);
    els.docList.append(item);
  }
}

async function deleteDocument(doc) {
  if (!window.confirm(`确定删除「${doc.filename}」及其全部向量片段吗？`)) return;
  try {
    const response = await fetch(`/api/documents/${encodeURIComponent(doc.doc_id)}`, {
      method: 'DELETE',
    });
    if (!response.ok) {
      const data = await parseJson(response);
      throw new Error(data.detail || '删除失败');
    }
    await loadDocuments();
  } catch (error) {
    window.alert(error.message);
  }
}

async function uploadFiles(fileList) {
  const files = Array.from(fileList || []);
  if (!files.length) return;

  els.uploadStatus.hidden = false;
  let uploaded = 0;

  for (let index = 0; index < files.length; index += 1) {
    const file = files[index];
    els.uploadStatus.className = 'upload-status';
    els.uploadStatus.textContent =
      `正在入库 ${file.name}（${index + 1}/${files.length}）…` +
      (index === 0 ? '首次使用需下载 BGE 模型，请稍候' : '');

    const form = new FormData();
    form.append('file', file);

    try {
      const response = await fetch('/api/documents', { method: 'POST', body: form });
      const data = await parseJson(response);
      if (!response.ok) throw new Error(data.detail || `上传失败（${response.status}）`);
      uploaded += 1;
    } catch (error) {
      els.uploadStatus.className = 'upload-status error';
      els.uploadStatus.textContent = `${file.name} 入库失败：${error.message}`;
      await loadDocuments();
      els.fileInput.value = '';
      return;
    }
  }

  els.uploadStatus.className = 'upload-status success';
  els.uploadStatus.textContent = `已完成入库：${uploaded} 个文件`;
  els.fileInput.value = '';
  await loadDocuments();
  await loadHealth();
}

/* ---------------- 对话 ---------------- */

function hideWelcome() {
  if (els.welcome && els.welcome.parentNode) els.welcome.remove();
}

function appendUserMessage(text) {
  const wrap = document.createElement('div');
  wrap.className = 'message user';

  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = '我';

  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;

  wrap.append(avatar, bubble);
  els.messages.append(wrap);
  scrollToBottom();
}

function createAssistantMessage() {
  const wrap = document.createElement('div');
  wrap.className = 'message assistant';

  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.innerHTML = ICONS.bot;

  const bubble = document.createElement('div');
  bubble.className = 'bubble';

  const role = document.createElement('div');
  role.className = 'role-name';
  role.textContent = '知识库助手';

  const answer = document.createElement('div');
  answer.className = 'answer';

  const typing = document.createElement('div');
  typing.className = 'typing';
  typing.innerHTML = '<span></span><span></span><span></span>';
  answer.append(typing);

  const sources = document.createElement('div');
  sources.className = 'sources';
  sources.hidden = true;

  bubble.append(role, answer, sources);
  wrap.append(avatar, bubble);
  els.messages.append(wrap);
  scrollToBottom();

  return { answer, sources, typing };
}

function renderSources(container, sources) {
  if (!Array.isArray(sources) || !sources.length) {
    container.hidden = true;
    return;
  }
  container.hidden = false;
  container.innerHTML = '';

  const title = document.createElement('div');
  title.className = 'sources-title';
  title.innerHTML = `${ICONS.quote}<span>引用来源（${sources.length}）</span>`;

  const list = document.createElement('div');
  list.className = 'source-list';

  for (const source of sources) {
    const item = document.createElement('div');
    item.className = 'source-item';

    const head = document.createElement('button');
    head.type = 'button';
    head.className = 'source-head';

    const index = document.createElement('span');
    index.className = 'source-index';
    index.textContent = String(source.index);

    const name = document.createElement('span');
    name.className = 'source-name';
    name.textContent = source.filename;
    name.title = source.filename;

    const score = document.createElement('span');
    score.className = 'source-score';
    score.textContent = `相似度 ${(source.score * 100).toFixed(0)}%`;

    head.append(index, name, score);

    const snippet = document.createElement('div');
    snippet.className = 'source-snippet';
    snippet.hidden = true;
    snippet.textContent = source.content;

    head.addEventListener('click', () => {
      snippet.hidden = !snippet.hidden;
    });

    item.append(head, snippet);
    list.append(item);
  }

  container.append(title, list);
}

function showAssistantError(answerEl, typing, message, hasText) {
  if (typing && typing.parentNode) typing.remove();
  const note = document.createElement('div');
  note.className = 'error-note';
  note.textContent = message;
  if (!hasText) answerEl.innerHTML = '';
  answerEl.append(note);
}

async function sendQuestion(rawQuestion) {
  const question = String(rawQuestion || '').trim();
  if (!question || busy) return;

  hideWelcome();
  appendUserMessage(question);
  els.question.value = '';
  autoGrow();

  const { answer, sources, typing } = createAssistantMessage();
  let answerText = '';
  let receivedSources = false;

  setBusy(true);

  try {
    const response = await fetch('/api/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    });

    if (!response.ok || !response.body) {
      const data = await parseJson(response);
      throw new Error(data.detail || `请求失败（${response.status}）`);
    }

    await readSSE(response.body, (event, payload) => {
      if (event === 'sources') {
        receivedSources = true;
        renderSources(sources, payload.sources);
      } else if (event === 'delta') {
        if (typing.parentNode) typing.remove();
        answerText += payload.text || '';
        answer.innerHTML = renderMarkdown(answerText);
        scrollToBottom();
      } else if (event === 'error') {
        throw new Error(payload.detail || '生成回答失败');
      }
    });

    if (!answerText) {
      if (typing.parentNode) typing.remove();
      answer.innerHTML = '<p>（模型未返回内容，请稍后重试）</p>';
    }
  } catch (error) {
    if (!receivedSources) sources.hidden = true;
    showAssistantError(answer, typing, error.message, Boolean(answerText));
    scrollToBottom();
  } finally {
    setBusy(false);
    await loadHealth();
  }
}

async function readSSE(body, onEvent) {
  const reader = body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';

  const handleChunk = (raw) => {
    let event = 'message';
    const dataLines = [];
    for (const line of raw.split('\n')) {
      if (line.startsWith('event:')) event = line.slice(6).trim();
      else if (line.startsWith('data:')) dataLines.push(line.slice(5).trimStart());
    }
    if (!dataLines.length) return;
    let payload;
    try {
      payload = JSON.parse(dataLines.join('\n'));
    } catch (_) {
      return;
    }
    onEvent(event, payload);
  };

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer = (buffer + decoder.decode(value, { stream: true })).replace(/\r\n/g, '\n');
    let boundary = buffer.indexOf('\n\n');
    while (boundary !== -1) {
      handleChunk(buffer.slice(0, boundary));
      buffer = buffer.slice(boundary + 2);
      boundary = buffer.indexOf('\n\n');
    }
  }

  buffer = (buffer + decoder.decode()).replace(/\r\n/g, '\n');
  if (buffer.trim()) handleChunk(buffer);
}

function setBusy(value) {
  busy = value;
  els.sendBtn.disabled = value;
  els.question.disabled = value;
  if (!value) els.question.focus();
}

function autoGrow() {
  const el = els.question;
  el.style.height = 'auto';
  el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
}

/* ---------------- 事件绑定 ---------------- */

els.fileInput.addEventListener('change', (event) => uploadFiles(event.target.files));

['dragenter', 'dragover'].forEach((type) => {
  els.dropzone.addEventListener(type, (event) => {
    event.preventDefault();
    els.dropzone.classList.add('dragging');
  });
});

['dragleave', 'drop'].forEach((type) => {
  els.dropzone.addEventListener(type, (event) => {
    event.preventDefault();
    els.dropzone.classList.remove('dragging');
  });
});

els.dropzone.addEventListener('drop', (event) => {
  if (event.dataTransfer && event.dataTransfer.files) uploadFiles(event.dataTransfer.files);
});

els.composer.addEventListener('submit', (event) => {
  event.preventDefault();
  sendQuestion(els.question.value);
});

els.question.addEventListener('input', autoGrow);

els.question.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    sendQuestion(els.question.value);
  }
});

els.messages.addEventListener('click', (event) => {
  const trigger = event.target.closest('[data-question]');
  if (trigger) sendQuestion(trigger.dataset.question);
});

els.clearChat.addEventListener('click', () => {
  if (busy) return;
  els.messages.innerHTML = welcomeHtml;
  els.welcome = document.getElementById('welcome');
  els.question.focus();
});

/* ---------------- 初始化 ---------------- */

(async function init() {
  welcomeHtml = els.messages.innerHTML;
  await loadHealth();
  await loadDocuments();
  autoGrow();
})();
