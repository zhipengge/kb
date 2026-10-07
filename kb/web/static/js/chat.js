/* AI 会话页。
 *
 * 三个技术选择及其理由：
 *
 * 1. **用 fetch + ReadableStream 手工解析 SSE，不用 EventSource。**
 *    EventSource 只能发 GET、不能带请求头，也就发不出 CSRF token、
 *    带不了请求体（问题、附件列表都得放 body）。
 *
 * 2. **Markdown 用 Vditor.preview() 渲染**，而不是自己引一个 marked.js。
 *    会话里的公式、代码块要和笔记编辑器渲染得一样——两套渲染器迟早分叉。
 *
 * 3. **流式过程中节流渲染**（约 120ms 一次）。每个 token 都重排一次
 *    Markdown 会让长回答明显卡顿，而人眼根本看不出区别。
 */

import { fixMermaidSizing, wrapWideTables } from '/static/js/vditor-fix.js';

const RENDER_THROTTLE_MS = 120;

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

export function createChat(opts) {
  const messagesEl = document.querySelector('[data-chat-messages]');
  const inputEl = document.querySelector('[data-chat-input]');
  const sendBtn = document.querySelector('[data-chat-send]');
  const statusEl = document.querySelector('[data-chat-status]');
  const titleEl = document.querySelector('[data-chat-title]');
  const attachEl = document.querySelector('[data-chat-attachments]');
  const mentionEl = document.querySelector('[data-chat-mention]');
  const fileEl = document.querySelector('[data-chat-file]');

  if (!messagesEl || !inputEl) return null;

  let conversationId = opts.currentId || '';
  let streaming = false;
  let attachments = [];  // [{kind, name, media_type, path}]
  let mentionStart = -1;

  const url = (key, id = conversationId) => (opts.urls[key] || '').replace('__ID__', id);
  const setStatus = (text) => { if (statusEl) statusEl.textContent = text || ''; };

  async function jsonRequest(target, { method = 'POST', body } = {}) {
    const response = await fetch(target, {
      method,
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
        'X-CSRFToken': opts.csrfToken,
      },
      body: body ? JSON.stringify(body) : undefined,
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || data.ok === false) {
      throw new Error(data.error || `HTTP ${response.status}`);
    }
    return data;
  }

  /* ---------------- 消息渲染 ---------------- */

  function scrollToBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  /** 引用卡片。三态直接映射成颜色：绿=可核对，灰=没给引文，红=引文对不上。 */
  function citationCards(citations) {
    if (!citations || !citations.length) return null;
    const wrap = document.createElement('div');
    wrap.className = 'cite-list';
    citations.forEach((cite) => {
      const state = cite.check || (cite.verified ? 'verified' : 'unverified');
      const card = document.createElement('a');
      card.className = `cite-card cite-${state}`;
      card.href = cite.paper_id ? `/papers/${cite.paper_id}` : '#';
      card.title = {
        verified: '引文能在被引分块中逐字找到',
        unverified: '没有给出可核对的引文',
        mismatched: '给出的引文在被引分块里找不到——最需要人工核对',
      }[state] || '';
      card.innerHTML = `
        <div class="cite-head">
          <span class="cite-marker">${cite.marker}</span>
          <span class="cite-title">${escapeHtml(cite.title || '（无标题）')}</span>
          <span class="cite-state"></span>
        </div>
        ${cite.locator ? `<div class="cite-locator mono small">${escapeHtml(cite.locator)}</div>` : ''}
        ${cite.quote ? `<div class="cite-quote small">「${escapeHtml(cite.quote.slice(0, 160))}」</div>` : ''}
      `;
      wrap.appendChild(card);
    });
    return wrap;
  }

  /**
   * 联网来源卡片。
   *
   * 与知识库引用**分开渲染**不是形式问题：知识库引用带 `check` 三态
   * （引文能否逐字核对），联网结果没有这种校验，可靠度天然更低。
   * 混在一列里会让人以为它们同等可信。
   */
  function webCards(citations) {
    if (!citations || !citations.length) return null;
    const wrap = document.createElement('div');
    wrap.className = 'web-list';
    const head = document.createElement('div');
    head.className = 'web-list-head';
    head.textContent = '联网检索来源（未经知识库核对）';
    wrap.appendChild(head);

    const grid = document.createElement('div');
    grid.className = 'cite-list';
    citations.forEach((item) => {
      const card = document.createElement('a');
      card.className = 'cite-card cite-web';
      card.href = item.url || '#';
      card.target = '_blank';
      card.rel = 'noopener noreferrer';
      const kindLabel = { paper: '论文', code: '代码', web: '网页' }[item.kind] || '';
      const extra = [];
      if (item.published) extra.push(item.published);
      if (item.source) extra.push(item.source);
      card.innerHTML = `
        <div class="cite-head">
          <span class="cite-marker">${escapeHtml(item.marker || '')}</span>
          <span class="cite-title">${escapeHtml(item.title || '')}</span>
        </div>
        ${extra.length ? `<div class="cite-locator small">${escapeHtml(kindLabel)} · ${escapeHtml(extra.join(' · '))}</div>` : ''}
        ${item.snippet ? `<div class="cite-quote small">${escapeHtml(item.snippet.slice(0, 140))}</div>` : ''}
      `;
      grid.appendChild(card);
    });
    wrap.appendChild(grid);
    return wrap;
  }

  /** 生成一条消息底部的操作条（复制 / 元信息）。 */
  function buildActions(node) {
    const bar = document.createElement('div');
    bar.className = 'msg-actions';

    const copy = document.createElement('button');
    copy.type = 'button';
    copy.className = 'msg-action';
    copy.textContent = '复制';
    copy.addEventListener('click', async () => {
      // 复制的是**纯文本**而不是渲染后的 HTML：粘到别处时不该带一堆标签
      const text = node.querySelector('.msg-text')?.innerText || '';
      try {
        await navigator.clipboard.writeText(text);
        copy.textContent = '已复制';
      } catch {
        copy.textContent = '复制失败';
      }
      setTimeout(() => { copy.textContent = '复制'; }, 1400);
    });
    bar.appendChild(copy);

    const meta = document.createElement('span');
    meta.className = 'msg-meta';
    bar.appendChild(meta);

    node.querySelector('.msg-body').appendChild(bar);
    return { bar, meta };
  }

  function addMessage(role, text = '') {
    const node = document.createElement('div');
    node.className = `msg msg-${role}`;
    node.innerHTML = `
      <div class="msg-role">${role === 'user' ? '我' : 'AI'}</div>
      <div class="msg-body"><div class="msg-text"></div></div>
    `;
    const textEl = node.querySelector('.msg-text');
    if (role === 'user') textEl.textContent = text;

    const { bar, meta } = buildActions(node);
    if (role === 'user') {
      // 用户消息才可编辑。AI 的回答不能改——它是对提问的响应，
      // 想要不同答案应该改提问，而不是篡改回答（那会让引用失去依据）。
      const edit = document.createElement('button');
      edit.type = 'button';
      edit.className = 'msg-action msg-edit';
      edit.textContent = '编辑';
      edit.title = '编辑这条提问并重新提问';
      edit.addEventListener('click', () => startEdit(node));
      bar.insertBefore(edit, meta);
    }
    meta.textContent = new Date().toLocaleTimeString('zh-CN', { hour12: false });

    messagesEl.appendChild(node);
    messagesEl.querySelector('.chat-empty')?.remove();
    scrollToBottom();
    return { node, textEl, bodyEl: node.querySelector('.msg-body'), bar, meta };
  }

  /** 空回答气泡里的「正在输入」指示器。 */
  function showTyping(bodyEl) {
    const box = document.createElement('div');
    box.className = 'typing';
    box.innerHTML = '<i></i><i></i><i></i>';
    bodyEl.insertBefore(box, bodyEl.firstChild);
    return () => box.remove();
  }

  /** 把一条用户消息就地变成编辑框。 */
  function startEdit(node) {
    if (streaming) {
      setStatus('正在生成回答，等这一轮结束再编辑');
      return;
    }
    const messageId = node.dataset.messageId;
    if (!messageId) return;

    const textEl = node.querySelector('.msg-text');
    const original = textEl.textContent;

    // 这条之后还有多少条消息会被删掉——必须先让用户知道
    const following = [...messagesEl.querySelectorAll('.msg')];
    const index = following.indexOf(node);
    const affected = following.length - index - 1;

    const editor = document.createElement('div');
    editor.className = 'msg-edit-box';
    editor.innerHTML = `
      <textarea rows="3"></textarea>
      <div class="row" style="justify-content:space-between; margin-top:6px">
        <span class="small faint"></span>
        <span class="row" style="gap:6px">
          <button class="btn btn-ghost" type="button" data-cancel>取消</button>
          <button class="btn btn-primary" type="button" data-submit>重新提问</button>
        </span>
      </div>
    `;
    const textarea = editor.querySelector('textarea');
    textarea.value = original;
    editor.querySelector('.small.faint').textContent = affected
      ? `重新提问会删除这条之后的 ${affected} 条消息`
      : '重新提问会替换这条提问';

    textEl.hidden = true;
    node.querySelector('.msg-edit').hidden = true;
    node.querySelector('.msg-body').appendChild(editor);
    textarea.focus();
    textarea.setSelectionRange(textarea.value.length, textarea.value.length);

    editor.querySelector('[data-cancel]').addEventListener('click', () => {
      editor.remove();
      textEl.hidden = false;
      node.querySelector('.msg-edit').hidden = false;
    });

    editor.querySelector('[data-submit]').addEventListener('click', () => {
      const question = textarea.value.trim();
      if (!question) return;
      if (affected && !window.confirm(
        `重新提问会删除这条之后的 ${affected} 条消息，且不可恢复。继续？`)) {
        return;
      }
      editor.remove();
      restream(node, messageId, question);
    });
  }

  /** 编辑后重发：后端会截断后续消息，这里同步把界面上多余的节点删掉。 */
  async function restream(node, messageId, question) {
    streaming = true;
    if (sendBtn) sendBtn.disabled = true;

    const all = [...messagesEl.querySelectorAll('.msg')];
    const index = all.indexOf(node);
    // 界面上先删，随后服务端返回的 removed 数量用于核对
    all.slice(index + 1).forEach((el) => el.remove());

    node.dataset.messageId = messageId;
    const textEl = node.querySelector('.msg-text');
    textEl.hidden = false;
    textEl.textContent = question;
    const editBtn = node.querySelector('.msg-edit');
    if (editBtn) editBtn.hidden = false;

    const startedAt = Date.now();
    const aiNode = addMessage('assistant', '');
    const stopTyping = showTyping(aiNode.bodyEl);
    let typingStopped = false;
    const firstToken = () => {
      if (typingStopped) return;
      typingStopped = true;
      stopTyping();
    };
    const isDark = document.documentElement.dataset.theme === 'dark';
    let text = '';
    let renderTimer = null;
    let dirty = false;

    const scheduleRender = () => {
      dirty = true;
      if (renderTimer) return;
      renderTimer = setTimeout(async () => {
        renderTimer = null;
        if (!dirty) return;
        dirty = false;
        await renderMarkdown(aiNode.textEl, text, isDark);
        scrollToBottom();
      }, RENDER_THROTTLE_MS);
    };

    try {
      const target = opts.urls.restream
        .replace('__ID__', conversationId)
        .replace('_MSGID_', messageId);
      const response = await fetch(
        target,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Accept: 'text/event-stream',
            'X-CSRFToken': opts.csrfToken,
          },
          body: JSON.stringify({ question }),
        },
      );
      if (!response.ok || !response.body) throw new Error(`HTTP ${response.status}`);
      await consumeStream(response, makeHandlers({
        aiNode,
        isDark,
        onText: (chunk) => { firstToken(); text += chunk; scheduleRender(); },
        onThinking: (chunk) => {
          const box = ensureThinking(aiNode.bodyEl);
          box.textContent = (box.textContent || '') + chunk;
          if (!typingStopped) setStatus('正在思考…');
        },
        onStart: (event) => {
          // 服务端告诉我们截断了多少条；与自己删的数量对不上说明界面不同步
          if (event.removed > 0) setStatus(`已删除后续 ${event.removed} 条消息`);
        },
      }));
      if (renderTimer) clearTimeout(renderTimer);
      await renderMarkdown(aiNode.textEl, text, isDark);
      if (!text.trim()) aiNode.textEl.textContent = '（没有生成内容）';
      aiNode.meta.textContent =
        `${new Date().toLocaleTimeString('zh-CN', { hour12: false })} · ${((Date.now() - startedAt) / 1000).toFixed(1)}s`;
    } catch (err) {
      aiNode.textEl.textContent = `请求失败：${err.message}`;
    } finally {
      firstToken();
      streaming = false;
      if (sendBtn) sendBtn.disabled = false;
      setStatus('');
    }
  }

  /** 思考块默认收起——实测一个回答的思考量常常是正文的数倍。 */
  function ensureThinking(bodyEl) {
    let box = bodyEl.querySelector('.msg-thinking');
    if (!box) {
      box = document.createElement('details');
      box.className = 'msg-thinking';
      box.innerHTML = '<summary>思考过程</summary><div class="msg-thinking-body"></div>';
      bodyEl.insertBefore(box, bodyEl.firstChild);
    }
    return box.querySelector('.msg-thinking-body');
  }

  async function renderMarkdown(el, markdown, isDark) {
    if (!window.Vditor || !window.Vditor.preview) {
      el.textContent = markdown;  // 渲染器没加载出来也不能让内容消失
      return;
    }
    try {
      await window.Vditor.preview(el, markdown, {
        cdn: opts.cdn,
        markdown: { toc: false, mark: false },
        math: { engine: 'KaTeX', inlineDigit: true },
        theme: { current: isDark ? 'dark' : 'light' },
        hljs: { style: isDark ? 'atom-one-dark' : 'atom-one-light' },
      });
    } catch {
      el.textContent = markdown;
    }
    // Vditor 硬编码 mermaid 的 useMaxWidth，大图会被压到读不出来；
    // 宽表格同理。每次重渲染后都要修——流式输出会反复渲染。
    fixMermaidSizing(el);
    wrapWideTables(el);
  }

  /**
   * 组装一组流式事件处理函数。
   *
   * 「新提问」和「编辑重发」的事件处理完全一样，只有 start（截断数量）
   * 那一条不同。各写一份的话，加了新事件类型就会漏改其中一处——
   * 表现是「某个入口偶尔少显示东西」，很难联想到是这里。
   */
  function makeHandlers({ aiNode, isDark, onText, onThinking, onStart }) {
    return {
      onText,
      onThinking,
      onSources: (event) => setStatus(`检索到 ${(event.sources || []).length} 条相关内容`),
      onWebSearching: (event) => {
        const label = { paper: '学术文献', code: '代码仓库', web: '网页', auto: '全部来源' }[event.kind] || '';
        setStatus(`正在联网检索${label ? `（${label}）` : ''}：${event.query}`);
      },
      onWebSources: (event) => {
        const count = (event.results || []).length;
        if ((event.errors || []).length && !count) {
          // 联网失败必须说出来。不说的话用户只看到答案变短了，
          // 会以为是模型偷懒，实际是某个来源没配置或没连上。
          setStatus(`联网未返回结果：${event.errors.join('；')}`);
        } else {
          setStatus(`联网返回 ${count} 条`);
        }
      },
      onDone: (event) => {
        const cards = citationCards(event.citations);
        if (cards) aiNode.bodyEl.appendChild(cards);
        const web = webCards(event.web_citations);
        if (web) aiNode.bodyEl.appendChild(web);
      },
      onError: (message) => setStatus(message || '生成出错'),
      onSaved: (event) => { aiNode.node.dataset.messageId = event.assistant_message_id || ''; },
      onStart,
      isDark,
    };
  }

  /**
   * 消费一个 SSE 流，按事件类型回调。
   *
   * 抽出来给「新提问」和「编辑重发」共用——两处的解析逻辑一模一样，
   * 各写一份的话迟早出现「一处支持了某事件、另一处没有」的错位。
   */
  async function consumeStream(response, handlers) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // SSE 以空行分隔事件；最后一段可能不完整，留在 buffer 里等下一批
      const chunks = buffer.split('\n\n');
      buffer = chunks.pop();

      for (const chunk of chunks) {
        const line = chunk.split('\n').find((l) => l.startsWith('data: '));
        if (!line) continue;
        let event;
        try {
          event = JSON.parse(line.slice(6));
        } catch {
          continue;
        }
        if (event.type === 'text') handlers.onText?.(event.text || '');
        else if (event.type === 'thinking') handlers.onThinking?.(event.text || '');
        else if (event.type === 'sources') handlers.onSources?.(event);
        else if (event.type === 'web_searching') handlers.onWebSearching?.(event);
        else if (event.type === 'web_sources') handlers.onWebSources?.(event);
        else if (event.type === 'done') handlers.onDone?.(event);
        else if (event.type === 'error') handlers.onError?.(event.message);
        else if (event.type === 'saved') handlers.onSaved?.(event);
        else if (event.type === 'start') handlers.onStart?.(event);
      }
    }
  }

  /* ---------------- 发送 ---------------- */

  async function send() {
    if (streaming || !conversationId) return;
    const question = inputEl.value.trim();
    if (!question && !attachments.length) return;

    const payload = {
      question,
      attachments: attachments.map((a) => ({
        kind: a.kind, name: a.name, media_type: a.media_type, path: a.path,
      })),
    };

    inputEl.value = '';
    attachments = [];
    renderAttachments();
    streaming = true;
    if (sendBtn) sendBtn.disabled = true;
    setStatus('');

    // 附件里的图片本地也能立刻显示，不用等服务端回显
    const userNode = addMessage('user', question || '（附件）');
    payload.attachments
      .filter((a) => a.kind === 'image')
      .forEach((a) => {
        const img = document.createElement('img');
        img.className = 'msg-image';
        img.alt = a.name;
        // 服务端存的是内容寻址路径；这里用文件名回读不到，改为显示占位说明
        img.title = a.name;
        img.src = '';
        img.style.display = 'none';
        userNode.bodyEl.insertBefore(img, userNode.textEl);
      });

    const startedAt = Date.now();
    const aiNode = addMessage('assistant', '');
    const stopTyping = showTyping(aiNode.bodyEl);
    const isDark = document.documentElement.dataset.theme === 'dark';
    let text = '';
    let thinking = '';
    let dirty = false;
    let renderTimer = null;
    let typingStopped = false;

    /** 第一段正文到达就把「正在输入」撤掉。 */
    const firstToken = () => {
      if (typingStopped) return;
      typingStopped = true;
      stopTyping();
    };

    const scheduleRender = () => {
      dirty = true;
      if (renderTimer) return;
      renderTimer = setTimeout(async () => {
        renderTimer = null;
        if (!dirty) return;
        dirty = false;
        await renderMarkdown(aiNode.textEl, text, isDark);
        scrollToBottom();
      }, RENDER_THROTTLE_MS);
    };

    try {
      const response = await fetch(url('stream'), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'text/event-stream',
          'X-CSRFToken': opts.csrfToken,
        },
        body: JSON.stringify(payload),
      });
      if (!response.ok || !response.body) {
        throw new Error(`HTTP ${response.status}`);
      }

      await consumeStream(response, makeHandlers({
        aiNode,
        isDark,
        onText: (chunk) => { firstToken(); text += chunk; scheduleRender(); },
        onThinking: (chunk) => {
          thinking += chunk;
          ensureThinking(aiNode.bodyEl).textContent = thinking;
          // 模型思考时正文还没来，但界面上得让人知道它在动
          if (!typingStopped) setStatus('正在思考…');
        },
      }));

      // 收尾：用最终文本渲染一次，保证与落库内容一致
      if (renderTimer) clearTimeout(renderTimer);
      await renderMarkdown(aiNode.textEl, text, isDark);
      if (!text.trim()) {
        aiNode.textEl.textContent = '（没有生成内容）';
      }
      aiNode.meta.textContent =
        `${new Date().toLocaleTimeString('zh-CN', { hour12: false })} · ${((Date.now() - startedAt) / 1000).toFixed(1)}s`;
    } catch (err) {
      aiNode.textEl.textContent = `请求失败：${err.message}`;
    } finally {
      firstToken();
      streaming = false;
      if (sendBtn) sendBtn.disabled = false;
      setStatus('');
    }
  }

  /* ---------------- 附件 ---------------- */

  function renderAttachments() {
    if (!attachEl) return;
    attachEl.innerHTML = '';
    attachEl.hidden = attachments.length === 0;
    attachments.forEach((item, index) => {
      const chip = document.createElement('span');
      chip.className = 'attach-chip';
      chip.innerHTML = `<span>${escapeHtml(item.name || item.kind)}</span>
        <button type="button" title="移除">×</button>`;
      chip.querySelector('button').addEventListener('click', () => {
        attachments.splice(index, 1);
        renderAttachments();
      });
      attachEl.appendChild(chip);
    });
  }

  async function uploadFiles(files) {
    for (const file of files) {
      setStatus(`上传 ${file.name}…`);
      const form = new FormData();
      form.append('file', file);
      try {
        const response = await fetch(opts.urls.upload, {
          method: 'POST',
          headers: { Accept: 'application/json', 'X-CSRFToken': opts.csrfToken },
          body: form,
        });
        const data = await response.json().catch(() => ({}));
        if (!response.ok || data.ok === false) throw new Error(data.error || '上传失败');
        attachments.push(data.attachment);
      } catch (err) {
        setStatus(`上传失败：${err.message}`);
        return;
      }
    }
    renderAttachments();
    setStatus('');
  }

  /* ---------------- @ 引用论文 ---------------- */

  function hideMention() {
    if (mentionEl) mentionEl.hidden = true;
    mentionStart = -1;
  }

  async function showMention(keyword) {
    if (!mentionEl) return;
    try {
      const response = await fetch(
        `${opts.urls.paperSearch}?q=${encodeURIComponent(keyword)}`,
        { headers: { Accept: 'application/json' } },
      );
      const data = await response.json();
      const items = data.papers || [];
      mentionEl.innerHTML = '';
      if (!items.length) {
        mentionEl.hidden = true;
        return;
      }
      items.forEach((paper) => {
        const row = document.createElement('button');
        row.type = 'button';
        row.className = 'mention-item';
        row.innerHTML = `<span class="truncate">${escapeHtml(paper.title)}</span>
          <span class="faint small">${paper.year || ''}</span>`;
        row.addEventListener('click', () => pickMention(paper));
        mentionEl.appendChild(row);
      });
      mentionEl.hidden = false;
    } catch {
      mentionEl.hidden = true;
    }
  }

  async function pickMention(paper) {
    // 把 @关键词 从输入框里去掉，改把这篇论文加进会话范围
    if (mentionStart >= 0) {
      inputEl.value = inputEl.value.slice(0, mentionStart);
    }
    hideMention();
    inputEl.focus();
    if (!conversationId) return;
    try {
      const current = await jsonRequest(url('scope'), {
        method: 'PATCH',
        body: { paper_ids: [paper.id] },
      });
      setStatus(`已限定在这篇论文内检索：${paper.title.slice(0, 40)}`);
      if (current) inputEl.dispatchEvent(new Event('input'));
    } catch (err) {
      setStatus(`限定范围失败：${err.message}`);
    }
  }

  /* ---------------- 会话操作 ---------------- */

  async function newConversation() {
    try {
      const data = await jsonRequest(opts.urls.create, { body: {} });
      window.location.href = `${opts.urls.chatPage}?c=${data.conversation.id}`;
    } catch (err) {
      setStatus(`新建失败：${err.message}`);
    }
  }

  async function rename() {
    if (!conversationId || !titleEl) return;
    const title = (titleEl.value || '').trim();
    if (!title) return;
    try {
      await jsonRequest(url('rename'), { method: 'PATCH', body: { title } });
      setStatus('已重命名');
      const item = document.querySelector(`[data-conversation-id="${conversationId}"] .chat-item-title`);
      if (item) item.textContent = title;
    } catch (err) {
      setStatus(`重命名失败：${err.message}`);
    }
  }

  async function remove() {
    if (!conversationId) return;
    if (!window.confirm('删除这个会话？消息会一并删除，不可恢复。')) return;
    try {
      await jsonRequest(url('remove'), { method: 'DELETE' });
      window.location.href = opts.urls.chatPage;
    } catch (err) {
      setStatus(`删除失败：${err.message}`);
    }
  }

  /* ---------------- 事件绑定 ---------------- */

  document.querySelector('[data-chat-new]')?.addEventListener('click', newConversation);
  document.querySelector('[data-chat-rename]')?.addEventListener('click', rename);
  document.querySelector('[data-chat-delete]')?.addEventListener('click', remove);

  sendBtn?.addEventListener('click', send);

  // 触屏设备上回车是「换行」——软键盘的回车键长在拇指正上方，
  // 想换行却误发一条半截消息是这里最容易踩的坑（而且 Shift+回车在
  // 手机软键盘上根本按不出来）。手机上发送统一交给发送按钮。
  // 用 pointer:coarse 判断输入方式而不是屏幕宽度：平板外接键盘时仍该回车发送。
  const isTouch = window.matchMedia('(pointer: coarse)').matches;

  inputEl.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
      if (isTouch) return;   // 交给默认行为：插入换行
      event.preventDefault();
      if (mentionEl && !mentionEl.hidden) return;  // 正在选论文，回车不发送
      send();
    } else if (event.key === 'Escape') {
      hideMention();
    }
  });

  inputEl.addEventListener('input', () => {
    const value = inputEl.value;
    const caret = inputEl.selectionStart;
    const at = value.lastIndexOf('@', caret - 1);
    if (at >= 0 && caret - at <= 40 && !/\s/.test(value.slice(at + 1, caret))) {
      mentionStart = at;
      showMention(value.slice(at + 1, caret));
    } else {
      hideMention();
    }
  });

  // 粘贴图片：这是最常用的多模态入口（截个公式/图表直接问）
  inputEl.addEventListener('paste', (event) => {
    const files = [...(event.clipboardData?.files || [])];
    if (files.length) {
      event.preventDefault();
      uploadFiles(files);
    }
  });

  fileEl?.addEventListener('change', () => {
    if (fileEl.files?.length) uploadFiles([...fileEl.files]);
    fileEl.value = '';
  });

  // 拖拽到输入区
  ['dragover', 'drop'].forEach((type) => {
    inputEl.addEventListener(type, (event) => {
      event.preventDefault();
      if (type === 'drop' && event.dataTransfer?.files?.length) {
        uploadFiles([...event.dataTransfer.files]);
      }
    });
  });

  // 服务端渲染出来的历史消息也要有操作条（复制 / 编辑）。
  // 它们不是 addMessage 建的，所以在这里补上——否则「刷新之后就不能编辑了」
  // 这种只在特定路径下出现的怪毛病会很难解释。
  messagesEl.querySelectorAll('.msg[data-message-id]').forEach((node) => {
    if (node.querySelector('.msg-actions')) return;
    const isUser = node.classList.contains('msg-user');
    const { bar, meta } = buildActions(node);
    if (isUser) {
      const edit = document.createElement('button');
      edit.type = 'button';
      edit.className = 'msg-action msg-edit';
      edit.textContent = '编辑';
      edit.title = '编辑这条提问并重新提问';
      edit.addEventListener('click', () => startEdit(node));
      bar.insertBefore(edit, meta);
    }
    meta.textContent = '历史消息';
  });

  /* ---------------- 输入框：自动增高 ---------------- */

  const grow = () => {
    inputEl.style.height = 'auto';
    inputEl.style.height = `${Math.min(inputEl.scrollHeight, 240)}px`;
  };
  inputEl.addEventListener('input', grow);
  grow();

  /* ---------------- 回到底部 ---------------- */

  const scrollBtn = document.querySelector('[data-chat-scroll]');
  if (scrollBtn) {
    scrollBtn.addEventListener('click', () => {
      messagesEl.scrollTo({ top: messagesEl.scrollHeight, behavior: 'smooth' });
    });
    // 只在「用户往上翻、不在底部」时出现。一直显示会挡住内容。
    messagesEl.addEventListener('scroll', () => {
      const away = messagesEl.scrollHeight - messagesEl.scrollTop - messagesEl.clientHeight;
      scrollBtn.classList.toggle('is-visible', away > 160);
    });
  }

  /* ---------------- 空状态的示例问题 ---------------- */

  document.querySelectorAll('[data-chat-examples] .example-chip').forEach((chip) => {
    chip.addEventListener('click', async () => {
      inputEl.value = chip.textContent.trim();
      grow();
      inputEl.focus();
      // 还没有会话时（首次进入）先建一个再发。不这么做的话，
      // 输入框是禁用的、示例卡片点了毫无反应——空状态下最容易踩的一脚。
      if (!conversationId) {
        try {
          const data = await jsonRequest(opts.urls.create, { body: {} });
          conversationId = data.conversation.id;
          // 地址栏跟着变，刷新后才不会又回到「没有会话」的状态
          window.history.replaceState(null, '', `${opts.urls.chatPage}?c=${conversationId}`);
        } catch (err) {
          setStatus(`新建会话失败：${err.message}`);
          return;
        }
      }
      send();
    });
  });

  /* ---------------- 会话列表：搜索与删除 ---------------- */

  const searchInput = document.querySelector('[data-chat-search]');
  searchInput?.addEventListener('input', () => {
    const keyword = searchInput.value.trim().toLowerCase();
    document.querySelectorAll('.chat-item').forEach((item) => {
      const title = (item.querySelector('.chat-item-title')?.textContent || '').toLowerCase();
      item.style.display = !keyword || title.includes(keyword) ? '' : 'none';
    });
  });

  document.querySelectorAll('[data-chat-del]').forEach((btn) => {
    btn.addEventListener('click', async (event) => {
      // 必须阻止冒泡：这个按钮在 <a> 里面，不拦下来会先跳转到那个会话再删除
      event.preventDefault();
      event.stopPropagation();
      const id = btn.dataset.chatDel;
      if (!window.confirm('删除这个会话？消息会一并删除，不可恢复。')) return;
      try {
        await jsonRequest(opts.urls.remove.replace('__ID__', id), { method: 'DELETE' });
        const item = btn.closest('.chat-item');
        const wasActive = item?.classList.contains('is-active');
        item?.remove();
        // 删掉的是当前打开的会话时就回列表页，否则界面会停在
        // 一个已经不存在的会话上，之后所有操作都会 404
        if (wasActive) window.location.href = opts.urls.chatPage;
      } catch (err) {
        setStatus(`删除失败：${err.message}`);
      }
    });
  });

  scrollToBottom();
  return { send, uploadFiles };
}
