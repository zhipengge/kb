/* 笔记编辑器：Vditor 即时渲染（IR）模式 + 静默保存。
 *
 * 为什么是 IR 模式：它是浏览器里的 Typora——Markdown 标记自动隐去，
 * 光标移进去时重新展开。编辑和预览是同一个面板，不需要左右分栏也不用来回切。
 *
 * 三个容易踩的点，都在代码里标了原因：
 *   1. `cdn` 必须指到本地，否则它会去 unpkg 拿 lute，内网部署直接卡住；
 *   2. `cache.enable` 必须关，否则 Vditor 自己往 localStorage 存草稿，
 *      会和「DB 为准 + 磁盘镜像是给别人（Obsidian）编辑的」这套模型打架；
 *   3. 保存走 fetch 而不是表单提交，否则每次保存页面重定向，光标位置全丢。
 */

/* 停止输入多久之后自动保存。
 *
 * 2.5 秒是权衡的结果：太短会在连续打字时反复触发请求（每次都要写库 + 写盘），
 * 太长则失去「不用管保存」的意义。实测这个间隔下连续输入一段话只会存一次。
 */
import { fixMermaidSizing, observeMermaid } from '/static/js/vditor-fix.js';

const AUTOSAVE_MS = 2500;

/**
 * @param {object} opts
 * @param {string} opts.mountId     编辑器容器 id
 * @param {string} opts.saveUrl     保存端点
 * @param {string} opts.csrfToken   CSRF token
 * @param {string} opts.value       初始 Markdown
 * @param {number} opts.version     乐观锁版本号
 * @param {string} opts.title       笔记标题（保存时一并提交）
 */
export function createNoteEditor(opts) {
  const mount = document.getElementById(opts.mountId);
  if (!mount || typeof window.Vditor === 'undefined') return null;

  const statusEl = document.querySelector('[data-editor-status]');
  const saveBtn = document.querySelector('[data-editor-save]');

  let version = opts.version;
  let dirty = false;
  let saving = false;
  let lastSavedAt = null;
  // 乐观锁冲突之后**停掉自动保存**。继续自动重试的后果是：每次都拿同一个
  // 过期版本号去写，每次都失败；更糟的情况是用户同时用 Obsidian 改了磁盘
  // 文件，自动保存会把这个冲突拖成一场看不见的拉锯。
  let conflicted = false;
  let autosaveTimer = null;

  const setStatus = (text, kind) => {
    if (!statusEl) return;
    statusEl.textContent = text;
    statusEl.dataset.state = kind || '';
  };

  const setDirty = (next) => {
    dirty = next;
    // 保存按钮一直保持醒目（btn-primary）——它本来就是这一页的主操作。
    // 之前靠 primary 与否来暗示「有没有改动」，结果常态下它是灰的，
    // 看起来像个次要按钮，反而找不到。
    if (saveBtn) {
      saveBtn.disabled = saving || (!next && !conflicted);
    }
    if (next) setStatus('未保存', 'dirty');
  };

  /** 排一次自动保存。连续输入时会被不断推后，只在停手后真正触发。 */
  const scheduleAutosave = () => {
    if (autosaveTimer) clearTimeout(autosaveTimer);
    autosaveTimer = setTimeout(() => {
      autosaveTimer = null;
      if (dirty && !saving && !conflicted) save({ auto: true });
    }, AUTOSAVE_MS);
  };

  async function save({ auto = false } = {}) {
    if (saving || !dirty || conflicted) return;
    saving = true;
    setStatus(auto ? '自动保存中…' : '保存中…', 'saving');

    // 正文与版本号由这里决定，其余字段（标题、类型、状态）从表单里取——
    // 这样改了「类型/状态」点保存也会一起生效，不需要各自接一套监听。
    const payload = { content_md: vditor.getValue(), version };
    const form = document.getElementById(opts.formId);
    if (form) {
      for (const [key, value] of new FormData(form).entries()) {
        // version 用 JS 跟踪的那个：表单里的是渲染时的旧值，
        // 保存成功后它不会更新，再存一次就会被乐观锁误判成冲突。
        if (key === 'csrf_token' || key === 'content_md' || key === 'version') continue;
        payload[key] = value;
      }
    }

    try {
      const response = await fetch(opts.saveUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          // 后端据此返回 JSON 而不是 302——见 views._wants_json()
          Accept: 'application/json',
          'X-CSRFToken': opts.csrfToken,
        },
        body: JSON.stringify(payload),
      });
      const data = await response.json().catch(() => ({}));

      if (response.status === 409) {
        // 乐观锁冲突：多半是另一处（网页另一标签页、外部编辑器同步、
        // AI 任务）改过这篇笔记。**不能自动覆盖**，否则会静默吞掉那边的改动。
        conflicted = true;
        if (autosaveTimer) clearTimeout(autosaveTimer);
        setStatus('内容已被其它地方修改，自动保存已停止', 'error');
        if (window.confirm(`${data.error || '笔记已被其它操作修改。'}\n\n刷新页面放弃当前改动？`)) {
          window.location.reload();
        }
        return;
      }
      if (!response.ok || data.ok === false) {
        setStatus(`保存失败：${data.error || response.status}`, 'error');
        return;
      }

      version = data.version;
      lastSavedAt = new Date();
      setDirty(false);
      const time = lastSavedAt.toLocaleTimeString('zh-CN', { hour12: false });
      setStatus(auto ? `已自动保存 ${time}` : `已保存 ${time}`, 'ok');
    } catch (err) {
      setStatus(`保存失败：${err && err.message ? err.message : err}`, 'error');
    } finally {
      saving = false;
      if (saveBtn) saveBtn.disabled = !dirty;
    }
  }

  const vditor = new window.Vditor(mount, {
    cdn: opts.cdn,
    mode: 'ir',
    height: '100%',
    lang: 'zh_CN',
    icon: 'ant',
    theme: document.documentElement.dataset.theme === 'dark' ? 'dark' : 'classic',
    value: opts.value,
    // 关掉内置草稿：内容以数据库为准，磁盘 md 是给别人（Obsidian）编辑的镜像。
    // 让它再存一份 localStorage 草稿，会出现「到底哪份是真的」的第三个副本。
    cache: { enable: false },
    counter: { enable: true, type: 'markdown' },
    outline: { enable: false, position: 'left' },
    // 公式引擎。论文笔记里全是 LaTeX，不渲染等于没法读。
    math: { engine: 'KaTeX', inlineDigit: true, macros: {} },
    preview: {
      math: { engine: 'KaTeX', inlineDigit: true },
      markdown: { toc: true, mark: false },
      theme: { current: document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light' },
    },
    toolbar: [
      'headings', 'bold', 'italic', 'strike', '|',
      'list', 'ordered-list', 'check', '|',
      'quote', 'line', 'code', 'inline-code', '|',
      'link', 'table', '|',
      'undo', 'redo', '|',
      'edit-mode', 'both', 'preview', 'outline', '|',
      'fullscreen',
    ],
    input: () => {
      setDirty(true);
      scheduleAutosave();
    },
    after: () => {
      setStatus(dirty ? '未保存' : '已就绪', dirty ? 'dirty' : '');
      // Vditor 硬编码了 mermaid 的 useMaxWidth，大图会被压成 4px 字号。
      // 渲染完立刻修一次，并持续盯着——编辑时 Vditor 会不断重渲染。
      fixMermaidSizing(mount);
      observeMermaid(mount);
    },
  });

  saveBtn?.addEventListener('click', () => save());

  // Ctrl/Cmd + S 立即保存（不等自动保存的延时），并吃掉浏览器的「保存网页」行为
  document.addEventListener('keydown', (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 's') {
      event.preventDefault();
      if (autosaveTimer) clearTimeout(autosaveTimer);
      save();
    }
  });

  // 有未保存改动时离开要拦一下
  window.addEventListener('beforeunload', (event) => {
    if (!dirty) return;
    event.preventDefault();
    event.returnValue = '';
  });

  // 主题切换时把编辑器也换过去，否则暗色页面里嵌一个亮色编辑器很刺眼
  window.addEventListener('kb:theme-changed', (event) => {
    const dark = event.detail && event.detail.theme === 'dark';
    try {
      vditor.setTheme(dark ? 'dark' : 'classic',
                      dark ? 'dark' : 'light',
                      dark ? 'dark' : 'light');
    } catch { /* 主题切换失败不影响编辑 */ }
  });

  return { save, get value() { return vditor.getValue(); } };
}
