/* kb 前端脚本
 *
 * 刻意保持很小：服务端渲染已经给出了完整可用的页面，JS 只负责三件事——
 * 主题切换、跟随系统时的实时响应、以及表单的少量体验优化。
 * 关闭 JavaScript 时页面仍然完全可用。
 */

(function () {
  'use strict';

  var root = document.documentElement;

  /* ---------------- 主题 ---------------- */

  function effectiveTheme(mode) {
    if (mode !== 'system') return mode;
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }

  function applyTheme(mode) {
    root.dataset.themeMode = mode;
    root.dataset.theme = effectiveTheme(mode);
    try { localStorage.setItem('kb-theme-mode', mode); } catch (e) { /* 隐私模式 */ }
    /* 通知内嵌的第三方组件。Vditor 有自己的一套主题，只改页面的 data-theme
       它不会跟着变——暗色页面里嵌一个亮色编辑器很刺眼，而且它自己不监听任何东西。 */
    window.dispatchEvent(new CustomEvent('kb:theme-changed', {
      detail: { theme: root.dataset.theme },
    }));
  }

  /* 用户在别处（localStorage）明确选过主题时，覆盖服务端渲染的值。
     服务端的值来自数据库设置，是这台实例的默认；localStorage 是这台浏览器的偏好。
     两者不一致时以浏览器为准——同一个知识库可能被多台设备以不同偏好访问。 */
  var stored = null;
  try { stored = localStorage.getItem('kb-theme-mode'); } catch (e) { /* ignore */ }
  if (stored && stored !== root.dataset.themeMode) {
    applyTheme(stored);
  }

  /* 跟随系统时，系统主题变化要实时响应 */
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function () {
    if (root.dataset.themeMode === 'system') {
      root.dataset.theme = effectiveTheme('system');
    }
  });

  var toggle = document.getElementById('theme-toggle');
  if (toggle) {
    toggle.addEventListener('click', function () {
      /* 三态循环：当前亮 -> 暗 -> 跟随系统 -> 亮。
         只做两态的话，「跟随系统」这个选项一旦离开就回不去了。 */
      var order = ['light', 'dark', 'system'];
      var current = root.dataset.themeMode || 'system';
      var next = order[(order.indexOf(current) + 1) % order.length];
      applyTheme(next);
      toggle.title = '切换主题（当前：' + next + '）';
    });
  }

  /* ---------------- 表单体验 ---------------- */

  /* 数字输入框：滚轮常常是误触，会静默改掉值 */
  document.querySelectorAll('input[type="number"]').forEach(function (input) {
    input.addEventListener('wheel', function (event) {
      if (document.activeElement === input) event.preventDefault();
    }, { passive: false });
  });

  /* 密钥输入框：清空时给出明确提示，避免「以为改了其实没改」 */
  document.querySelectorAll('input[type="password"]').forEach(function (input) {
    input.addEventListener('input', function () {
      var hint = input.parentElement.querySelector('.field-help');
      if (!hint) return;
      if (input.value === '') {
        hint.textContent = '留空保存将清除已存储的密钥。';
      }
    });
  });

  /* ---------------- 表单防重复提交 ---------------- */

  document.querySelectorAll('form').forEach(function (form) {
    form.addEventListener('submit', function () {
      var button = form.querySelector('button[type="submit"]');
      if (!button || form.dataset.nosubmit === '1') return;
      /* 延迟禁用：立刻禁用会让按钮的 name/value 不被提交 */
      setTimeout(function () {
        button.disabled = true;
        button.style.opacity = '.6';
      }, 0);
      setTimeout(function () {
        button.disabled = false;
        button.style.opacity = '';
      }, 8000);
    });
  });

  /* ---------------- 相对时间的本地化刷新 ----------------
     服务端渲染的时间是相对时间（"3 分钟前"），页面停留久了会失真。
     这里定时刷新一次，成本很低。 */

  function refreshRelativeTimes() {
    document.querySelectorAll('[data-timestamp]').forEach(function (el) {
      var ts = new Date(el.dataset.timestamp);
      if (isNaN(ts)) return;
      var seconds = (Date.now() - ts.getTime()) / 1000;
      var text;
      if (seconds < 10) text = '刚刚';
      else if (seconds < 60) text = Math.floor(seconds) + ' 秒前';
      else if (seconds < 3600) text = Math.floor(seconds / 60) + ' 分钟前';
      else if (seconds < 86400) text = Math.floor(seconds / 3600) + ' 小时前';
      else text = Math.floor(seconds / 86400) + ' 天前';
      el.textContent = text;
    });
  }
  refreshRelativeTimes();
  setInterval(refreshRelativeTimes, 60000);

  /* ---------------- 公式渲染 ----------------

     论文公式以原始 LaTeX 保存（把 \frac{1}{2} 转成 "1/2" 是不可逆的丢失），
     靠 KaTeX 在浏览器里渲染成真正的数学排版。

     两个必须处理的点：

     1. **必须在 DOM 就绪后再渲染**，否则这段脚本先执行、公式元素还没解析出来。
     2. **HTMX 局部更新后要重新渲染**。HTMX 换进来的新节点不会自动触发渲染，
        表现是「刷新后公式正常、局部更新后变成一串 $...$」——这类问题很难
        跟公式本身的问题区分开。
  */

  function renderMath(root) {
    if (typeof renderMathInElement !== 'function') return;
    try {
      renderMathInElement(root || document.body, {
        delimiters: [
          { left: '$$', right: '$$', display: true },
          { left: '\\[', right: '\\]', display: true },
          { left: '$', right: '$', display: false },
          { left: '\\(', right: '\\)', display: false },
        ],
        // 遇到渲染不了的公式时保留原文并标红，而不是抛错中断整页渲染——
        // 论文里的自定义宏（\dmodel 之类）KaTeX 不认识，这种情况会经常出现
        throwOnError: false,
        errorColor: '#b91c1c',
        ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code', 'option'],
      });
    } catch (error) {
      console.warn('公式渲染失败：', error);
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function () { renderMath(); });
  } else {
    renderMath();
  }

  // HTMX 换入新内容后重新渲染公式
  document.body.addEventListener('htmx:afterSwap', function (event) {
    renderMath(event.detail.target);
  });
})();
