/* 文库的列表 / 卡片视图切换。
 *
 * 纯前端切换、状态存本地：这不是数据，是「这台设备上我习惯怎么看」。
 * 存服务端要为一个显示偏好加设置项和一次写库，不划算。
 * 服务端仍然渲染两种视图（都在 DOM 里，切换只是隐藏其一），
 * 所以禁用 JS 时至少列表视图是可用的。
 */

const STORAGE_KEY = 'kb.library.view';
const VIEWS = ['list', 'card'];

function readView() {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    return VIEWS.includes(saved) ? saved : 'list';
  } catch {
    return 'list';
  }
}

function applyView(view) {
  document.querySelectorAll('[data-view]').forEach((el) => {
    el.hidden = el.dataset.view !== view;
  });
  document.querySelectorAll('[data-view-set]').forEach((btn) => {
    const active = btn.dataset.viewSet === view;
    btn.classList.toggle('btn-primary', active);
    btn.setAttribute('aria-pressed', String(active));
  });
}

function init() {
  const buttons = document.querySelectorAll('[data-view-set]');
  if (!buttons.length) return;

  applyView(readView());

  buttons.forEach((btn) => {
    btn.addEventListener('click', () => {
      const view = btn.dataset.viewSet;
      applyView(view);
      try { localStorage.setItem(STORAGE_KEY, view); } catch { /* 隐私模式 */ }
    });
  });
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
