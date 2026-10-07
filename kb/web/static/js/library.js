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

/* 窄屏（手机）一律用卡片视图。
 *
 * 列表视图是四列表格（标题 / 年份会议 / 标签 / 状态），在 390px 宽里
 * 每格只剩几十像素，标题会被压成竖排单字——不是「不好看」，是读不了。
 * 卡片视图本来就是自适应的，窄屏下每张卡占满一行，正好。
 *
 * 这里**不覆盖用户的存储偏好**：只在窄屏生效，回到宽屏仍是用户选的那个。
 * 切换按钮同时也藏起来，免得点到一个当前无效的开关。
 */
const NARROW = '(max-width: 820px)';
const narrow = window.matchMedia(NARROW);

function applyResponsiveView(storedView) {
  applyView(narrow.matches ? 'card' : storedView);
  document.querySelectorAll('[data-view-set]').forEach((btn) => {
    // 隐藏而不是禁用：手机上一整排无效控件只是占地方
    btn.hidden = narrow.matches;
  });
}

function init() {
  const buttons = document.querySelectorAll('[data-view-set]');
  if (!buttons.length) return;

  applyResponsiveView(readView());

  // MediaQueryList 的 change 只在**跨越断点**时触发，普通 resize 不会——
  // 这正是想要的：用户在手机上手动切了视图，不会被随后的滚动/resize 打回去。
  const onCross = () => applyResponsiveView(readView());
  if (narrow.addEventListener) narrow.addEventListener('change', onCross);
  else if (narrow.addListener) narrow.addListener(onCross);   // 老 Safari

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
