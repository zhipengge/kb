/* 阅读工作台的面板控制：折叠、聚焦、标签页。
 *
 * 折叠状态存 localStorage 而不是服务端——这是**看**的偏好，不是数据。
 * 存服务端要为一个纯视觉开关加一次写库与一个设置项，不划算。
 *
 * 布局状态放在容器的 data-* 属性上，CSS 据此改 grid 模板列，
 * JS 不直接碰样式，避免两处各写一套尺寸。
 */

const STORAGE_KEY = 'kb.wb.layout';

function loadLayout() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY)) || {};
  } catch {
    return {};
  }
}

function saveLayout(state) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch {
    /* 隐私模式下 localStorage 可能不可写，忽略即可，不影响使用 */
  }
}

/**
 * @param {HTMLElement} root 带 .workbench 类的容器
 */
export function createWorkbench(root) {
  if (!root) return null;

  const saved = loadLayout();
  const setPanel = (side, collapsed) => {
    if (collapsed) root.dataset[side] = 'collapsed';
    else delete root.dataset[side];
  };

  // 恢复上次布局。默认两边都展开。
  setPanel('left', saved.left === 'collapsed');
  setPanel('right', saved.right === 'collapsed');

  const persist = () => {
    saveLayout({
      left: root.dataset.left === 'collapsed' ? 'collapsed' : 'open',
      right: root.dataset.right === 'collapsed' ? 'collapsed' : 'open',
    });
  };

  const toggle = (side) => {
    setPanel(side, root.dataset[side] !== 'collapsed');
    persist();
  };

  /** 聚焦某一侧 = 折叠另一侧；已经处于聚焦态时再按一次还原。 */
  const focus = (side) => {
    const other = side === 'left' ? 'right' : 'left';
    const isFocused = root.dataset[other] === 'collapsed' && root.dataset[side] !== 'collapsed';
    setPanel(other, !isFocused);
    setPanel(side, false);
    persist();
  };

  root.querySelectorAll('[data-wb-toggle]').forEach((btn) => {
    btn.addEventListener('click', () => toggle(btn.dataset.wbToggle));
  });
  root.querySelectorAll('[data-wb-rail]').forEach((rail) => {
    rail.addEventListener('click', () => toggle(rail.dataset.wbRail));
  });
  root.querySelectorAll('[data-wb-focus]').forEach((btn) => {
    btn.addEventListener('click', () => focus(btn.dataset.wbFocus));
  });

  /* ---- 标签页（笔记 / 信息） ---- */
  const tabs = [...root.querySelectorAll('[data-wb-tab]')];
  const panels = [...root.querySelectorAll('[data-wb-tabpanel]')];
  const selectTab = (name, { remember = true } = {}) => {
    tabs.forEach((t) => t.setAttribute('aria-selected', String(t.dataset.wbTab === name)));
    panels.forEach((p) => { p.hidden = p.dataset.wbTabpanel !== name; });
    if (remember) {
      const state = loadLayout();
      state.tab = name;
      saveLayout(state);
    }
  };
  tabs.forEach((tab) => tab.addEventListener('click', () => selectTab(tab.dataset.wbTab)));
  if (tabs.length) {
    const initial = tabs.some((t) => t.dataset.wbTab === saved.tab) ? saved.tab : tabs[0].dataset.wbTab;
    selectTab(initial, { remember: false });
  }

  /* ---- 快捷键：[ 折叠左侧，] 折叠右侧 ---- */
  document.addEventListener('keydown', (event) => {
    if (event.metaKey || event.ctrlKey || event.altKey) return;
    const tag = (event.target.tagName || '').toLowerCase();
    // 编辑区内不抢键——方括号在 Markdown 里是常用字符
    if (tag === 'input' || tag === 'textarea' || event.target.isContentEditable) return;
    if (event.key === '[') { event.preventDefault(); toggle('left'); }
    else if (event.key === ']') { event.preventDefault(); toggle('right'); }
  });

  return { toggle, focus, selectTab };
}
