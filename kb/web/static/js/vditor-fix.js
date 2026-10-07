/* Vditor 渲染结果的两处修正。
 *
 * 都是「库写死了默认值、不给你配置入口」造成的，只能在渲染后动 DOM。
 */

/**
 * 修正 Mermaid 图的尺寸。
 *
 * **为什么必须修。** Vditor 把 mermaid 的配置写死在源码里：
 *
 *     flowchart: {htmlLabels: true, useMaxWidth: true}
 *
 * `useMaxWidth: true` 会让 mermaid 把 SVG 设成 `width:100%` + `max-width`，
 * 也就是**等比缩放到容器宽度**。对一张 40 个节点的流程图，塞进工作台右栏
 * 那 500px 里，字号会被压到 4px —— 图还在，但一个字都读不出来。
 *
 * 论文笔记里的架构图/流程图天生就宽（横向的数据流、并排的模块），
 * 被压扁等于白画。所以这里把宽度还原成图的**自然尺寸**，外面套一层
 * 可横向滚动的容器：图大就左右滚，而不是被压成一片糊。
 */
export function fixMermaidSizing(root = document) {
  const containers = root.querySelectorAll('.language-mermaid');
  containers.forEach((box) => {
    const svg = box.querySelector('svg');
    if (!svg) return;
    // 已经处理过就不重复套容器（Vditor 会因为编辑而重渲染）
    if (box.parentElement?.classList.contains('mermaid-scroll')) return;

    const viewBox = svg.getAttribute('viewBox');
    if (!viewBox) return;
    const parts = viewBox.split(/[\s,]+/).map(Number);
    const natural = parts[2];
    if (!natural || !Number.isFinite(natural)) return;

    // 还原自然宽度。mermaid 同时写死了 style 里的 max-width，
    // 不清掉的话设了 width 也还是会被它压回去。
    svg.style.maxWidth = 'none';
    svg.style.width = `${Math.ceil(natural)}px`;
    svg.style.height = 'auto';
    svg.removeAttribute('width');
    svg.removeAttribute('height');

    const scroller = document.createElement('div');
    scroller.className = 'mermaid-scroll';
    box.parentNode.insertBefore(scroller, box);
    scroller.appendChild(box);
  });
}

/**
 * 持续监听容器内的 Mermaid 重渲染。
 *
 * 编辑器里改一个字，Vditor 就会把整段 mermaid 重新渲染一遍——新生成的 SVG
 * 带着 `useMaxWidth` 的痕迹回来，上面那次修正白做。所以得盯着 DOM 变化
 * 反复修，而不是渲染完修一次就完事。
 *
 * 节流是必要的：mermaid 渲染会引发一连串 DOM 变更，每次都同步扫一遍
 * 会让打字变卡。
 */
export function observeMermaid(root, { throttleMs = 300 } = {}) {
  if (!root || typeof MutationObserver === 'undefined') return () => {};

  let timer = null;
  const observer = new MutationObserver(() => {
    if (timer) return;
    timer = setTimeout(() => {
      timer = null;
      fixMermaidSizing(root);
    }, throttleMs);
  });

  observer.observe(root, { childList: true, subtree: true });
  return () => observer.disconnect();
}

/**
 * 表格外套一层可横向滚动的容器。
 *
 * 论文笔记里的对比表动辄七八列，在窄栏里会被挤成每格两三个字。
 * 和 mermaid 同理：能滚就不要压。
 */
export function wrapWideTables(root = document) {
  root.querySelectorAll('table').forEach((table) => {
    if (table.parentElement?.classList.contains('table-scroll')) return;
    const scroller = document.createElement('div');
    scroller.className = 'table-scroll';
    table.parentNode.insertBefore(scroller, table);
    scroller.appendChild(table);
  });
}
