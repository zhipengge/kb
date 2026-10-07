/* PDF 阅读器：连续滚动。
 *
 * **为什么是连续滚动而不是一页一页翻。** 读论文时经常需要往回翻对照
 * 前面的公式、或者快速掠过中间几页看图表结构。翻页模式每换一页都要
 * 点一下、而且看不到「下一页大概是什么」；滚动是读长文档最自然的手势，
 * 也让人能靠滚动条的位置建立对全文长度的感觉。
 *
 * 页数可能上百，所以**不能一次性全渲染**：每页都画成 canvas 会吃掉
 * 几百 MB 显存并卡住主线程。做法是先按尺寸铺占位（保证滚动条长度正确、
 * 滚动时不跳），再用 IntersectionObserver 只渲染进入视野的页。
 *
 * 加一个 rootMargin 提前渲染下一页：等它进入视野才开始画的话，
 * 快速滚动时会看到一片空白，然后页面「跳」出来。
 */

import * as pdfjsLib from '/static/vendor/pdf.min.mjs';

pdfjsLib.GlobalWorkerOptions.workerSrc = '/static/vendor/pdf.worker.min.mjs';

const MIN_SCALE = 0.4;
const MAX_SCALE = 4;
const SCALE_STEP = 1.2;
// 提前一屏渲染，滚动时不会看到空白页
const PRELOAD_MARGIN = '600px';

export function createReader(root, url) {
  const canvas0 = root.querySelector('[data-reader-canvas]');
  const stage = root.querySelector('[data-reader-stage]');
  const pageInput = root.querySelector('[data-reader-page]');
  const totalLabel = root.querySelector('[data-reader-total]');
  const zoomLabel = root.querySelector('[data-reader-zoom]');
  const status = root.querySelector('[data-reader-status]');

  // 连续滚动模式下不再有「唯一那个 canvas」——每页各有一个。
  // 模板里那个是给无 JS 时占位的，这里把它去掉。
  canvas0?.remove();
  if (!stage) return null;

  let pdf = null;
  let pageCount = 0;
  let scale = 1;
  let fitWidth = true;
  let baseWidth = 612;   // letter 宽度，拿到第一页后会被真实值覆盖
  let pages = [];        // [{num, wrap, canvas, rendered}]
  let observer = null;
  let visible = 1;

  const clamp = (n, lo, hi) => Math.min(hi, Math.max(lo, n));
  const setStatus = (text) => { if (status) status.textContent = text || ''; };

  function pageSize() {
    // 减 2px 避免出现水平滚动条
    const width = fitWidth && stage.clientWidth > 0
      ? Math.max(stage.clientWidth - 2, 120)
      : baseWidth * scale;
    return width;
  }

  function applySize() {
    const width = pageSize();
    scale = clamp(width / baseWidth, MIN_SCALE, MAX_SCALE);
    if (zoomLabel) zoomLabel.textContent = `${Math.round(scale * 100)}%`;
    pages.forEach((entry) => {
      entry.wrap.style.width = `${Math.round(width)}px`;
      // 高度按比例先撑开，等真正渲染后会被 canvas 的实际高度覆盖
      entry.wrap.style.height = `${Math.round(width / entry.ratio)}px`;
    });
  }

  async function renderPage(entry) {
    if (entry.rendered || entry.rendering) return;
    entry.rendering = true;
    try {
      const page = await pdf.getPage(entry.num);
      const base = page.getViewport({ scale: 1 });
      const width = pageSize();
      const viewport = page.getViewport({ scale: width / base.width });

      const dpr = window.devicePixelRatio || 1;
      const canvas = entry.canvas;
      canvas.width = Math.floor(viewport.width * dpr);
      canvas.height = Math.floor(viewport.height * dpr);
      canvas.style.width = `${Math.floor(viewport.width)}px`;
      canvas.style.height = `${Math.floor(viewport.height)}px`;

      await page.render({
        canvasContext: canvas.getContext('2d'),
        viewport,
        transform: dpr !== 1 ? [dpr, 0, 0, dpr, 0, 0] : undefined,
      }).promise;

      entry.wrap.style.height = '';
      entry.rendered = true;
    } catch (err) {
      entry.wrap.innerHTML =
        `<div class="reader-error small">第 ${entry.num} 页渲染失败</div>`;
    } finally {
      entry.rendering = false;
    }
  }

  function clearRendered() {
    pages.forEach((entry) => {
      if (!entry.rendered) return;
      const canvas = entry.canvas;
      canvas.width = 0;
      canvas.height = 0;
      entry.rendered = false;
    });
    applySize();
  }

  async function renderVisible() {
    // 缩放过之后，视野内的页要重画
    const rect = stage.getBoundingClientRect();
    for (const entry of pages) {
      const box = entry.wrap.getBoundingClientRect();
      const inView = box.bottom > rect.top - 600 && box.top < rect.bottom + 600;
      if (inView) renderPage(entry);
    }
  }

  function currentPage() {
    // 取「顶端最接近视口顶部」的那一页作为当前页——
    // 用中心点判断的话，两页各占一半时会来回跳
    const stageTop = stage.getBoundingClientRect().top;
    let best = pages[0];
    let bestDelta = Infinity;
    for (const entry of pages) {
      const delta = Math.abs(entry.wrap.getBoundingClientRect().top - stageTop - 8);
      if (delta < bestDelta) { bestDelta = delta; best = entry; }
    }
    return best ? best.num : 1;
  }

  function goTo(n) {
    const target = pages[clamp(Math.round(n) || 1, 1, pageCount) - 1];
    if (!target) return;
    // 用 scrollIntoView 而不是算 offsetTop：后者在有缩放/边距时容易偏
    target.wrap.scrollIntoView({ block: 'start', behavior: 'auto' });
    pageInput.value = String(target.num);
  }

  function bindToolbar() {
    root.querySelector('[data-reader-prev]')?.addEventListener('click', () => goTo(visible - 1));
    root.querySelector('[data-reader-next]')?.addEventListener('click', () => goTo(visible + 1));
    root.querySelector('[data-reader-zoom-in]')?.addEventListener('click', () => {
      fitWidth = false; clearRendered(); scale = clamp(scale * SCALE_STEP, MIN_SCALE, MAX_SCALE);
      applySize(); renderVisible();
    });
    root.querySelector('[data-reader-zoom-out]')?.addEventListener('click', () => {
      fitWidth = false; clearRendered(); scale = clamp(scale / SCALE_STEP, MIN_SCALE, MAX_SCALE);
      applySize(); renderVisible();
    });
    root.querySelector('[data-reader-fit]')?.addEventListener('click', () => {
      fitWidth = true; clearRendered(); applySize(); renderVisible();
    });

    pageInput?.addEventListener('change', () => goTo(Number(pageInput.value)));
    pageInput?.addEventListener('keydown', (event) => {
      if (event.key === 'Enter') { event.preventDefault(); goTo(Number(pageInput.value)); pageInput.blur(); }
    });

    // 上一页/下一页的快捷键。上下键交给浏览器原生滚动——那才是滚动的直觉。
    root.addEventListener('mouseenter', () => { root.dataset.hover = '1'; });
    root.addEventListener('mouseleave', () => { delete root.dataset.hover; });
    document.addEventListener('keydown', (event) => {
      if (root.dataset.hover !== '1') return;
      const tag = (event.target.tagName || '').toLowerCase();
      if (tag === 'input' || tag === 'textarea' || event.target.isContentEditable) return;
      if (event.key === 'PageUp') { event.preventDefault(); goTo(visible - 1); }
      else if (event.key === 'PageDown') { event.preventDefault(); goTo(visible + 1); }
    });
  }

  /* ---- 双指捏合缩放 ---- */
  /*
   * 桌面靠工具栏的 ＋/− 缩放，手机上那两个按钮在窄屏里被 CSS 隐藏了——
   * 手指的直觉是捏合，不是去戳一个 40px 的按钮。
   *
   * 实现上分两段：捏合过程中**只改 CSS 尺寸**（画布位图不重画），
   * 松手后再按最终倍率重画一次。原因是重画一页要几十毫秒，
   * 跟着 touchmove 重画会卡成幻灯片；而拉伸位图是合成器的事，跟手。
   * 代价是捏合途中画面偏糊，松手即清晰——这是各家阅读器的通行做法。
   */
  function bindPinch() {
    let startDistance = 0;
    let startScale = 1;
    let liveScale = 1;      // 捏合过程中的当前倍率，松手时按它定稿
    let pinching = false;

    const distanceOf = (touches) => {
      const [a, b] = touches;
      return Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY);
    };

    /** 捏合途中：按倍率拉伸已有的画布，不重新渲染。 */
    const previewScale = (next) => {
      liveScale = next;
      const width = Math.round(baseWidth * next);
      pages.forEach((entry) => {
        entry.wrap.style.width = `${width}px`;
        entry.wrap.style.height = `${Math.round(width / entry.ratio)}px`;
        if (entry.rendered) {
          // canvas 的位图尺寸不动，只改它的显示尺寸
          entry.canvas.style.width = `${width}px`;
          entry.canvas.style.height = `${Math.round(width / entry.ratio)}px`;
        }
      });
      if (zoomLabel) zoomLabel.textContent = `${Math.round(next * 100)}%`;
    };

    stage.addEventListener('touchstart', (event) => {
      if (event.touches.length !== 2) return;
      pinching = true;
      startDistance = distanceOf(event.touches);
      // 从「当前实际显示倍率」起算，而不是从 scale 变量——
      // 适应宽度模式下 scale 是上一次算出来的值，和眼前看到的不一定一致
      startScale = clamp(pageSize() / baseWidth, MIN_SCALE, MAX_SCALE);
      liveScale = startScale;
    }, { passive: true });

    stage.addEventListener('touchmove', (event) => {
      if (!pinching || event.touches.length !== 2) return;
      const distance = distanceOf(event.touches);
      if (!startDistance) return;
      previewScale(
        clamp(startScale * (distance / startDistance), MIN_SCALE, MAX_SCALE),
      );
      // 阻止浏览器把这次捏合当成「缩放整个页面」。
      // 监听器因此不能是 passive 的（注册时显式 passive:false）。
      event.preventDefault();
    }, { passive: false });

    const finish = (event) => {
      if (!pinching) return;
      // 还有手指没离开就不算结束——两指先后抬起会触发两次，白重排一遍
      if (event.touches && event.touches.length > 0) return;
      pinching = false;
      startDistance = 0;
      // 按捏合结束时的倍率定稿，重画一次得到清晰画面
      fitWidth = false;
      scale = liveScale;
      clearRendered();
      applySize();
      renderVisible();
    };
    stage.addEventListener('touchend', finish);
    stage.addEventListener('touchcancel', finish);
  }

  function watchScroll() {
    let ticking = false;
    stage.addEventListener('scroll', () => {
      if (ticking) return;
      ticking = true;
      requestAnimationFrame(() => {
        ticking = false;
        const num = currentPage();
        if (num !== visible) {
          visible = num;
          if (pageInput) pageInput.value = String(num);
          root.dispatchEvent(new CustomEvent('reader:page', { detail: { page: num } }));
        }
      });
    });

    if (window.ResizeObserver) {
      let timer = null;
      new ResizeObserver(() => {
        if (!fitWidth) return;
        clearTimeout(timer);
        timer = setTimeout(() => {
          // 容器变宽变窄都要重排：占位尺寸改了，已渲染的画布也得重画
          clearRendered();
          applySize();
          renderVisible();
        }, 160);
      }).observe(stage);
    }
  }

  async function load() {
    setStatus('正在加载…');
    try {
      pdf = await pdfjsLib.getDocument({ url }).promise;
      pageCount = pdf.numPages;
      if (totalLabel) totalLabel.textContent = String(pageCount);

      // 先用第一页的尺寸铺全部占位。逐页去取真实尺寸在上百页的 PDF 上
      // 要等一串 await，而绝大多数论文的页面尺寸是统一的，不值得。
      const first = await pdf.getPage(1);
      const base = first.getViewport({ scale: 1 });
      baseWidth = base.width;
      const ratio = base.width / base.height;

      for (let num = 1; num <= pageCount; num += 1) {
        const wrap = document.createElement('div');
        wrap.className = 'reader-page';
        wrap.dataset.page = String(num);
        const canvas = document.createElement('canvas');
        wrap.appendChild(canvas);
        stage.appendChild(wrap);
        pages.push({ num, wrap, canvas, ratio, rendered: false, rendering: false });
      }

      applySize();

      // 提前一屏渲染，快速滚动时不会看到空白页
      observer = new IntersectionObserver(
        (entries) => {
          entries.forEach((item) => {
            if (!item.isIntersecting) return;
            const entry = pages[Number(item.target.dataset.page) - 1];
            if (entry) renderPage(entry);
          });
        },
        { root: stage, rootMargin: PRELOAD_MARGIN },
      );
      pages.forEach((entry) => observer.observe(entry.wrap));

      bindToolbar();
      bindPinch();
      watchScroll();
      setStatus('');
      renderPage(pages[0]);
    } catch (err) {
      setStatus(`PDF 加载失败：${err && err.message ? err.message : err}`);
    }
  }

  load();

  return {
    goTo,
    get page() { return visible; },
    get pageCount() { return pageCount; },
  };
}
