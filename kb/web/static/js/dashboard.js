/* 概览页的分类汇总图表。
 *
 * 数据由服务端内联成 JSON（见模板里的 #kb-stats），**不再发一次请求**。
 * 这一页的全部意义是「一眼看到构成」，如果图表要等一个 XHR 回来才出现，
 * 首屏就会先闪一排空框——比慢一点更难受。
 *
 * 主题色从 CSS 变量里读，而不是在 JS 里再写一份色板：
 * 暗色模式下图表用的还是亮色配色，是这类页面最典型的破绽。
 */

function palette() {
  const css = getComputedStyle(document.documentElement);
  const read = (name, fallback) => (css.getPropertyValue(name) || '').trim() || fallback;
  return {
    text: read('--text', '#1f2937'),
    muted: read('--text-muted', '#6b7280'),
    faint: read('--text-faint', '#9ca3af'),
    border: read('--border', '#e5e7eb'),
    accent: read('--accent', '#4f46e5'),
    surface: read('--surface', '#ffffff'),
    // 分类配色：够区分即可，不追求花哨
    series: ['#4f46e5', '#0ea5e9', '#10b981', '#f59e0b', '#ef4444',
             '#8b5cf6', '#14b8a6', '#f97316', '#ec4899', '#64748b'],
  };
}

function baseOption(p) {
  return {
    color: p.series,
    textStyle: { color: p.text, fontSize: 12 },
    grid: { left: 8, right: 12, top: 24, bottom: 8, containLabel: true },
    tooltip: { trigger: 'item', confine: true },
  };
}

/** 横向条形：标签名通常较长，横着放不用把标签斜过来。 */
function barOption(data, p, { horizontal = false } = {}) {
  const labels = data.map((d) => d.label);
  const values = data.map((d) => d.value);
  const axis = {
    type: 'category',
    data: labels,
    axisLine: { lineStyle: { color: p.border } },
    axisLabel: { color: p.muted, fontSize: 11 },
    axisTick: { show: false },
  };
  const valueAxis = {
    type: 'value',
    splitLine: { lineStyle: { color: p.border, type: 'dashed' } },
    axisLabel: { color: p.muted, fontSize: 11 },
  };
  return {
    ...baseOption(p),
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, confine: true },
    xAxis: horizontal ? valueAxis : axis,
    yAxis: horizontal ? { ...axis, inverse: true } : valueAxis,
    series: [{
      type: 'bar',
      data: values,
      barMaxWidth: 22,
      itemStyle: { borderRadius: horizontal ? [0, 3, 3, 0] : [3, 3, 0, 0] },
    }],
  };
}

function donutOption(data, p) {
  return {
    ...baseOption(p),
    tooltip: { trigger: 'item', formatter: '{b}: {c} ({d}%)', confine: true },
    legend: {
      bottom: 0,
      textStyle: { color: p.muted, fontSize: 11 },
      itemWidth: 10,
      itemHeight: 10,
    },
    series: [{
      type: 'pie',
      radius: ['46%', '68%'],
      center: ['50%', '44%'],
      avoidLabelOverlap: true,
      itemStyle: { borderColor: p.surface, borderWidth: 2 },
      label: { show: false },
      data: data.map((d) => ({ name: d.label, value: d.value })),
    }],
  };
}

function init() {
  const holder = document.getElementById('kb-stats');
  if (!holder || typeof window.echarts === 'undefined') return;

  let data;
  try {
    data = JSON.parse(holder.textContent);
  } catch {
    return;
  }

  const p = palette();
  const charts = [];

  /** 空维度不画图——一张没有数据的空坐标系只会让人以为页面坏了。 */
  const mount = (id, option, isEmpty) => {
    const el = document.getElementById(id);
    if (!el) return;
    if (isEmpty) {
      el.innerHTML = '<div class="empty small">暂无数据</div>';
      return;
    }
    const chart = window.echarts.init(el, null, { renderer: 'canvas' });
    chart.setOption(option);
    charts.push(chart);
  };

  const nonZero = (rows) => rows.filter((r) => r.value > 0);

  mount('chart-year', barOption(data.years, p), !data.years.length);
  mount('chart-reading', donutOption(data.reading, p), !nonZero(data.reading).length);
  mount('chart-ingest', donutOption(data.ingest, p), !nonZero(data.ingest).length);
  mount('chart-notes', barOption(nonZero(data.note_kinds), p), !nonZero(data.note_kinds).length);
  mount('chart-chunks', barOption(data.chunks, p), !data.chunks.length);

  const code = [
    { label: '有开源代码', value: data.code.with_code },
    { label: '无代码', value: data.code.without_code },
  ];
  mount('chart-code', donutOption(code, p), !nonZero(code).length);

  // 标签维度：把各维度的 Top N 拼成一张横向条形，维度名作为前缀
  const tagRows = [];
  Object.entries(data.tags).forEach(([dimension, items]) => {
    items.forEach((item) => {
      tagRows.push({ label: `${dimension} · ${item.label}`, value: item.value });
    });
  });
  tagRows.sort((a, b) => b.value - a.value);
  mount('chart-tags', barOption(tagRows.slice(0, 12), p, { horizontal: true }), !tagRows.length);

  // 容器尺寸变化时重排。ECharts 不会自己监听——侧栏折叠、窗口缩放都会让图变形。
  let timer = null;
  window.addEventListener('resize', () => {
    clearTimeout(timer);
    timer = setTimeout(() => charts.forEach((c) => c.resize()), 150);
  });

  // 主题切换：颜色是从 CSS 变量读的，换主题后必须重新 setOption 才生效
  window.addEventListener('kb:theme-changed', () => {
    const next = palette();
    charts.forEach((chart) => {
      chart.setOption({ color: next.series, textStyle: { color: next.text } });
    });
  });
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
