/* 知识图谱（部分）：只使用 /static/data/graph.json——名称在离线生成时已打码，被隐藏的节点在这份数据里根本没有名称。
   力导向 + 簇着色（按章）+ 悬停高亮邻居 + 点击已公开节点看名称 / 章节 / 关联数 + 章节筛选 + 加载生长动画 + 连线流光。 */
(function () {
  'use strict';
  const COLORS = ['#60a5fa', '#a78bfa', '#34d399', '#f472b6', '#fbbf24', '#22d3ee', '#fb923c', '#a3e635', '#f87171', '#818cf8', '#2dd4bf'];
  const reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const $ = (id) => document.getElementById(id);
  const msg = $('msg'), statsEl = $('stats'), filtersEl = $('filters'), info = $('info');

  function fail(t) { msg.textContent = t; msg.style.display = 'flex'; }
  if (!window.echarts) { fail('图表组件加载失败'); return; }

  fetch('/static/data/graph.json').then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); }).then(init).catch(() => fail('知识图谱数据加载失败'));

  function init(g) {
    const chart = echarts.init($('chart'), null, { renderer: 'canvas' });
    const chapters = g.chapters;                       // {"1": "数据要素", ...}
    const chKeys = Object.keys(chapters).map(Number).sort((a, b) => a - b);
    const colorOf = (c) => COLORS[(c - 1) % COLORS.length];
    const maxDeg = Math.max(1, ...g.nodes.map((n) => n.d));
    let viewMax = maxDeg;                              // 当前视图内的最大关联数：筛选单章时标签阈值按本章算
    const named = (n) => typeof n.n === 'string';
    const state = { chapter: 0 };                       // 0 = 全部

    function sizeOf(n) { return (named(n) ? 9 : 3.2) + Math.sqrt(n.d / maxDeg) * (named(n) ? 17 : 6); }

    function build() {
      const keep = new Set(g.nodes.filter((n) => !state.chapter || n.c === state.chapter).map((n) => n.i));
      viewMax = Math.max(1, ...g.nodes.filter((n) => keep.has(n.i)).map((n) => n.d));
      const nodes = g.nodes.filter((n) => keep.has(n.i)).map((n) => ({
        id: String(n.i), value: n.d, category: n.c - 1, symbolSize: sizeOf(n), _n: n,
        name: named(n) ? n.n : '',                        // 无名称的节点：只是光点
        label: { show: named(n) && (state.chapter ? true : n.d >= viewMax * 0.12), formatter: named(n) ? n.n : '', color: '#e2e8f0', fontSize: 11, textBorderColor: '#070b18', textBorderWidth: 3 },
        itemStyle: { color: colorOf(n.c), opacity: named(n) ? 1 : 0.55, shadowBlur: named(n) ? 16 : 7, shadowColor: colorOf(n.c) },
      }));
      const links = g.edges.filter((e) => keep.has(e[0]) && keep.has(e[1])).map((e) => ({ source: String(e[0]), target: String(e[1]), rel: e[2] }));
      return { nodes, links };
    }

    function option(first) {
      const { nodes, links } = build();
      const pct = Math.round((g.totals.shown_names / g.totals.concepts) * 100);
      const total = g.totals.concepts_total || g.totals.concepts;
      statsEl.innerHTML = `<span>共 <b>${g.totals.chapters}</b> 章 · <b>${total}</b> 个概念（图中示意 <b>${g.totals.concepts}</b> 个） · <b>${g.totals.relations}</b> 条关系 · 当前展示名称 <b>${g.totals.shown_names}</b> 个（<b>${pct}%</b>）`
        + (state.chapter ? ` · 筛选：第${state.chapter}章 ${chapters[state.chapter]}（${nodes.length} 个节点）` : '') + '</span>';
      return {
        backgroundColor: 'transparent',
        animation: !reduce, animationDuration: first ? 1800 : 600, animationEasing: 'cubicOut', animationDurationUpdate: 500,
        tooltip: { show: false },
        legend: { show: false },
        series: [{
          type: 'graph', layout: 'force', roam: true, draggable: true, nodes, links,
          categories: chKeys.map((c) => ({ name: chapters[c], itemStyle: { color: colorOf(c) } })),
          scaleLimit: { min: 0.3, max: 6 },
          force: { repulsion: 110, gravity: 0.12, edgeLength: [22, 80], friction: 0.2, layoutAnimation: !reduce },
          edgeSymbol: ['none', 'none'],
          lineStyle: { color: 'source', width: 0.7, opacity: 0.32, curveness: 0.08 },
          emphasis: { focus: 'adjacency', scale: 1.35, lineStyle: { width: 1.8, opacity: 0.9 }, label: { show: true } },
          blur: { itemStyle: { opacity: 0.08 }, lineStyle: { opacity: 0.04 }, label: { opacity: 0.1 } },
          select: { disabled: true },
        }],
      };
    }

    function render(first) {
      chart.dispatchAction({ type: 'downplay' });          // 先清掉悬停产生的“淡出”状态，否则新数据会整体变暗
      $('tip').style.display = 'none';
      chart.setOption(option(first), { notMerge: true });
    }
    render(true);
    msg.style.display = 'none';

    /* 适配视口：力导向布局会把离群节点推得很远。布局稳定后读所有节点的坐标算包围盒，
       用 graphRoam 的 zoom 与 dx/dy 一次性把整张图缩放并平移进“避开顶部统计栏与底部筛选”的可视区。 */
    let fitted = 0;
    function fit() {
      try {
        const sm = chart.getModel().getSeriesByIndex(0);
        const data = sm.getData(), cs = sm.coordinateSystem;
        const W = chart.getWidth(), H = chart.getHeight();
        // 节点现在的屏幕坐标：cs.dataToPoint 已经带上当前的缩放 / 平移
        const px = [], py = [];
        for (let i = 0; i < data.count(); i++) {
          const l = data.getItemLayout(i);
          const p = l && cs.dataToPoint(l);
          if (p && isFinite(p[0]) && isFinite(p[1])) { px.push(p[0]); py.push(p[1]); }
        }
        if (px.length < 2) return;
        px.sort((a, b) => a - b); py.sort((a, b) => a - b);
        const q = (a, f) => a[Math.min(a.length - 1, Math.max(0, Math.round((a.length - 1) * f)))];
        const x0 = q(px, 0.03), x1 = q(px, 0.97), y0 = q(py, 0.03), y1 = q(py, 0.97);       // 分位数而非极值：少数离群光点不该把整张图缩成一团
        const top = 96, bottom = 104, side = 28;
        const k = Math.min((W - 2 * side) / Math.max(x1 - x0, 1), (H - top - bottom) / Math.max(y1 - y0, 1));
        const z = Math.max(0.3 / cs.getZoom(), Math.min(4 / cs.getZoom(), k));                  // graphRoam 的 zoom 是相对当前缩放的倍数
        const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
        const tx = W / 2, ty = top + (H - top - bottom) / 2;
        chart.dispatchAction({ type: 'graphRoam', seriesIndex: 0, zoom: z, originX: cx, originY: cy });   // 以内容中心为原点缩放，中心位置不动
        chart.dispatchAction({ type: 'graphRoam', seriesIndex: 0, dx: tx - cx, dy: ty - cy });             // 再把内容中心平移到可视区中心
        fitted += 1;
      } catch (e) { /* 适配失败不影响使用 */ }
    }
    chart.on('finished', function once() { if (fitted === 0 || state._refit) { state._refit = false; setTimeout(fit, 60); } });
    setTimeout(fit, reduce ? 300 : 2600);               // 兜底：动画结束后再适配一次


    /* 连线上的流动光点：在 canvas 上层叠一层轻量绘制太重，这里用 ECharts 的 lines 系列 effect 复刻不方便（graph 不支持），
       改为让连线随时间做“呼吸”式明暗流动：定时轻微改变 lineStyle.opacity / dash 偏移，开销很小。 */
    let phase = 0, timer = null;
    function startFlow() {
      if (reduce || timer) return;
      timer = setInterval(() => {
        phase = (phase + 1) % 60;
        chart.setOption({ series: [{ lineStyle: { type: [3, 9], dashOffset: -phase * 0.2 * 12 % 12, opacity: 0.34 } }] }, false);
      }, 90);
    }
    setTimeout(startFlow, 2200);
    document.addEventListener('visibilitychange', () => { if (document.hidden && timer) { clearInterval(timer); timer = null; } else if (!document.hidden) startFlow(); });

    /* 点击：只有已公开名称的节点才有信息卡；信息卡里没有定义 */
    chart.on('click', (p) => {
      if (p.dataType !== 'node') { info.classList.remove('show'); return; }
      const n = p.data._n;
      if (!named(n)) { info.classList.remove('show'); return; }
      $('i-name').textContent = n.n;
      $('i-ch').textContent = `第${n.c}章 ${chapters[n.c] || ''}`;
      $('i-deg').textContent = String(n.d);
      info.classList.add('show');
    });
    $('info-x').addEventListener('click', () => info.classList.remove('show'));
    chart.getZr().on('click', (e) => { if (!e.target) info.classList.remove('show'); });

    /* 悬停提示：已公开的节点显示名称；光点只提示“名称受保护”；连线只提示关系类型 */
    const tip = $('tip');
    chart.on('mouseover', (p) => {
      let t = '';
      if (p.dataType === 'node') t = named(p.data._n) ? p.data._n.n : '名称受保护';
      else if (p.dataType === 'edge') t = '关系：' + p.data.rel;
      if (!t) return;
      tip.textContent = t; tip.style.display = 'block';
    });
    chart.on('mouseout', () => { tip.style.display = 'none'; });
    window.addEventListener('mousemove', (e) => { tip.style.left = e.clientX + 12 + 'px'; tip.style.top = e.clientY + 12 + 'px'; });

    /* 章节筛选 */
    const mk = (label, key, color) => {
      const b = document.createElement('button');
      b.className = 'f' + (key === 0 ? ' all' : '') + (state.chapter === key ? ' on' : '');
      b.style.setProperty('--c', color || '#818cf8');
      b.innerHTML = (key ? '<i></i>' : '') + label;
      b.addEventListener('click', () => {
        state.chapter = key;
        filtersEl.querySelectorAll('.f').forEach((x) => x.classList.remove('on'));
        b.classList.add('on');
        info.classList.remove('show');
        render(false);
        state._refit = true;
        setTimeout(fit, reduce ? 300 : 1200);
      });
      filtersEl.appendChild(b);
    };
    mk('全部章节', 0);
    chKeys.forEach((c) => mk(`第${c}章 ${chapters[c]}`, c, colorOf(c)));
    filtersEl.querySelector('.f.all').classList.add('on');

    window.addEventListener('resize', () => chart.resize());
  }
})();
