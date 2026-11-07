"""词云 / 知识图谱离线数据与网页：打码在生成阶段完成（被隐藏的名称根本不在发给浏览器的文件里）、可复现、无定义原文；
页面由服务提供；前端不依赖外网 CDN；入口链接与词云交互的结构。全部离线，不读真实 xlsx（用现造的小表）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import gen_kg_assets as g  # noqa: E402

WEB = ROOT / "rag" / "web"
DEF = "这是一段很长的定义原文，绝不能出现在发给浏览器的任何文件里：数据要素是指根据特定生产需求形成的数据资源。"


def rows():
    """3 章：第 1 章 10 个概念（主概念 A 关联最多），第 2 章 4 个，第 3 章 1 个孤立概念。"""
    out = []

    def add(ch, s, rel, o, axiom=""):
        out.append({"集合ID": f"K{len(out)}", "章节": f"Ch{ch:02d}", "主体": s, "关系": rel, "客体": o, "公理": axiom})
    c1 = [f"概念甲{n}" for n in "一二三四五六七八九十"]
    for i, c in enumerate(c1[1:], 1):
        add(1, c1[0], "hasPart", c)                      # 甲一 是枢纽
        if i % 2 == 0:
            add(1, c, "relatedTo", c1[(i + 1) % 10])
    add(1, c1[0], "hasDefinition", DEF)                   # 定义：结构性关系，不计入、不输出
    add(1, c1[0], "hasSourceSpan", "P0001-P0009")
    c2 = [f"概念乙{n}" for n in "一二三四"]
    for c in c2[1:]:
        add(2, c2[0], "hasFeature", c)
    add(2, "第二章 小标题", "hasPart", c2[0])             # 章节标题类：不当概念
    add(3, "孤立概念", "instanceOf", "核心概念")
    for c in c1 + c2:                                     # 保证每个概念都作为主体出现
        add(1 if c in c1 else 2, c, "hasDefinition", DEF)
    return _prep(out)


def _prep(rs):
    for r in rs:
        r["主体"], r["客体"], r["关系"] = g.clean_name(r["主体"]), g.clean_name(r["客体"]), r["关系"]
        r["_ch"] = g.chapter_no(r["章节"])
    return rs


@pytest.fixture(scope="module")
def built():
    return g.build(rows(), n_words=8, reveal=0.3)


def test_each_chapter_reveals_about_thirty_percent_of_names_the_most_connected_ones(built):
    _, graph, rep = built
    by = {}
    for n in graph["nodes"]:
        by.setdefault(n["c"], []).append(n)
    named1 = [n for n in by[1] if "n" in n]
    assert len(by[1]) == 10 and len(named1) == 3                          # 10 × 30% = 3
    assert {n["n"] for n in named1} >= {"概念甲一"} and min(n["d"] for n in named1) >= sorted((n["d"] for n in by[1]), reverse=True)[2]
    named2 = [n for n in by[2] if "n" in n]
    assert len(named2) == 1 and named2[0]["d"] == max(n["d"] for n in by[2])      # 第 2 章 4 个概念 → 公开 1 个，且是关联最多的
    assert all(any("n" in n for n in v) for v in by.values())               # 每章至少公开 1 个
    assert 0.25 <= graph["totals"]["shown_ratio"] <= 0.45 and rep["ratio"] == graph["totals"]["shown_ratio"]


def test_hidden_nodes_have_no_name_field_at_all_and_hidden_names_are_not_in_the_serialised_file(built):
    _, graph, _ = built
    raw = json.dumps(graph, ensure_ascii=False)
    hidden = [f"概念甲{n}" for n in "一二三四五六七八九十"] + [f"概念乙{n}" for n in "一二三四"]
    named = {n["n"] for n in graph["nodes"] if "n" in n}
    for h in hidden:
        if h not in named:
            assert h not in raw, h                                            # 被打码的名称不在文件里（不是靠前端 CSS 遮挡）
    assert all(set(n) <= {"i", "c", "d", "n"} for n in graph["nodes"])
    assert sum(1 for n in graph["nodes"] if "n" not in n) == len(graph["nodes"]) - len(named) > 0


def test_no_definition_or_source_text_and_edges_carry_only_a_relation_type(built):
    cloud, graph, _ = built
    for blob in (json.dumps(graph, ensure_ascii=False), json.dumps(cloud, ensure_ascii=False)):
        assert "定义原文" not in blob and "P0001" not in blob and "数据要素是指" not in blob
    assert all(len(e) == 3 and isinstance(e[0], int) and isinstance(e[1], int) and e[2].isascii() for e in graph["edges"])
    assert {e[2] for e in graph["edges"]} <= {"hasPart", "relatedTo", "hasFeature"}          # 结构性关系（定义 / 出处）不输出


def test_structural_relations_and_chapter_headings_do_not_count_as_connections_or_concepts(built):
    _, graph, _ = built
    names = [n.get("n", "") for n in graph["nodes"]]
    assert "第二章小标题" not in names and len(graph["nodes"]) == 15           # 10 + 4 + 孤立概念；“第二章 小标题”不是概念
    iso = [n for n in graph["nodes"] if n["c"] == 3]
    assert len(iso) == 1 and iso[0]["d"] == 0                                 # instanceOf 是结构性关系，不计关联数
    hub = next(n for n in graph["nodes"] if n.get("n") == "概念甲一")
    assert hub["d"] == 9 + 0 and all(n["d"] <= hub["d"] for n in graph["nodes"])    # 9 条 hasPart；hasDefinition / hasSourceSpan 不计


def test_wordcloud_has_only_word_and_tier_in_one_to_five_and_is_sorted_by_importance(built):
    cloud, _, _ = built
    assert 1 <= len(cloud["words"]) <= 8 and all(set(w) == {"w", "s"} and 1 <= w["s"] <= 5 for w in cloud["words"])
    tiers = [w["s"] for w in cloud["words"]]
    assert tiers == sorted(tiers, reverse=True) and cloud["words"][0]["w"] == "概念甲一"


def test_generation_is_reproducible():
    a, b = g.build(rows(), 8, 0.3), g.build(rows(), 8, 0.3)
    assert json.dumps(a[:2], ensure_ascii=False) == json.dumps(b[:2], ensure_ascii=False)


def test_non_concept_names_are_filtered():
    for bad in ("第四章数据政策", "《促进和规范数据跨境流动规定》", "淘宝公司诉美景公司不正当竞争案", "小结", "（一）技术迭代快", "规则落地：制度框架的操作化", "效率性模块", "数据", "我"):
        assert not g.is_concept_name(bad), bad
    assert g.is_concept_name("数据产权登记") and g.is_concept_name("效能性")


# ---------------------------------------------------------------- 已提交的真实产物

def test_committed_assets_are_masked_and_in_range():
    cloud = json.loads((WEB / "data" / "wordcloud.json").read_text(encoding="utf-8"))
    graph = json.loads((WEB / "data" / "graph.json").read_text(encoding="utf-8"))
    assert 60 <= len(cloud["words"]) <= 80 and all(set(w) == {"w", "s"} for w in cloud["words"]) and {w["s"] for w in cloud["words"]} == {1, 2, 3, 4, 5}
    t = graph["totals"]
    assert t["chapters"] == 11 and t["concepts_total"] >= 600 and t["concepts"] == len(graph["nodes"]) and t["relations"] == len(graph["edges"])
    assert t["per_chapter_cap"] == g.GRAPH_PER_CHAPTER == 70
    named = sum(1 for n in graph["nodes"] if "n" in n)
    assert named == t["shown_names"] and 0.28 <= named / t["concepts"] <= 0.32
    assert all(set(n) <= {"i", "c", "d", "n"} for n in graph["nodes"]) and max(len(n["n"]) for n in graph["nodes"] if "n" in n) <= 14
    assert (WEB / "data" / "graph.json").stat().st_size < 120_000                # 浏览器流畅渲染的量级


def test_committed_assets_contain_no_long_text_that_could_be_a_definition():
    graph = json.loads((WEB / "data" / "graph.json").read_text(encoding="utf-8"))
    strings = [n["n"] for n in graph["nodes"] if "n" in n] + list(graph["chapters"].values()) + [e[2] for e in graph["edges"]]
    assert max(len(s) for s in strings) <= 20 and not any("，" in s or "。" in s for s in strings)


# ---------------------------------------------------------------- 页面与服务

def test_pages_and_data_are_served_by_the_app(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from rag import serve
    import inspect
    src = inspect.getsource(serve.build_app)
    assert 'app.mount("/static", StaticFiles(directory=str(web_dir))' in src            # /static 指向 rag/web：graph.html、vendor/、data/ 都在其下
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles
    app = FastAPI()
    app.mount("/static", StaticFiles(directory=str(WEB)), name="static")
    c = TestClient(app)
    for path, ctype in (("graph.html", "text/html"), ("graph.js", "javascript"), ("vendor/echarts.min.js", "javascript"),
                        ("data/graph.json", "json"), ("data/wordcloud.json", "json"), ("vendor/LICENSE-echarts.txt", "text")):
        r = c.get("/static/" + path)
        assert r.status_code == 200 and ctype in r.headers["content-type"], path


def test_no_page_loads_anything_from_the_internet():
    import re
    for f in ("index.html", "graph.html", "graph.js", "app.js", "style.css"):
        text = (WEB / f).read_text(encoding="utf-8")
        assert not re.search(r"(?:src|href)=[\"']https?://|@import\s+url\(\s*[\"']?https?://|fetch\(\s*[\"']https?://", text), f


def test_vendor_library_is_apache_licensed_and_ships_its_license():
    assert "Apache License" in (WEB / "vendor" / "LICENSE-echarts.txt").read_text(encoding="utf-8")[:400]
    assert "echarts" in (WEB / "vendor" / "NOTICE-echarts.txt").read_text(encoding="utf-8").lower()
    assert (WEB / "vendor" / "echarts.min.js").stat().st_size > 500_000


def test_home_has_the_graph_link_in_the_header_and_the_wordcloud_inside_the_welcome_area():
    html = (WEB / "index.html").read_text(encoding="utf-8")
    sub = html[html.index('class="subtitle"'):html.index("</p>", html.index('class="subtitle"'))]
    assert "数字治理科研团队 · 学术数字人" in sub and 'href="/static/graph.html"' in sub and 'target="_blank"' in sub and "查看知识图谱（部分）" in sub
    assert sub.index("学术数字人") < sub.index("查看知识图谱")                                  # 链接在副标题“后面”
    w = html[html.index('id="welcome"'):html.index("</main>")]
    assert w.index('class="suggestions"') < w.index('id="wordcloud"') < w.index('class="hint"')   # 在介绍文字和示例按钮下面
    assert 'id="wordcloud"' not in html[:html.index('id="welcome"')]


def test_wordcloud_script_fills_the_input_but_never_sends_and_hides_with_the_welcome_area():
    js = (WEB / "app.js").read_text(encoding="utf-8")
    seg = js[js.index("async function loadWordcloud"):js.index("/* ---- 探活")]
    assert "input.value = `什么是${w.w}？`" in seg and "input.focus()" in seg and "send(" not in seg and "sendBtn" not in seg
    assert "/static/data/wordcloud.json" in seg and "catch" in seg
    assert "loadWordcloud();" in js and "w.style.display = 'none'" in js                     # hideWelcome 隐藏整个欢迎区，词云在其内


def test_wordcloud_style_is_tech_styled_floats_slowly_supports_dark_and_reduced_motion_and_no_horizontal_scroll():
    css = (WEB / "style.css").read_text(encoding="utf-8")
    seg = css[css.index("/* ---- 首页词云"):]
    assert "linear-gradient(120deg" in seg and "drop-shadow" in seg and "@keyframes wc-float" in seg and "animation: wc-float var(--t)" in seg
    assert "@media (prefers-reduced-motion: reduce) { .wc { animation: none" in seg
    assert "prefers-color-scheme: dark" in seg and ".wordcloud:hover .wc" in seg and "scale(1.22)" in seg
    assert "overflow: visible" in seg and "overflow: hidden" not in seg[:seg.index("@keyframes")] and "max-height" not in seg[:seg.index("@keyframes")]   # 容器随内容长高：词不被裁切，也不压住下面的说明文字
    assert "@media (max-width: 520px)" in seg and "linear-gradient(rgba(99, 102, 241" in seg and "26px 26px" in seg                          # 网格背景（无外部资源）
    for t in range(1, 6):
        assert f".wc-{t} {{" in seg
    import re
    sizes = [float(x) for x in re.findall(r"\.wc-[1-5] \{ font-size: ([\d.]+)px", seg[:seg.index("@keyframes")])]
    assert len(sizes) == 5 and sizes == sorted(sizes) and max(sizes) <= 20                  # 最大档不压过 20px 的标题


def test_graph_page_has_the_required_markers_stats_filters_and_interactions():
    html = (WEB / "graph.html").read_text(encoding="utf-8")
    js = (WEB / "graph.js").read_text(encoding="utf-8")
    assert "部分展示 · 完整知识图谱受保护" in html and "/static/vendor/echarts.min.js" in html and 'id="stats"' in html and 'id="filters"' in html
    assert "layout: 'force'" in js and "roam: true" in js and "draggable: true" in js and "focus: 'adjacency'" in js     # 力导向 / 缩放拖拽 / 邻居高亮
    assert "g.totals.chapters" in js and "g.totals.concepts" in js and "g.totals.relations" in js and "shown_names" in js    # 统计栏
    assert "名称受保护" in js and "prefers-reduced-motion" in js
    info = js[js.index("chart.on('click'"):js.index("$('info-x')")]
    assert "n.n" in info and "n.c" in info and "n.d" in info and "定义" not in info and "def" not in info.lower()        # 点击卡片：名称 / 章节 / 关联数


def test_structural_relation_between_two_concepts_is_not_drawn_and_not_counted():
    rs = _prep([
        {"集合ID": "1", "章节": "Ch01", "主体": "概念丙一", "关系": "hasPart", "客体": "概念丙二", "公理": ""},
        {"集合ID": "2", "章节": "Ch01", "主体": "概念丙一", "关系": "hasSubsection", "客体": "概念丙三", "公理": ""},
        {"集合ID": "3", "章节": "Ch01", "主体": "概念丙一", "关系": "hasSourceSpan", "客体": "概念丙二", "公理": ""},
        {"集合ID": "4", "章节": "Ch01", "主体": "概念丙三", "关系": "hasFeature", "客体": "概念丙二", "公理": ""},
        {"集合ID": "5", "章节": "Ch01", "主体": "概念丙二", "关系": "hasFeature", "客体": "概念丙二", "公理": ""},      # 自环
    ])
    _, graph, _ = g.build(rs, n_words=3, reveal=1.0)
    by = {n["n"]: n for n in graph["nodes"]}
    assert {e[2] for e in graph["edges"]} == {"hasPart", "hasFeature"} and len(graph["edges"]) == 2
    assert by["概念丙一"]["d"] == 1 and by["概念丙二"]["d"] == 2 and by["概念丙三"]["d"] == 1          # 结构性关系与自环都不计


# ---------------------------------------------------------------- 每章节点上限：写进生成脚本，可复现

def _big_rows():
    """第 1 章 300 个概念，第 2、3 章各 8 个：不设上限时第 1 章会占满整张图。"""
    out = []
    for ch, n in ((1, 300), (2, 8), (3, 8)):
        hub = f"枢纽{ch}章概念"
        for i in range(n):
            out.append({"集合ID": f"{ch}-{i}", "章节": f"Ch{ch:02d}", "主体": hub, "关系": "hasPart", "客体": f"概念{ch}章{i:03d}号", "公理": ""})
            out.append({"集合ID": f"{ch}-s{i}", "章节": f"Ch{ch:02d}", "主体": f"概念{ch}章{i:03d}号", "关系": "hasFeature", "客体": f"特征{ch}章{i:03d}", "公理": ""})
    return _prep(out)


def test_per_chapter_cap_keeps_the_most_connected_and_every_chapter_stays_visible():
    _, graph, rep = g.build(_big_rows(), n_words=10, reveal=0.3, per_chapter=40)
    from collections import Counter
    per = Counter(n["c"] for n in graph["nodes"])
    assert per[1] == 40 and per[2] <= 40 and per[3] <= 40 and set(per) == {1, 2, 3}
    assert graph["totals"]["concepts"] == len(graph["nodes"]) == sum(per.values()) < graph["totals"]["concepts_total"]
    assert per[1] == 40 and per[1] < 300 and per[1] / sum(per.values()) < 300 / 316 - 0.1       # 第 1 章从 300 个压到 40 个，占比明显下降（不设上限时占 95%）
    hub = [n for n in graph["nodes"] if n.get("n") == "枢纽1章概念"]
    assert len(hub) == 1 and hub[0]["d"] == max(n["d"] for n in graph["nodes"] if n["c"] == 1)      # 保留的是关联度最高的那些


def test_capped_graph_keeps_about_thirty_percent_names_per_chapter_and_edges_stay_inside_the_kept_nodes():
    _, graph, _ = g.build(_big_rows(), n_words=10, reveal=0.3, per_chapter=40)
    from collections import Counter
    per, named = Counter(n["c"] for n in graph["nodes"]), Counter(n["c"] for n in graph["nodes"] if "n" in n)
    assert named[1] == 12 and all(abs(named[c] - round(per[c] * 0.3)) <= 1 for c in per)
    ids = {n["i"] for n in graph["nodes"]}
    assert ids == set(range(len(ids))) and all(e[0] in ids and e[1] in ids for e in graph["edges"])      # 编号连续、连线两端都在保留的节点里
    assert 0.25 <= graph["totals"]["shown_ratio"] <= 0.35


def test_node_degree_shown_to_users_is_the_real_relevance_count_not_the_count_inside_the_capped_graph():
    _, graph, _ = g.build(_big_rows(), n_words=10, reveal=1.0, per_chapter=40)
    hub = next(n for n in graph["nodes"] if n.get("n") == "枢纽1章概念")
    drawn = sum(1 for e in graph["edges"] if hub["i"] in (e[0], e[1]))
    assert hub["d"] == 300 and drawn == 39 and drawn < hub["d"]          # 点击卡片里的“关联数”是全图真实数；图里只画保留下来的连线


def test_cap_is_reproducible_and_the_committed_json_is_exactly_what_the_script_generates_from_the_same_rows():
    a, b = g.build(_big_rows(), 10, 0.3, 40), g.build(_big_rows(), 10, 0.3, 40)
    assert json.dumps(a[1], ensure_ascii=False) == json.dumps(b[1], ensure_ascii=False)
    src = Path("/Users/yizhimianyang/Desktop/NJU_ra/11.2update/三层集合/纯粹知识集合.xlsx")
    if not src.is_file():
        pytest.skip("源 xlsx 不在本机（只在开发机上校验提交的 JSON 与脚本输出一致）")
    cloud, graph, _ = g.build(g.load(src), 72, 0.3)
    committed = json.loads((WEB / "data" / "graph.json").read_text(encoding="utf-8"))
    assert graph == committed and cloud == json.loads((WEB / "data" / "wordcloud.json").read_text(encoding="utf-8"))


def test_committed_graph_has_a_visible_share_for_every_chapter():
    graph = json.loads((WEB / "data" / "graph.json").read_text(encoding="utf-8"))
    from collections import Counter
    per = Counter(n["c"] for n in graph["nodes"])
    assert set(per) == set(range(1, 12)) and max(per.values()) <= 70 and max(per.values()) / len(graph["nodes"]) <= 0.25
    assert graph["totals"]["concepts_total"] > graph["totals"]["concepts"]


# ---------------------------------------------------------------- 网页打磨：词云不压字 / 图谱适配视口 / 窄屏顶栏

def test_wordcloud_container_grows_with_its_content_and_has_clear_padding_and_spacing_below():
    css = (WEB / "style.css").read_text(encoding="utf-8")
    block = css[css.index(".wordcloud {"):css.index(".wordcloud[hidden]")]
    assert "overflow: visible" in block and "max-height" not in block and "padding: 22px 28px" in block and "margin: 18px auto 16px" in block
    assert "isolation: isolate" in block and "radial-gradient" in block and "linear-gradient(rgba(99, 102, 241" in block      # 光晕 + 网格背景


def test_wordcloud_colour_layers_by_tier_in_light_and_dark():
    css = (WEB / "style.css").read_text(encoding="utf-8")
    for t in range(1, 6):
        assert css.count(f".wc-{t} {{") >= 2                                              # 浅色一套、深色一套
    light = css[css.index(".wc-1 { font-size"):css.index("/* 颜色按层级加深")]
    assert "--wc-a: #4c1d95" in light and "--wc-a: #6366f1" in light                      # 重要的词颜色更深


def test_graph_fits_the_whole_graph_into_the_viewport_after_layout():
    js = (WEB / "graph.js").read_text(encoding="utf-8")
    seg = js[js.index("function fit()"):js.index("chart.on('finished'")]
    assert "graphRoam" in seg and "dataToPoint" in seg and "0.03" in seg and "0.97" in seg and "originX" in seg and "dx:" in seg
    assert "chart.on('finished'" in js and "setTimeout(fit" in js and "state._refit = true" in js                     # 初次加载与切换章节后都重新适配
    assert "concepts_total" in js and "图中示意" in js                                                                  # 统计栏仍显示总概念数


def test_narrow_screen_topbar_never_wraps_the_button_or_stacks_the_status_text():
    css = (WEB / "style.css").read_text(encoding="utf-8")
    tail = css[css.index("/* ---- 窄屏顶栏"):]
    assert ".new-chat, .status, #status-text { white-space: nowrap; }" in tail and "flex: none" in tail
    assert "@media (max-width: 640px)" in tail and "flex-wrap: nowrap" in tail and "#status-text { display: none; }" in tail
    js = (WEB / "app.js").read_text(encoding="utf-8")
    assert "st.title = text" in js                                                       # 只剩圆点时，悬停 / 长按仍能看到状态文字
