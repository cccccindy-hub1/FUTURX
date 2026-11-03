"""判例库 / 名言库建向量库的安全与正确性：mock 嵌入、假 Chroma 客户端；绝不碰论文库目录与集合。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag import kbbuild
from rag.kbbuild import UnsafeStore, build, case_rows, check_store_safe, quote_rows


class FakeCol:
    def __init__(self, name, store):
        self.name, self.store, self.rows = name, store, []

    def add(self, ids, documents, metadatas, embeddings):
        assert len(ids) == len(documents) == len(metadatas) == len(embeddings)
        self.rows += list(zip(ids, documents, metadatas, embeddings))

    def count(self):
        return len(self.rows)


class FakeClient:
    registry: dict = {}                    # (path, name) -> FakeCol，模拟磁盘上的状态
    deleted: list = []
    opened: list = []

    def __init__(self, path):
        self.path = str(path)
        FakeClient.opened.append(self.path)
        Path(self.path).mkdir(parents=True, exist_ok=True)
        (Path(self.path) / "chroma.sqlite3").touch()                      # 真实 PersistentClient 一打开就会在目录里建库文件

    def list_collections(self):
        return [type("C", (), {"name": n})() for (p, n) in FakeClient.registry if p == self.path]

    def get_collection(self, name):
        return FakeClient.registry[(self.path, name)]

    def create_collection(self, name, metadata=None):
        col = FakeCol(name, self.path)
        FakeClient.registry[(self.path, name)] = col
        return col

    def delete_collection(self, name):
        FakeClient.deleted.append((self.path, name))
        del FakeClient.registry[(self.path, name)]


@pytest.fixture(autouse=True)
def reset():
    FakeClient.registry, FakeClient.deleted, FakeClient.opened = {}, [], []


def embed(texts):
    return [[float(len(t)), 1.0] for t in texts]


def write_chunks(dir_, rows):
    dir_.mkdir(parents=True, exist_ok=True)
    (dir_ / "chunks.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return dir_


CASE_CHUNKS = [
    {"chunk_id": "case:01:card:aa", "case_id": "case:01", "seq": 1, "chapter": "第一章 数据要素", "chapter_no": 1, "title": "甲案", "kind": "card",
     "part": 0, "heading": "", "text": "【第一章 数据要素】甲案\n裁判要旨：…", "meta": {"case_number": "（2021）浙8601民初1号", "court": "某法院"}},
    {"chunk_id": "case:01:body01:bb", "case_id": "case:01", "seq": 1, "chapter": "第一章 数据要素", "chapter_no": 1, "title": "甲案", "kind": "body",
     "part": 1, "heading": "基本案情", "text": "【第一章 数据要素】甲案\n基本案情…", "meta": {}},
]
QUOTE_CHUNKS = [
    {"chunk_id": "quote:2018:01:aa", "speech_id": "quote:2018:01", "year": 2018, "n": 1, "title": "某会议上的讲话", "kind": "讲话",
     "text": "2018年　某会议上的讲话（讲话）\n[1] 第一段\n[2] 第二段", "paragraphs": ["第一段", "第二段"], "chars": 6, "removed_duplicates": 0},
]


# ---------------------------------------------------------------- 安全：绝不碰论文库

def test_refuses_the_thesis_store_directory_itself(tmp_path):
    thesis = tmp_path / "chroma_db"
    with pytest.raises(UnsafeStore):
        check_store_safe(thesis, "data_law_cases", thesis_store=thesis, thesis_collection="nlp_degree_thesis")


def test_refuses_a_directory_inside_or_containing_the_thesis_store(tmp_path):
    thesis = tmp_path / "chroma_db"
    for bad in (thesis / "case_store", tmp_path, tmp_path / "chroma_db" / "x" / "y"):
        with pytest.raises(UnsafeStore):
            check_store_safe(bad, "data_law_cases", thesis_store=thesis, thesis_collection="nlp_degree_thesis")


def test_refuses_the_thesis_collection_name(tmp_path):
    with pytest.raises(UnsafeStore):
        check_store_safe(tmp_path / "case_store", "nlp_degree_thesis", thesis_store=tmp_path / "chroma_db", thesis_collection="nlp_degree_thesis")


def test_sibling_directories_are_fine(tmp_path):
    check_store_safe(tmp_path / "case_store", "data_law_cases", thesis_store=tmp_path / "chroma_db", thesis_collection="nlp_degree_thesis")
    check_store_safe(tmp_path / "chroma_db_v2", "data_law_cases", thesis_store=tmp_path / "chroma_db", thesis_collection="nlp_degree_thesis")


def test_default_targets_are_separate_from_the_thesis_store():
    from rag.cases import config as cc
    from rag.quotes import config as qc
    check_store_safe(cc.CASE_STORE_DIR, cc.CASE_COLLECTION)
    check_store_safe(qc.QUOTE_STORE_DIR, qc.QUOTE_COLLECTION)


def test_build_never_opens_the_thesis_store_when_target_is_unsafe(tmp_path, monkeypatch):
    monkeypatch.setattr(kbbuild.base, "STORE_DIR", tmp_path / "chroma_db")
    cache = write_chunks(tmp_path / "cc", CASE_CHUNKS)
    with pytest.raises(UnsafeStore):
        build("cases", cache, tmp_path / "chroma_db", "data_law_cases", embed=embed, client_factory=FakeClient)
    assert FakeClient.opened == [] and FakeClient.registry == {}               # 连客户端都没有创建


# ---------------------------------------------------------------- 写入与覆盖保护

def test_builds_case_collection_with_embeddings_and_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(kbbuild.base, "STORE_DIR", tmp_path / "chroma_db")
    cache = write_chunks(tmp_path / "cc", CASE_CHUNKS)
    info = build("cases", cache, tmp_path / "case_store", "data_law_cases", embed=embed, client_factory=FakeClient)
    assert info["written"] == 2 and info["collection"] == "data_law_cases"
    col = FakeClient.registry[(str(tmp_path / "case_store"), "data_law_cases")]
    ids, docs, metas, embs = zip(*col.rows)
    assert ids == ("case:01:card:aa", "case:01:body01:bb") and all(len(e) == 2 for e in embs)
    assert metas[0]["case_id"] == "case:01" and metas[0]["chapter_no"] == 1 and metas[0]["kind"] == "card" and metas[0]["case_number"] == "（2021）浙8601民初1号"
    assert metas[1]["heading"] == "基本案情" and all(isinstance(v, (str, int)) for m in metas for v in m.values())      # Chroma 元数据只能是标量


def test_builds_quote_collection_keeping_original_paragraphs(tmp_path, monkeypatch):
    monkeypatch.setattr(kbbuild.base, "STORE_DIR", tmp_path / "chroma_db")
    cache = write_chunks(tmp_path / "qc", QUOTE_CHUNKS)
    build("quotes", cache, tmp_path / "quote_store", "xi_digital_quotes", embed=embed, client_factory=FakeClient)
    meta = FakeClient.registry[(str(tmp_path / "quote_store"), "xi_digital_quotes")].rows[0][2]
    assert meta["speech_id"] == "quote:2018:01" and meta["year"] == 2018 and json.loads(meta["paragraphs_json"]) == ["第一段", "第二段"]


def test_refuses_to_overwrite_a_nonempty_collection_without_rebuild(tmp_path, monkeypatch):
    monkeypatch.setattr(kbbuild.base, "STORE_DIR", tmp_path / "chroma_db")
    cache = write_chunks(tmp_path / "cc", CASE_CHUNKS)
    build("cases", cache, tmp_path / "case_store", "data_law_cases", embed=embed, client_factory=FakeClient)
    with pytest.raises(SystemExit, match="已存在且非空"):                          # 目录非空就拒绝（比集合非空更早、更严）
        build("cases", cache, tmp_path / "case_store", "data_law_cases", embed=embed, client_factory=FakeClient)
    assert FakeClient.deleted == []                                             # 没有删任何东西


def test_rebuild_deletes_only_its_own_collection(tmp_path, monkeypatch):
    monkeypatch.setattr(kbbuild.base, "STORE_DIR", tmp_path / "chroma_db")
    cache = write_chunks(tmp_path / "cc", CASE_CHUNKS)
    store = tmp_path / "case_store"
    other = FakeClient(store).create_collection("someone_else")                  # 同一目录里别人的集合
    other.add(["x"], ["d"], [{}], [[1.0]])
    build("cases", cache, store, "data_law_cases", rebuild=True, embed=embed, client_factory=FakeClient)       # 目录非空：必须显式 --rebuild
    build("cases", cache, store, "data_law_cases", rebuild=True, embed=embed, client_factory=FakeClient)
    assert FakeClient.deleted == [(str(store), "data_law_cases")] and (str(store), "someone_else") in FakeClient.registry


def test_dry_run_loads_no_model_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(kbbuild.base, "STORE_DIR", tmp_path / "chroma_db")
    cache = write_chunks(tmp_path / "cc", CASE_CHUNKS)
    info = build("cases", cache, tmp_path / "case_store", "data_law_cases", dry_run=True,
                 embed=lambda t: pytest.fail("dry-run 不应嵌入"), client_factory=lambda p: pytest.fail("dry-run 不应打开向量库"))
    assert info["dry_run"] is True and info["chunks"] == 2 and not (tmp_path / "case_store").exists()


def test_missing_chunks_file_and_duplicate_ids_are_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(kbbuild.base, "STORE_DIR", tmp_path / "chroma_db")
    with pytest.raises(SystemExit, match="找不到"):
        build("cases", tmp_path / "nowhere", tmp_path / "case_store", "c", dry_run=True)
    dup = write_chunks(tmp_path / "d", [CASE_CHUNKS[0], CASE_CHUNKS[0]])
    with pytest.raises(SystemExit, match="重复"):
        build("cases", dup, tmp_path / "case_store", "c", dry_run=True)


def test_embedding_is_done_in_batches(tmp_path, monkeypatch):
    monkeypatch.setattr(kbbuild.base, "STORE_DIR", tmp_path / "chroma_db")
    monkeypatch.setattr(kbbuild, "BATCH", 2)
    rows = [dict(CASE_CHUNKS[1], chunk_id=f"case:01:body{i:02d}:x", part=i) for i in range(5)]
    cache = write_chunks(tmp_path / "cc", rows)
    sizes = []
    build("cases", cache, tmp_path / "case_store", "data_law_cases", embed=lambda t: (sizes.append(len(t)), embed(t))[1], client_factory=FakeClient)
    assert sizes == [2, 2, 1]


def test_chunks_from_the_real_chunk_command_have_the_columns_the_builder_needs(tmp_path):
    """第一阶段 chunk 命令的输出就是建库的输入：字段必须对得上（用合成 docx 现场生成，不依赖真实资料）。"""
    from test_cases_parse import _case_doc, _quote_doc, base_spec
    from rag.cases import cli as case_cli
    from rag.quotes import cli as quote_cli
    case_cli.export_cases(_case_doc(tmp_path / "c.docx", base_spec()), tmp_path / "co")
    ids, docs, metas = case_rows(kbbuild.load_chunks(tmp_path / "co"))
    assert len(ids) == len(docs) == len(metas) > 4 and {m["kind"] for m in metas} == {"card", "body"}
    toc = [(2020, ["考察杭州城市大脑运营指挥中心作出重要指示"])]
    quote_cli.export_quotes(_quote_doc(tmp_path / "q.docx", toc, [(2020, ["1.指示内容。"])]), tmp_path / "qo")
    qids, qdocs, qmetas = quote_rows(kbbuild.load_chunks(tmp_path / "qo"))
    assert qids == ["quote:2020:01:" + qids[0].split(":")[-1]] and json.loads(qmetas[0]["paragraphs_json"]) == ["指示内容。"]


def test_kbbuild_source_never_references_thesis_store_for_writing():
    src = Path(kbbuild.__file__).read_text(encoding="utf-8")
    assert "STORE_DIR" in src and "check_store_safe(store, collection)" in src         # 写入前必经安全检查
    assert "rm -" not in src and "shutil.rmtree" not in src                              # 不做任何目录删除，只删自己的集合


# ================================================================ serve：启动时加载判例 / 名言（降级，不影响其它库）

def _serve(monkeypatch, tmp_path):
    from rag import serve
    import rag.cases.config as cc
    import rag.quotes.config as qc
    monkeypatch.setattr(cc, "CASE_STORE_DIR", tmp_path / "case_store")
    monkeypatch.setattr(cc, "CASE_CACHE_DIR", tmp_path / "case_cache")
    monkeypatch.setattr(qc, "QUOTE_STORE_DIR", tmp_path / "quote_store")
    return serve, cc, qc


def test_load_backends_return_none_when_store_dirs_missing(monkeypatch, tmp_path, caplog):
    serve, *_ = _serve(monkeypatch, tmp_path)
    import logging
    with caplog.at_level(logging.WARNING):
        assert serve.load_case_quote_backends() == (None, None, None)
    assert "判例库未启用" in caplog.text and "名言库未启用" in caplog.text and str(tmp_path) not in caplog.text      # 日志不带路径


def test_load_backends_pass_independent_store_and_collection_to_rag_service(monkeypatch, tmp_path):
    serve, cc, qc = _serve(monkeypatch, tmp_path)
    (tmp_path / "case_store").mkdir(); (tmp_path / "quote_store").mkdir(); (tmp_path / "case_cache").mkdir()
    (tmp_path / "case_cache" / "cases.jsonl").write_text(json.dumps(
        {"case_id": "case:01", "seq": 1, "chapter": "第一章 数据要素", "chapter_no": 1, "title": "甲案", "meta": {}}, ensure_ascii=False) + "\n", encoding="utf-8")
    made = []
    import rag.query as q
    monkeypatch.setattr(q, "RAGService", lambda **kw: (made.append(kw), object())[1])
    case_service, refs, quote_service = serve.load_case_quote_backends()
    assert case_service is not None and quote_service is not None and [r.case_id for r in refs] == ["case:01"]
    assert made == [{"mode": "hybrid", "store_dir": tmp_path / "case_store", "collection_name": cc.CASE_COLLECTION},
                    {"mode": "hybrid", "store_dir": tmp_path / "quote_store", "collection_name": qc.QUOTE_COLLECTION}]
    from rag import config as base
    assert all(m["store_dir"] != base.STORE_DIR and m["collection_name"] != base.COLLECTION_NAME for m in made)       # 绝不是论文库的


def test_load_backends_failure_degrades_that_library_only_without_leaking(monkeypatch, tmp_path, caplog):
    serve, *_ = _serve(monkeypatch, tmp_path)
    (tmp_path / "case_store").mkdir(); (tmp_path / "quote_store").mkdir()
    import logging, rag.query as q

    def boom(**kw):
        raise RuntimeError(f"cannot open {tmp_path} key=SECRET")
    monkeypatch.setattr(q, "RAGService", boom)
    with caplog.at_level(logging.WARNING):
        assert serve.load_case_quote_backends() == (None, None, None)
    assert "RuntimeError" in caplog.text and "SECRET" not in caplog.text and str(tmp_path) not in caplog.text


def test_missing_cases_jsonl_disables_cases_but_not_quotes(monkeypatch, tmp_path):
    serve, *_ = _serve(monkeypatch, tmp_path)
    (tmp_path / "case_store").mkdir(); (tmp_path / "quote_store").mkdir()           # 判例目录在，但没有 case_cache/cases.jsonl
    import rag.query as q
    monkeypatch.setattr(q, "RAGService", lambda **kw: object())
    case_service, refs, quote_service = serve.load_case_quote_backends()
    assert case_service is None and refs is None and quote_service is not None


def test_build_app_wires_case_and_quote_into_the_unified_service(monkeypatch, tmp_path):
    from rag import serve
    from rag.cases.retrieval import CaseRef
    refs = [CaseRef("case:01", 1, "第一章 数据要素", 1, "甲案", {})]
    sentinel = (object(), refs, object())
    monkeypatch.setattr(serve, "RAGService", lambda **kw: type("S", (), {"llm": None, "mode": "dense"})())
    monkeypatch.setattr(serve, "load_course_backend", lambda: (None, None))
    monkeypatch.setattr(serve, "load_case_quote_backends", lambda: sentinel)
    u = serve.build_app().state.unified
    assert u.case_service is sentinel[0] and u.case_refs == refs and u.quote_service is sentinel[2]
    assert u.case_retriever is not None and u.case_retriever.svc is sentinel[0] and u.quote_retriever.svc is sentinel[2]
    assert u.case_llm is None and u.quote_llm is None            # 课程 LLM 没配置时没有可用的作答模型 -> 两个库仍是 unavailable（见 _available）
    assert u._available(["case"]) == ["case"] and u._available(["quote"]) == ["quote"]


# ---------------------------------------------------------------- 目录非空即拒绝 / 一键入口 rag.cases.build、rag.quotes.build

def _patch_build(monkeypatch, tmp_path):
    """让 run_cli 用假 Chroma / 假嵌入；返回 (cache, store)。"""
    monkeypatch.setattr(kbbuild.base, "STORE_DIR", tmp_path / "chroma_db")
    import sys, types
    chroma = types.ModuleType("chromadb")
    chroma.PersistentClient = FakeClient
    st = types.ModuleType("sentence_transformers")
    st.SentenceTransformer = lambda *a, **k: type("M", (), {"encode": staticmethod(lambda t, **kw: type("A", (), {"tolist": lambda self: embed(t)})())})()
    monkeypatch.setitem(sys.modules, "chromadb", chroma)
    monkeypatch.setitem(sys.modules, "sentence_transformers", st)


def test_nonempty_store_directory_is_refused_unless_rebuild_even_without_collection(tmp_path, monkeypatch):
    monkeypatch.setattr(kbbuild.base, "STORE_DIR", tmp_path / "chroma_db")
    cache = write_chunks(tmp_path / "cache", CASE_CHUNKS)
    store = tmp_path / "case_store"
    store.mkdir()
    (store / "something.sqlite3").write_text("x")                                           # 目录里已有东西（哪怕不是我们的集合）
    with pytest.raises(SystemExit, match="已存在且非空"):
        build("cases", cache, store, "data_law_cases", embed=embed, client_factory=FakeClient)
    assert FakeClient.opened == [] and (store / "something.sqlite3").exists()                # 拒绝时连客户端都没打开，文件原样
    info = build("cases", cache, store, "data_law_cases", rebuild=True, embed=embed, client_factory=FakeClient)
    assert info["written"] == 2


def test_empty_or_missing_store_directory_is_fine(tmp_path, monkeypatch):
    monkeypatch.setattr(kbbuild.base, "STORE_DIR", tmp_path / "chroma_db")
    cache = write_chunks(tmp_path / "cache", QUOTE_CHUNKS)
    empty = tmp_path / "quote_store"
    empty.mkdir()
    assert build("quotes", cache, empty, "xi_digital_quotes", embed=embed, client_factory=FakeClient)["written"] == 1
    assert build("quotes", cache, tmp_path / "newdir", "xi_digital_quotes", embed=embed, client_factory=FakeClient)["written"] == 1


def test_dry_run_does_not_trip_the_nonempty_directory_check_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(kbbuild.base, "STORE_DIR", tmp_path / "chroma_db")
    cache = write_chunks(tmp_path / "cache", CASE_CHUNKS)
    store = tmp_path / "case_store"
    store.mkdir()
    (store / "f").write_text("x")
    assert build("cases", cache, store, "data_law_cases", dry_run=True)["chunks"] == 2 and sorted(p.name for p in store.iterdir()) == ["f"]


def test_cases_build_module_chunks_missing_cache_from_env_doc_then_builds(tmp_path, monkeypatch):
    """python -m rag.cases.build：缓存里还没有 chunks.jsonl 时，从 CASE_DOC 切块再建库；不传 --doc 就读 .env 的路径。"""
    docx = pytest.importorskip("docx")
    from test_cases_parse import _case_doc, base_spec
    _patch_build(monkeypatch, tmp_path)
    (tmp_path / "case_data").mkdir()
    doc = _case_doc(tmp_path / "case_data" / "判例.docx", base_spec())
    from rag.cases import config as cc
    monkeypatch.setattr(cc, "CASE_DOC", doc)
    monkeypatch.setattr(cc, "CASE_CACHE_DIR", tmp_path / "case_cache")
    monkeypatch.setattr(cc, "CASE_STORE_DIR", tmp_path / "case_store")
    from rag.cases import build as cases_build                                                # noqa: F401  (模块可导入)
    assert kbbuild.run_cli("cases", ["--dry-run"]) == 0
    assert (tmp_path / "case_cache" / "chunks.jsonl").is_file() and not (tmp_path / "case_store").exists()      # dry-run：切了块，没建库
    assert kbbuild.run_cli("cases", []) == 0
    col = FakeClient.registry[(str(tmp_path / "case_store"), "data_law_cases")]
    assert col.count() > 4 and str(tmp_path / "chroma_db") not in FakeClient.opened
    with pytest.raises(SystemExit, match="已存在且非空"):
        kbbuild.run_cli("cases", [])                                                          # 第二次：目录非空，拒绝
    assert kbbuild.run_cli("cases", ["--rebuild"]) == 0 and FakeClient.deleted == [(str(tmp_path / "case_store"), "data_law_cases")]


def test_quotes_build_module_uses_quote_doc_and_independent_store(tmp_path, monkeypatch):
    pytest.importorskip("docx")
    from test_cases_parse import _quote_doc
    _patch_build(monkeypatch, tmp_path)
    (tmp_path / "quote_data").mkdir()
    doc = _quote_doc(tmp_path / "quote_data" / "讲话.docx", [(2018, ["某会议上的讲话"])], [(2018, ["1.第一段。", "第二段。"])])
    from rag.quotes import config as qc
    monkeypatch.setattr(qc, "QUOTE_DOC", doc)
    monkeypatch.setattr(qc, "QUOTE_CACHE_DIR", tmp_path / "quote_cache")
    monkeypatch.setattr(qc, "QUOTE_STORE_DIR", tmp_path / "quote_store")
    assert kbbuild.run_cli("quotes", []) == 0
    col = FakeClient.registry[(str(tmp_path / "quote_store"), "xi_digital_quotes")]
    assert col.count() == 1 and json.loads(col.rows[0][2]["paragraphs_json"]) == ["第一段。", "第二段。"]


def test_missing_source_doc_gives_actionable_message(tmp_path, monkeypatch):
    _patch_build(monkeypatch, tmp_path)
    from rag.cases import config as cc
    monkeypatch.setattr(cc, "CASE_DOC", tmp_path / "nope.docx")
    monkeypatch.setattr(cc, "CASE_CACHE_DIR", tmp_path / "case_cache")
    monkeypatch.setattr(cc, "CASE_STORE_DIR", tmp_path / "case_store")
    with pytest.raises(SystemExit, match="找不到原始资料.*CASE_DOC"):
        kbbuild.run_cli("cases", [])


@pytest.mark.parametrize("kind", ["cases", "quotes"])
def test_build_entrypoints_refuse_when_store_is_inside_or_equal_to_thesis_store(tmp_path, monkeypatch, kind):
    _patch_build(monkeypatch, tmp_path)
    for bad in (tmp_path / "chroma_db", tmp_path / "chroma_db" / "sub"):
        with pytest.raises(SystemExit, match="拒绝"):
            kbbuild.run_cli(kind, ["--store", str(bad), "--dry-run"])
    assert FakeClient.opened == []                                                            # 一次都没打开过客户端


def test_default_raw_data_dirs_and_ignore_rules():
    from rag.cases import config as cc
    from rag.quotes import config as qc
    assert cc.CASE_DOC.parent.name == "case_data" and qc.QUOTE_DOC.parent.name == "quote_data"
    ig = Path(".gitignore").read_text(encoding="utf-8").splitlines()
    for d in ("rag/case_store/", "rag/quote_store/", "rag/case_data/", "rag/quote_data/"):
        assert d in ig
    for mod in ("rag/cases/build.py", "rag/quotes/build.py"):
        src = Path(mod).read_text(encoding="utf-8")
        assert "run_cli" in src and "chroma_db" not in src.replace("rag/chroma_db", "")        # 入口只转发给 kbbuild，不自己碰任何库


def test_collection_level_guard_still_blocks_when_directory_check_is_bypassed(tmp_path, monkeypatch):
    """目录检查之外还有集合级保险：假客户端不在磁盘落文件（目录看起来是空的），但集合里已有数据 -> 仍然拒绝覆盖。"""
    monkeypatch.setattr(kbbuild.base, "STORE_DIR", tmp_path / "chroma_db")
    cache = write_chunks(tmp_path / "cc", CASE_CHUNKS)
    store = tmp_path / "case_store"

    class Ghost(FakeClient):
        def __init__(self, path):
            self.path = str(path)
            FakeClient.opened.append(self.path)                                  # 不创建目录里的文件
    Ghost(store).create_collection("data_law_cases").add(["a"], ["d"], [{}], [[1.0]])
    with pytest.raises(SystemExit, match="拒绝覆盖"):
        build("cases", cache, store, "data_law_cases", embed=embed, client_factory=Ghost)
    assert FakeClient.deleted == []
