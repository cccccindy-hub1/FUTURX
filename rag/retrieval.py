"""检索层：稠密向量 + BM25 稀疏 + RRF 融合 + 可选重排序 + 元数据过滤。

设计要点：
- 全部能力默认关闭，`HybridRetriever(mode="dense")` 行为与旧的纯向量检索一致。
- 重排序模型懒加载：只有 mode/rerank 真正需要时才下载与加载（本机无 GPU，默认不启用）。
- `rrf_fuse` 是纯函数，便于单测。
- BM25 语料从 Chroma 读出后缓存到磁盘，避免常驻服务每次启动重新分词数千文档。
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field

from .config import CACHE_DIR, FETCH_K, RRF_K, TOP_K

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- 分词

_JIEBA = None


def _load_jieba():
    global _JIEBA
    if _JIEBA is None:
        try:
            import jieba

            _JIEBA = jieba
        except ImportError:
            logger.warning("未安装 jieba，BM25 退化为字符 bigram 分词（检索质量下降）")
            _JIEBA = False
    return _JIEBA


def tokenize(text: str) -> list[str]:
    """中文分词；无 jieba 时退化为字符 bigram。"""
    jieba = _load_jieba()
    if jieba:
        return [t for t in jieba.lcut(text) if t.strip()]
    cleaned = "".join(text.split())
    return [cleaned[i : i + 2] for i in range(max(len(cleaned) - 1, 0))] or [cleaned]


# ---------------------------------------------------------------- RRF 融合

def rrf_fuse(rank_lists: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion：把多路排名的 id 列表融成一个带分数的排序。

    score(d) = Σ_i 1 / (k + rank_i(d))，rank 从 1 开始。返回按分数降序的 [(id, score)]。
    同分时按 id 字符串排序，保证结果稳定可复现。
    """
    scores: dict[str, float] = {}
    for ranks in rank_lists:
        for rank, doc_id in enumerate(ranks, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


# ---------------------------------------------------------------- BM25

class BM25Index:
    """基于 rank_bm25 的稀疏检索索引，带磁盘缓存。"""

    def __init__(self, ids: list[str], docs: list[str]):
        from rank_bm25 import BM25Okapi

        self.ids = list(ids)
        self.docs = list(docs)
        self._bm25 = BM25Okapi([tokenize(d) for d in self.docs])

    def search(self, query: str, n: int) -> list[tuple[str, float]]:
        scores = self._bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], self.ids[i]))[:n]
        return [(self.ids[i], float(scores[i])) for i in order if scores[i] > 0]

    @classmethod
    def build(cls, collection, name: str) -> "BM25Index":
        """从 Chroma collection 构建，命中磁盘缓存则直接复用。"""
        data = collection.get(include=["documents"])
        ids, docs = list(data["ids"]), list(data["documents"])
        cache_file = CACHE_DIR / f"{name}.bm25.pkl"
        fingerprint = (len(ids), hash(tuple(sorted(ids))))

        if cache_file.exists():
            try:
                with open(cache_file, "rb") as fh:
                    cached = pickle.load(fh)
                if cached.get("fingerprint") == fingerprint:
                    logger.info("命中 BM25 缓存（%d docs）", len(ids))
                    obj = cls.__new__(cls)
                    obj.ids, obj.docs, obj._bm25 = cached["ids"], cached["docs"], cached["bm25"]
                    return obj
            except Exception as e:  # 缓存损坏则重建
                logger.warning("BM25 缓存不可用，重建：%s", e)

        logger.info("构建 BM25 索引（%d docs）…", len(ids))
        obj = cls(ids, docs)
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "wb") as fh:
                pickle.dump(
                    {"fingerprint": fingerprint, "ids": obj.ids, "docs": obj.docs, "bm25": obj._bm25},
                    fh,
                )
        except Exception as e:
            logger.warning("BM25 缓存写入失败：%s", e)
        return obj


# ---------------------------------------------------------------- 重排序

class Reranker:
    """CrossEncoder 重排序，懒加载模型（首次调用才下载/加载）。"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def _ensure(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("加载重排序模型 %s（首次较慢，会下载模型）…", self.model_name)
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, docs: list[str], top_n: int) -> list[tuple[int, float]]:
        """返回按相关性降序的 [(原索引, 分数)]，取前 top_n。"""
        if not docs:
            return []
        model = self._ensure()
        scores = model.predict([(query, d) for d in docs])
        order = sorted(range(len(docs)), key=lambda i: (-float(scores[i]), i))[:top_n]
        return [(i, float(scores[i])) for i in order]


# ---------------------------------------------------------------- 检索结果

@dataclass
class RetrievalResult:
    ids: list[str] = field(default_factory=list)
    documents: list[str] = field(default_factory=list)
    metadatas: list[dict] = field(default_factory=list)
    distances: list[float] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    debug: dict = field(default_factory=dict)

    def as_chroma(self) -> dict:
        """转回 Chroma `query()` 的嵌套列表形状，兼容既有调用方。"""
        return {
            "ids": [self.ids],
            "documents": [self.documents],
            "metadatas": [self.metadatas],
            "distances": [self.distances],
        }


# ---------------------------------------------------------------- 混合检索

class HybridRetriever:
    """稠密 + 稀疏融合检索，可选重排序与元数据过滤。"""

    def __init__(self, collection, embed_fn, mode: str = "dense", rrf_k: int = RRF_K):
        self.collection = collection
        self.embed_fn = embed_fn
        self.mode = mode
        self.rrf_k = rrf_k
        self._bm25: BM25Index | None = None
        self._reranker: Reranker | None = None

    # -- 惰性资源 --
    def bm25(self, name: str) -> BM25Index:
        if self._bm25 is None:
            self._bm25 = BM25Index.build(self.collection, name)
        return self._bm25

    def attach_reranker(self, reranker: Reranker) -> None:
        self._reranker = reranker

    # -- 各路召回 --
    def _dense(self, query: str, n: int, where: dict | None) -> tuple[list[str], dict[str, dict], dict[str, float]]:
        emb = self.embed_fn(query)
        res = self.collection.query(query_embeddings=[emb], n_results=n, where=where)
        ids = list(res["ids"][0])
        metas = {i: m for i, m in zip(ids, res["metadatas"][0])}
        dists = {i: float(d) for i, d in zip(ids, res["distances"][0])}
        return ids, metas, dists

    def _sparse(self, query: str, n: int, where: dict | None, name: str) -> list[str]:
        hits = self.bm25(name).search(query, n)
        ids = [i for i, _ in hits]
        if where and ids:
            got = self.collection.get(ids=ids, include=["metadatas"])
            by_id = dict(zip(got["ids"], got["metadatas"]))
            ids = [i for i in ids if _match(by_id.get(i, {}), where)]
        return ids

    def search(
        self,
        query: str,
        top_k: int = TOP_K,
        fetch_k: int = FETCH_K,
        where: dict | None = None,
        collection_name: str = "collection",
    ) -> RetrievalResult:
        rank_lists: list[list[str]] = []
        metas: dict[str, dict] = {}
        dists: dict[str, float] = {}
        debug: dict = {"mode": self.mode}

        if self.mode in ("dense", "hybrid"):
            d_ids, d_metas, d_dists = self._dense(query, fetch_k, where)
            metas.update(d_metas)
            dists.update(d_dists)
            rank_lists.append(d_ids)
            debug["dense"] = d_ids

        if self.mode in ("bm25", "hybrid"):
            s_ids = self._sparse(query, fetch_k, where, collection_name)
            rank_lists.append(s_ids)
            debug["bm25"] = s_ids

        if not rank_lists:
            return RetrievalResult(debug=debug)

        if self.mode == "dense":
            # 稠密模式：直接用余弦距离排序（越小越近），与旧行为一致
            fused = [(i, dists.get(i, 1.0)) for i in rank_lists[0]][:top_k]
        elif self.mode == "bm25":
            fused = [(i, 0.0) for i in rank_lists[0]][:top_k]
        else:
            fused = rrf_fuse(rank_lists, self.rrf_k)[:top_k]

        debug["fused"] = [i for i, _ in fused]
        ids = [i for i, _ in fused]
        docs = self._fetch_docs(ids)
        fused_scores = [s for _, s in fused]

        # 重排序（可选）：对候选重排后重新取 top_k
        if self._reranker is not None and docs:
            pairs = self._reranker.rerank(query, docs, min(top_k, len(docs)))
            rerank_scores = {ids[i]: s for i, s in pairs}
            order = [i for i, _ in pairs]
            ids = [ids[i] for i in order]
            docs = [docs[i] for i in order]
            fused_scores = [rerank_scores[i] for i in ids]
            debug["rerank_scores"] = rerank_scores

        out_dists = []
        for i in ids:
            d = dists.get(i)
            out_dists.append(float(d) if d is not None else 0.0)

        return RetrievalResult(
            ids=ids,
            documents=docs,
            metadatas=[metas.get(i, {}) for i in ids],
            distances=out_dists,
            scores=[float(s) for s in fused_scores],
            debug=debug,
        )

    def _fetch_docs(self, ids: list[str]) -> list[str]:
        if not ids:
            return []
        got = self.collection.get(ids=ids, include=["documents"])
        by_id = dict(zip(got["ids"], got["documents"]))
        return [by_id.get(i, "") for i in ids]


def _match(meta: dict, where: dict) -> bool:
    """朴素元数据匹配，支持 {"k": v} 与 {"k": {"$in": [...]}}。"""
    for key, cond in where.items():
        val = meta.get(key)
        if isinstance(cond, dict):
            if "$in" in cond and val not in cond["$in"]:
                return False
            if "$eq" in cond and val != cond["$eq"]:
                return False
        elif val != cond:
            return False
    return True


def parse_where(items: list[str] | None) -> dict | None:
    """把 CLI 的 ["category=006/研究方法", ...] 解析为 Chroma where 字典。"""
    if not items:
        return None
    where: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--where 需要 KEY=VALUE 形式，收到：{item!r}")
        key, _, value = item.partition("=")
        key, value = key.strip(), value.strip()
        if not key:
            raise ValueError(f"--where 的键不能为空：{item!r}")
        where[key] = value
    return where
