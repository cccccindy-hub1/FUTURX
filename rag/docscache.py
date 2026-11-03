"""三个文档库（判例 / 名言 / 政策）共用的解析缓存：启动时直接读原始 docx，不建向量库。

缓存是一个放在 data 目录里的 JSON（<docx 名>.<库名>.cache.json），内含解析结果与源文件的 mtime / 大小；
下次启动时 mtime 或大小任一不同就重新解析并覆盖缓存，相同则直接读 JSON。缓存读不出来（损坏 / 版本不对）也只是重新解析，
绝不因为缓存问题让服务起不来。找不到 docx 返回 None，由调用方跳过这个库，服务照常启动。
缓存写不进去（只读目录等）只记一行日志，不影响使用。
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger(__name__)
CACHE_VERSION = 1


def fingerprint(path: Path) -> dict:
    st = os.stat(path)
    return {"mtime_ns": st.st_mtime_ns, "size": st.st_size}


def cache_path(doc: Path, kind: str) -> Path:
    return doc.with_name(f"{doc.stem}.{kind}.cache.json")


def load_or_build(doc: Path | str | None, kind: str, build: Callable[[Path], Any], *, version: int = 1,
                  use_cache: bool = True) -> tuple[Any, dict] | None:
    """返回 (数据, 信息)；信息含 source（cache / parsed）、耗时、缓存路径。docx 不存在返回 None。
    build(doc) 必须返回可 JSON 序列化的对象（dict / list）。version：库自己的解析版本，变了缓存作废。"""
    if doc is None:
        return None
    doc = Path(doc)
    if not doc.is_file():
        return None
    fp = fingerprint(doc)
    cp = cache_path(doc, kind)
    t0 = time.perf_counter()
    if use_cache and cp.is_file():
        try:
            blob = json.loads(cp.read_text(encoding="utf-8"))
            if blob.get("cache_version") == CACHE_VERSION and blob.get("version") == version and blob.get("source") == fp:
                return blob["data"], {"source": "cache", "seconds": round(time.perf_counter() - t0, 3), "cache": str(cp)}
        except Exception as e:  # noqa: BLE001 - 缓存坏了就重新解析
            log.warning("%s 缓存不可用（%s），重新解析", kind, type(e).__name__)
    data = build(doc)
    info = {"source": "parsed", "seconds": round(time.perf_counter() - t0, 3), "cache": str(cp)}
    if use_cache:
        try:
            tmp = cp.with_suffix(".tmp")
            tmp.write_text(json.dumps({"cache_version": CACHE_VERSION, "version": version, "source": fp, "kind": kind, "data": data},
                                      ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, cp)                                   # 原子替换，避免半截文件
        except Exception as e:  # noqa: BLE001
            info["cache_write_error"] = type(e).__name__
            log.warning("%s 缓存写入失败（%s），不影响使用", kind, type(e).__name__)
    return data, info
