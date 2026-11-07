"""API key 集合的热加载与启动自检（不联网、不加载模型）。"""
from __future__ import annotations

import time

import pytest

from rag import config
from rag.config import ApiKeySet


def _write(p, text):
    p.write_text(text, encoding="utf-8")
    return p


def test_inline_keys_only(tmp_path):
    ks = ApiKeySet(frozenset({"sk-a"}), "", 0)
    assert "sk-a" in ks and len(ks) == 1 and bool(ks)


def test_file_keys_are_merged_with_inline(tmp_path):
    f = _write(tmp_path / "k.txt", "# 注释\nsk-file-1\n\nsk-file-2  # 行尾注释\n")
    ks = ApiKeySet(frozenset({"sk-inline"}), str(f), 0)
    assert len(ks) == 3
    assert "sk-inline" in ks and "sk-file-1" in ks and "sk-file-2" in ks


def test_reload_picks_up_revoke_and_issue(tmp_path):
    """撤销与新增都靠热加载生效：不重启服务。"""
    f = _write(tmp_path / "k.txt", "sk-a\nsk-b\n")
    ks = ApiKeySet(frozenset(), str(f), 0)
    assert "sk-b" in ks

    _write(f, "sk-a\nsk-c\n")          # 撤销 b，新增 c
    assert "sk-b" not in ks
    assert "sk-c" in ks


def test_deleted_file_keeps_last_known_keys(tmp_path):
    """文件被误删时不能退回空集合——空集合等于不鉴权，服务会对公网敞开。"""
    f = _write(tmp_path / "k.txt", "sk-a\n")
    ks = ApiKeySet(frozenset(), str(f), 0)
    f.unlink()
    assert "sk-a" in ks and len(ks) == 1


def test_throttle_skips_stat_before_next_check(tmp_path, monkeypatch):
    f = _write(tmp_path / "k.txt", "sk-a\n")
    ks = ApiKeySet(frozenset(), str(f), 3600)      # 一小时内不再检查
    _write(f, "sk-a\nsk-b\n")
    assert "sk-b" not in ks                        # 节流内看不到
    ks.reload()                                    # 强制重读才看得到
    assert "sk-b" in ks


def test_modified_time_change_is_detected(tmp_path):
    """mtime 变了要重读；只改内容不改大小也能发现（同一秒内改两次靠 mtime）。"""
    f = _write(tmp_path / "k.txt", "sk-a\n")
    ks = ApiKeySet(frozenset(), str(f), 0)
    time.sleep(0.01)
    _write(f, "sk-b\n")
    assert "sk-a" not in ks and "sk-b" in ks


def test_startup_guard_missing_file(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "API_KEYS_FILE", str(tmp_path / "nope.txt"))
    monkeypatch.setattr(config, "_API_KEYS_INLINE", frozenset())
    with pytest.raises(RuntimeError, match="不存在"):
        config._check_api_key_config()


def test_startup_guard_empty_file(monkeypatch, tmp_path):
    f = _write(tmp_path / "k.txt", "# 只有注释\n")
    monkeypatch.setattr(config, "API_KEYS_FILE", str(f))
    monkeypatch.setattr(config, "_API_KEYS_INLINE", frozenset())
    with pytest.raises(RuntimeError, match="没有任何 key"):
        config._check_api_key_config()


def test_startup_guard_passes_with_inline_fallback(monkeypatch, tmp_path):
    """文件为空但有 API_KEYS 兜底时可以启动。"""
    f = _write(tmp_path / "k.txt", "")
    monkeypatch.setattr(config, "API_KEYS_FILE", str(f))
    monkeypatch.setattr(config, "_API_KEYS_INLINE", frozenset({"sk-a"}))
    config._check_api_key_config()


def test_startup_guard_noop_without_file(monkeypatch):
    monkeypatch.setattr(config, "API_KEYS_FILE", "")
    config._check_api_key_config()                 # 未配置密钥文件：本机裸跑，允许
