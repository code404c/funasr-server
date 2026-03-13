"""model_pool.py 的单元测试 — 覆盖 get_or_create 重复调用与过期逻辑。"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

from funasr_server.model_pool import TTLModelPool


class TestTTLModelPool:
    """TTLModelPool 核心逻辑测试。"""

    def test_get_or_create_loads_on_first_call(self) -> None:
        """首次调用 get_or_create 时应执行 loader 并返回值。"""
        pool: TTLModelPool[str] = TTLModelPool(ttl_seconds=60)
        loader = MagicMock(return_value="model_v1")

        result = pool.get_or_create("key1", loader)

        assert result == "model_v1"
        loader.assert_called_once()

    def test_get_or_create_returns_cached_on_second_call(self) -> None:
        """重复调用 get_or_create 时应返回缓存值，不再调用 loader。"""
        pool: TTLModelPool[str] = TTLModelPool(ttl_seconds=60)
        loader = MagicMock(return_value="model_v1")

        result1 = pool.get_or_create("key1", loader)
        result2 = pool.get_or_create("key1", loader)

        assert result1 == "model_v1"
        assert result2 == "model_v1"
        loader.assert_called_once()

    def test_get_or_create_reloads_after_expiry(self) -> None:
        """TTL 过期后应重新调用 loader 加载模型。"""
        pool: TTLModelPool[str] = TTLModelPool(ttl_seconds=1)
        loader = MagicMock(side_effect=["model_v1", "model_v2"])

        result1 = pool.get_or_create("key1", loader)
        assert result1 == "model_v1"
        assert loader.call_count == 1

        # 模拟时间推进使缓存过期
        with patch("funasr_server.model_pool.time") as mock_time:
            # 第一次 monotonic 调用在 _is_expired 检查中
            # 第二次在存储新 entry 时
            initial = time.monotonic()
            mock_time.monotonic.side_effect = [initial + 100, initial + 100]

            result2 = pool.get_or_create("key1", loader)

        assert result2 == "model_v2"
        assert loader.call_count == 2

    def test_get_or_create_different_keys(self) -> None:
        """不同 key 应独立管理缓存。"""
        pool: TTLModelPool[str] = TTLModelPool(ttl_seconds=60)
        loader_a = MagicMock(return_value="model_a")
        loader_b = MagicMock(return_value="model_b")

        result_a = pool.get_or_create("key_a", loader_a)
        result_b = pool.get_or_create("key_b", loader_b)

        assert result_a == "model_a"
        assert result_b == "model_b"
        loader_a.assert_called_once()
        loader_b.assert_called_once()

    def test_negative_ttl_never_expires(self) -> None:
        """ttl_seconds < 0 时缓存永不过期。"""
        pool: TTLModelPool[str] = TTLModelPool(ttl_seconds=-1)
        loader = MagicMock(return_value="model_v1")

        pool.get_or_create("key1", loader)

        # 即使 last_used 很久之前也不该过期
        entry = pool._entries["key1"]
        entry.last_used = time.monotonic() - 999999

        result = pool.get_or_create("key1", loader)
        assert result == "model_v1"
        loader.assert_called_once()

    def test_is_expired_returns_true(self) -> None:
        """_is_expired 在超过 TTL 后返回 True。"""
        pool: TTLModelPool[str] = TTLModelPool(ttl_seconds=10)
        from funasr_server.model_pool import PoolEntry

        old_entry = PoolEntry(value="old", last_used=time.monotonic() - 20)
        assert pool._is_expired(old_entry) is True

    def test_is_expired_returns_false(self) -> None:
        """_is_expired 在未超过 TTL 时返回 False。"""
        pool: TTLModelPool[str] = TTLModelPool(ttl_seconds=60)
        from funasr_server.model_pool import PoolEntry

        fresh_entry = PoolEntry(value="fresh", last_used=time.monotonic())
        assert pool._is_expired(fresh_entry) is False
