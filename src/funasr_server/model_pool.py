"""带 TTL（生存时间）的模型对象池模块。

本模块实现了一个线程安全的泛型对象缓存池 ``TTLModelPool``，专为 FunASR 模型实例管理
而设计。核心特性：

1. **TTL 自动过期** — 每个缓存条目会记录最后一次使用时间，超过 TTL 后自动失效，
   下次请求时重新加载。这样可以在内存占用和加载延迟之间取得平衡。
2. **细粒度锁** — 采用"全局锁 + 按 key 锁"的双层策略：
   - 全局锁（``_lock``）仅保护字典读写，持锁时间极短；
   - 按 key 锁（``_key_locks``）确保同一个模型只有一个线程在加载，不同模型可并行加载。
3. **双重检查（Double-Check）** — 在获取 per-key 锁之后会再次检查缓存，避免多个线程
   重复加载同一个模型。

典型用法::

    pool = TTLModelPool[AutoModel](ttl_seconds=900)
    model = pool.get_or_create("cn_meeting", loader=lambda: AutoModel(...))
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from loguru import logger


@dataclass
class PoolEntry[T]:
    """模型池中的单个缓存条目。

    每个条目包装了实际的模型对象以及最后一次被使用的时间戳，
    用于 TTL 过期判断。

    Attributes:
        value: 缓存的模型对象（泛型 T）。
        last_used: 最后一次使用的时间戳，由 ``time.monotonic()`` 生成。
            使用单调时钟而非系统时钟，不受系统校时影响。
    """

    value: T
    last_used: float


class TTLModelPool[T]:
    """带 TTL 过期策略的线程安全模型对象池。

    通过缓存已加载的模型实例来避免重复加载带来的高延迟（FunASR 模型加载
    通常需要数秒到数十秒），同时通过 TTL 机制自动释放长期未使用的模型以
    节省 GPU/CPU 内存。

    泛型参数 ``T`` 代表缓存对象的类型，在本项目中通常是 FunASR 的 AutoModel 实例。

    Args:
        ttl_seconds: 缓存条目的生存时间（秒）。
            - 正整数：条目在最后一次使用后超过该时间即视为过期。
            - ``-1``：永不过期，模型一旦加载就常驻内存。
            - ``0``：每次都重新加载（仅用于调试，生产环境不建议）。
    """

    def __init__(self, ttl_seconds: int) -> None:
        self.ttl_seconds = ttl_seconds
        # 模型缓存字典，key 为模型标识（如 profile name），value 为 PoolEntry
        self._entries: dict[str, PoolEntry[T]] = {}
        # 全局锁：保护 _entries 和 _key_locks 字典的并发读写
        self._lock = threading.Lock()
        # 按 key 的细粒度锁字典：确保同一个 key 的模型加载操作互斥
        self._key_locks: dict[str, threading.Lock] = {}

    def get_or_create(self, key: str, loader: Callable[[], T]) -> T:
        """从缓存中获取模型实例，如果不存在或已过期则通过 loader 创建。

        整个方法采用"快路径 + 慢路径"的两阶段设计：
        - **快路径**：在全局锁内查缓存，命中即返回，延迟极低。
        - **慢路径**：缓存未命中时，释放全局锁，获取 per-key 锁后调用
          loader 加载模型，避免阻塞其他 key 的请求。

        Args:
            key: 模型的唯一标识符，通常是 profile 名称（如 ``"cn_meeting"``）。
            loader: 无参数的可调用对象，用于创建/加载模型实例。
                只有在缓存未命中时才会被调用。

        Returns:
            缓存或新加载的模型实例。
        """
        # === 快路径：仅持有全局锁做缓存查找 ===
        with self._lock:
            entry = self._entries.get(key)
            # 缓存命中且未过期 → 更新使用时间后直接返回
            if entry is not None and not self._is_expired(entry):
                entry.last_used = time.monotonic()
                logger.debug("model pool cache hit: key={}", key)
                return entry.value

            # 缓存存在但已过期 → 移除旧条目，后续重新加载
            if entry is not None:
                logger.info("model pool TTL expired: key={}", key)
                self._entries.pop(key, None)

            # 获取或创建该 key 对应的细粒度锁（仅操作字典，无 I/O，非常快）
            key_lock = self._key_locks.setdefault(key, threading.Lock())

        # === 慢路径：持有 per-key 锁，确保同一模型只有一个线程在加载 ===
        with key_lock:
            # 双重检查：在等待 per-key 锁期间，另一个线程可能已经完成加载
            with self._lock:
                entry = self._entries.get(key)
                if entry is not None and not self._is_expired(entry):
                    entry.last_used = time.monotonic()
                    logger.debug("model pool cache hit after wait: key={}", key)
                    return entry.value

            # 在全局锁之外调用 loader，避免加载过程中阻塞整个池
            logger.info("model pool loading: key={}", key)
            t0 = time.monotonic()  # 记录加载开始时间，用于性能监控
            value = loader()
            elapsed = time.monotonic() - t0
            logger.info("model pool loaded: key={}, elapsed={:.2f}s", key, elapsed)

            # 加载完成后，持全局锁写入缓存
            with self._lock:
                self._entries[key] = PoolEntry(value=value, last_used=time.monotonic())

            return value

    def _is_expired(self, entry: PoolEntry[T]) -> bool:
        """判断缓存条目是否已过期。

        过期逻辑：
        - 当 ``ttl_seconds < 0``（通常为 -1）时，永不过期，始终返回 False。
        - 否则，比较当前时间与条目的最后使用时间，超过 TTL 即视为过期。

        Args:
            entry: 需要检查的缓存条目。

        Returns:
            True 表示已过期（需要重新加载），False 表示仍然有效。
        """
        return self.ttl_seconds >= 0 and time.monotonic() - entry.last_used > self.ttl_seconds
