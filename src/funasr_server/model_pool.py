from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from loguru import logger


@dataclass
class PoolEntry[T]:
    value: T
    last_used: float


class TTLModelPool[T]:
    def __init__(self, ttl_seconds: int) -> None:
        self.ttl_seconds = ttl_seconds
        self._entries: dict[str, PoolEntry[T]] = {}
        self._lock = threading.Lock()
        self._key_locks: dict[str, threading.Lock] = {}

    def get_or_create(self, key: str, loader: Callable[[], T]) -> T:
        # --- fast path: global lock for cache lookup only ---
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None and not self._is_expired(entry):
                entry.last_used = time.monotonic()
                logger.debug("model pool cache hit: key={}", key)
                return entry.value

            if entry is not None:
                logger.info("model pool TTL expired: key={}", key)
                self._entries.pop(key, None)

            # get or create a per-key lock (cheap, no I/O)
            key_lock = self._key_locks.setdefault(key, threading.Lock())

        # --- slow path: per-key lock so only one thread loads a given key ---
        with key_lock:
            # double-check: another thread may have loaded while we waited
            with self._lock:
                entry = self._entries.get(key)
                if entry is not None and not self._is_expired(entry):
                    entry.last_used = time.monotonic()
                    logger.debug("model pool cache hit after wait: key={}", key)
                    return entry.value

            # call loader *outside* the global lock
            logger.info("model pool loading: key={}", key)
            t0 = time.monotonic()
            value = loader()
            elapsed = time.monotonic() - t0
            logger.info("model pool loaded: key={}, elapsed={:.2f}s", key, elapsed)

            with self._lock:
                self._entries[key] = PoolEntry(value=value, last_used=time.monotonic())

            return value

    def _is_expired(self, entry: PoolEntry[T]) -> bool:
        return self.ttl_seconds >= 0 and time.monotonic() - entry.last_used > self.ttl_seconds
