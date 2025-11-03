from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional, Tuple


class TTLCache:
    """Simple thread-safe TTL cache for small payloads.

    Not a distributed cache. For multi-process deployments, use Redis.
    """

    def __init__(self, ttl_seconds: int = 30, max_items: int = 1024) -> None:
        self.ttl = max(0, int(ttl_seconds))
        self.max_items = max_items
        self._data: Dict[str, Tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        if self.ttl <= 0:
            return None
        now = time.time()
        with self._lock:
            entry = self._data.get(key)
            if not entry:
                return None
            ts, value = entry
            if now - ts > self.ttl:
                # expired
                self._data.pop(key, None)
                return None
            return value

    def set(self, key: str, value: Any) -> None:
        if self.ttl <= 0:
            return
        now = time.time()
        with self._lock:
            if len(self._data) >= self.max_items:
                # Drop oldest
                oldest_key = min(self._data, key=lambda k: self._data[k][0])
                self._data.pop(oldest_key, None)
            self._data[key] = (now, value)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

