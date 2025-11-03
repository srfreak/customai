from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Set


class LiveCallRegistry:
    """Shared in-memory state tracking active calls for the ops dashboard."""

    def __init__(self) -> None:
        self._calls: Dict[str, Dict[str, Any]] = {}
        self._subscribers: Set[asyncio.Queue] = set()
        self._lock = asyncio.Lock()

    async def update_call(self, call_id: str, payload: Dict[str, Any]) -> None:
        async with self._lock:
            record = {**self._calls.get(call_id, {}), **payload}
            record["call_id"] = call_id
            record["updated_at"] = datetime.utcnow().isoformat()
            self._calls[call_id] = record
            await self._broadcast(record)

    async def remove_call(self, call_id: str) -> None:
        async with self._lock:
            self._calls.pop(call_id, None)
            await self._broadcast({"call_id": call_id, "status": "completed"})

    async def snapshot(self) -> List[Dict[str, Any]]:
        async with self._lock:
            return list(self._calls.values())

    async def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        async with self._lock:
            self._subscribers.add(queue)
        return queue

    async def unsubscribe(self, queue: asyncio.Queue) -> None:
        async with self._lock:
            self._subscribers.discard(queue)

    async def _broadcast(self, event: Dict[str, Any]) -> None:
        dead: Set[asyncio.Queue] = set()
        for queue in self._subscribers:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                dead.add(queue)
        for queue in dead:
            self._subscribers.discard(queue)


live_calls = LiveCallRegistry()
