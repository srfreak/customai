from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class LLMStreamState:
    queue: "asyncio.Queue[Optional[str]]"
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    finished_event: asyncio.Event = field(default_factory=asyncio.Event)
    tokens: List[str] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    async def cancel(self, reason: Optional[str] = None) -> None:
        if reason:
            logger.info("Cancelling LLM stream: %s", reason)
            self.metadata["cancel_reason"] = reason
        self.cancel_event.set()
        await self.queue.put(None)

    def append_token(self, token: str) -> None:
        self.tokens.append(token)

    def text(self) -> str:
        return "".join(self.tokens)

