from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Optional
import threading

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from apps.agents.agent_base import BaseAgent

logger = logging.getLogger(__name__)


@dataclass
class ManagedSubagent:
    name: str
    agent: "BaseAgent"
    ttl_seconds: int = 900
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_active_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def expires_at(self) -> datetime:
        return self.last_active_at + timedelta(seconds=self.ttl_seconds)

    def touch(self) -> None:
        self.last_active_at = datetime.utcnow()

    def expired(self) -> bool:
        return datetime.utcnow() > self.expires_at


class SubagentManager:
    """Lifecycle management for spawned subagents."""

    def __init__(self) -> None:
        self._agents: Dict[str, ManagedSubagent] = {}
        self._lock = threading.RLock()

    def spawn(
        self,
        name: str,
        agent: "BaseAgent",
        *,
        ttl_seconds: int = 900,
    ) -> ManagedSubagent:
        managed = ManagedSubagent(name=name, agent=agent, ttl_seconds=ttl_seconds)
        with self._lock:
            self._agents[name] = managed
        logger.info("Spawned subagent %s ttl=%ss", name, ttl_seconds)
        return managed

    def route(self, name: str) -> Optional["BaseAgent"]:
        with self._lock:
            managed = self._agents.get(name)
            if not managed:
                return None
            if managed.expired():
                logger.info("Subagent %s expired; removing before route.", name)
                self._agents.pop(name, None)
                return None
            managed.touch()
            return managed.agent

    def deactivate(self, name: str) -> None:
        with self._lock:
            if self._agents.pop(name, None):
                logger.info("Deactivated subagent %s", name)

    def prune_expired(self) -> None:
        with self._lock:
            expired = [name for name, managed in self._agents.items() if managed.expired()]
            for name in expired:
                self._agents.pop(name, None)
                logger.info("Pruned expired subagent %s", name)

    def snapshot(self) -> Dict[str, Dict[str, str]]:
        with self._lock:
            return {
                name: {
                    "agent_id": managed.agent.agent_id,
                    "expires_at": managed.expires_at.isoformat(),
                    "ttl_seconds": str(managed.ttl_seconds),
                }
                for name, managed in self._agents.items()
            }
