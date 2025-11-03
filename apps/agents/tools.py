from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

ToolCallable = Callable[..., Awaitable[Any]]


@dataclass
class RegisteredTool:
    name: str
    description: str
    schema: Dict[str, Any]
    handler: ToolCallable


class ToolRegistry:
    """Runtime tool registry exposed to LLM tool-calling."""

    def __init__(self) -> None:
        self._tools: Dict[str, RegisteredTool] = {}

    def register(
        self,
        name: str,
        handler: ToolCallable,
        *,
        description: str,
        schema: Dict[str, Any],
    ) -> None:
        self._tools[name] = RegisteredTool(name=name, description=description, schema=schema, handler=handler)
        logger.debug("Registered tool %s", name)

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def tool_specs(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.schema,
                },
            }
            for tool in self._tools.values()
        ]

    async def invoke(self, name: str, **kwargs: Any) -> Any:
        tool = self._tools.get(name)
        if not tool:
            raise KeyError(f"Tool '{name}' not registered")
        return await tool.handler(**kwargs)


# ---------------------------------------------------------------------------
# Built-in tool implementations (async for uniformity)
# ---------------------------------------------------------------------------

async def search_docs(query: str) -> Dict[str, Any]:
    """Stubbed documentation search, replace with vector store lookup."""
    logger.info("search_docs tool invoked with query=%s", query)
    # TODO: integrate with vector store / embeddings.
    await asyncio.sleep(0)
    return {"results": [{"title": "Getting Started", "snippet": "Deploy Scrappy Singh via FastAPI..."}]}


async def fetch_crm_data(lead_email: str) -> Dict[str, Any]:
    logger.info("fetch_crm_data tool invoked for %s", lead_email)
    await asyncio.sleep(0)
    return {
        "lead_email": lead_email,
        "last_contact": "2024-05-01",
        "lifecycle_stage": "opportunity",
        "open_deals": 2,
    }


async def send_followup_email(lead_email: str, subject: str, body: str) -> Dict[str, Any]:
    logger.info("send_followup_email queued email to %s", lead_email)
    await asyncio.sleep(0)
    return {"status": "queued", "lead_email": lead_email, "subject": subject}


async def read_product_faq(topic: str) -> Dict[str, Any]:
    logger.info("read_product_faq invoked for topic=%s", topic)
    await asyncio.sleep(0)
    return {"topic": topic, "answer": "Refer to the onboarding deck for pricing tiers and integration steps."}


def bootstrap_default_tools(registry: ToolRegistry) -> None:
    registry.register(
        "search_docs",
        search_docs,
        description="Search internal knowledge base for deployment and troubleshooting steps.",
        schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search terms"},
            },
            "required": ["query"],
        },
    )
    registry.register(
        "fetch_crm_data",
        fetch_crm_data,
        description="Fetch CRM metadata for a lead by email.",
        schema={
            "type": "object",
            "properties": {
                "lead_email": {"type": "string"},
            },
            "required": ["lead_email"],
        },
    )
    registry.register(
        "send_followup_email",
        send_followup_email,
        description="Queue a templated follow-up email to a lead.",
        schema={
            "type": "object",
            "properties": {
                "lead_email": {"type": "string"},
                "subject": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["lead_email", "subject", "body"],
        },
    )
    registry.register(
        "read_product_faq",
        read_product_faq,
        description="Fetch the FAQ answer for a given product topic.",
        schema={
            "type": "object",
            "properties": {
                "topic": {"type": "string"},
            },
            "required": ["topic"],
        },
    )
