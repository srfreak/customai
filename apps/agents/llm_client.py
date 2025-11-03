from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Iterable, List, Optional
import hashlib
import json as _json

from openai import OpenAI

from core.config import settings
from shared.cache import TTLCache
from apps.agents.llm_stream import LLMStreamState

client = OpenAI(api_key=settings.OPENAI_API_KEY)

logger = logging.getLogger(__name__)

# Cache for short-term reuse of identical prompts
_llm_cache = TTLCache(ttl_seconds=getattr(settings, "LLM_CACHE_TTL_SECONDS", 30))


def _inject_persona_metadata(
    messages: List[Dict[str, str]],
    *,
    persona: Optional[Dict[str, Any]] = None,
    goals: Optional[Iterable[str]] = None,
) -> List[Dict[str, str]]:
    if not persona and not goals:
        return messages
    system_parts: List[str] = []
    if persona:
        if persona.get("name"):
            system_parts.append(f"Persona: {persona['name']}")
        if persona.get("tone_override"):
            system_parts.append(f"Desired tone: {persona['tone_override']}")
        elif persona.get("tone"):
            system_parts.append(f"Tone: {persona['tone']}")
        if persona.get("description"):
            system_parts.append(f"Description: {persona['description']}")
        if persona.get("locale"):
            system_parts.append(f"Locale: {persona['locale']}")
    if goals:
        system_parts.append("Goals: " + "; ".join(goals))

    if system_parts:
        injected = {"role": "system", "content": "\n".join(system_parts)}
        return [injected] + messages
    return messages


async def stream_chat_completion(
    messages: List[Dict[str, str]],
    *,
    model: Optional[str] = None,
    persona: Optional[Dict[str, Any]] = None,
    goals: Optional[Iterable[str]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    temperature: float = 0.7,
) -> LLMStreamState:
    """Stream chat completion tokens into an asyncio.Queue."""
    if not settings.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    queue: "asyncio.Queue[Optional[str]]" = asyncio.Queue()
    stream_state = LLMStreamState(queue=queue)

    loop = asyncio.get_running_loop()

    def _run() -> None:
        try:
            request = {
                "model": model or settings.OPENAI_MODEL,
                "messages": _inject_persona_metadata(messages, persona=persona, goals=goals),
                "temperature": temperature,
                "stream": True,
            }
            if tools:
                request["tools"] = tools
                request["tool_choice"] = "auto"

            stream = client.chat.completions.create(**request)
            for chunk in stream:
                if stream_state.cancel_event.is_set():
                    logger.info("LLM stream cancellation acknowledged.")
                    break
                for choice in chunk.choices:
                    delta = choice.delta
                    if delta.content:
                        token = delta.content
                        stream_state.append_token(token)
                        loop.call_soon_threadsafe(queue.put_nowait, token)
                    if delta.tool_calls:
                        stream_state.tool_calls.extend(
                            [call.model_dump() for call in delta.tool_calls]
                        )
            loop.call_soon_threadsafe(queue.put_nowait, None)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("OpenAI streaming failure: %s", exc)
            stream_state.error = str(exc)
            loop.call_soon_threadsafe(queue.put_nowait, None)
        finally:
            loop.call_soon_threadsafe(stream_state.finished_event.set)

    loop.run_in_executor(None, _run)
    return stream_state


async def complete_chat(
    messages: List[Dict[str, str]],
    *,
    model: Optional[str] = None,
    persona: Optional[Dict[str, Any]] = None,
    goals: Optional[Iterable[str]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    """Synchronous completion helper used for non-streaming calls."""
    if not settings.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    target_model = model or getattr(settings, "OPENAI_MODEL", "gpt-4o")
    req_messages = _inject_persona_metadata(messages, persona=persona, goals=goals)
    request = {
        "model": target_model,
        "messages": req_messages,
        "temperature": temperature,
    }
    if tools:
        request["tools"] = tools
        request["tool_choice"] = "auto"

    # Build cache key
    try:
        cache_key = hashlib.sha256(
            (target_model + "\n" + _json.dumps(request, sort_keys=True, default=str)).encode("utf-8")
        ).hexdigest()
    except Exception:
        cache_key = ""

    if cache_key:
        cached = _llm_cache.get(cache_key)
        if cached is not None:
            return cached

    response = await asyncio.to_thread(client.chat.completions.create, **request)
    payload = response.model_dump()
    if cache_key:
        _llm_cache.set(cache_key, payload)
    return payload
