"""Service helpers for sales agent interactions."""
from __future__ import annotations

import asyncio
import base64
import inspect
import json
import logging
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence
from pathlib import Path

import httpx
from fastapi import HTTPException, status
from twilio.base.exceptions import TwilioRestException
from twilio.rest import Client

from core.config import settings
from core.database import get_collection
from shared.constants import (
    COLLECTION_STRATEGIES,
    COLLECTION_CALLS,
    CALL_STATUS_FAILED,
    CALL_STATUS_COMPLETED,
    CALL_STATUS_IN_PROGRESS,
)

logger = logging.getLogger(__name__)

_STRATEGY_FIELDS_PRIORITY: Sequence[str] = (
    "greeting",
    "pitch",
    "faqs",
    "objections",
    "closing",
)

# Core prompts/assets directory (bundled in repo)
_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def _read_json_file(path: Path) -> Dict[str, Any]:
    try:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def load_core_sales_strategy() -> Dict[str, Any]:
    """Load core, reusable sales skills (common across tenants)."""
    return _read_json_file(_PROMPTS_DIR / "strategy.json")


def load_core_objections_map() -> Dict[str, str]:
    data = _read_json_file(_PROMPTS_DIR / "objections_map.json")
    return {k: str(v) for k, v in data.items()} if isinstance(data, dict) else {}


def _merge_dict(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow dict merge with b overriding a, recursively for nested dicts."""
    out = dict(a or {})
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_dict(out[k], v)
        else:
            out[k] = v
    return out


def _merge_list(a: Optional[Sequence[Any]], b: Optional[Sequence[Any]]) -> List[Any]:
    out: List[Any] = []
    seen = set()
    for src in (a or []), (b or []):
        for item in src:
            key = json.dumps(item, sort_keys=True) if isinstance(item, (dict, list)) else str(item)
            if key not in seen:
                seen.add(key)
                out.append(item)
    return out


def merge_core_and_user_strategy(user_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Layer core sales skills (global) with user-specific business strategy (tenant-scoped).

    Rules:
    - persona: user overrides core fields
    - scripts: user values take precedence; lists/dicts merged; strings prefer user if provided
    - objections (top-level): core + user (user overrides)
    - goals: union
    - products, fallback_scenarios: user if provided; else core
    """
    core = load_core_sales_strategy() or {}
    core_scripts = core.get("scripts") or {}
    user_scripts = user_payload.get("scripts") or {}

    merged_scripts: Dict[str, Any] = {}
    # Strings
    for key in ("greeting", "pitch"):
        val = user_scripts.get(key) or core_scripts.get(key)
        if val:
            merged_scripts[key] = val
    # Lists
    for key in ("faqs", "closing"):
        merged_scripts[key] = _merge_list(core_scripts.get(key), user_scripts.get(key))
    # Dict inside scripts (e.g., objections snippet)
    if isinstance(core_scripts.get("objections"), dict) or isinstance(user_scripts.get("objections"), dict):
        merged_scripts["objections"] = _merge_dict(core_scripts.get("objections") or {}, user_scripts.get("objections") or {})

    # Persona
    merged_persona = _merge_dict(core.get("persona") or {}, user_payload.get("persona") or {})

    # Goals
    merged_goals = _merge_list(core.get("goals"), user_payload.get("goals"))

    # Top-level objections
    merged_objections = _merge_dict(load_core_objections_map(), _merge_dict(core.get("objections") or {}, user_payload.get("objections") or {}))

    merged = dict(core)
    # Overlay user specifics
    merged.update(user_payload or {})
    # Force merged components
    merged["persona"] = merged_persona
    merged["scripts"] = merged_scripts
    if merged_goals:
        merged["goals"] = merged_goals
    if merged_objections:
        merged["objections"] = merged_objections
    # products, fallback_scenarios: user overrides already applied by merged.update(user_payload)
    return merged


async def fetch_latest_strategy(user_id: str) -> Optional[Dict[str, Any]]:
    """Fetch the most recently updated strategy for a user."""
    strategies_collection = get_collection(COLLECTION_STRATEGIES)
    cursor = strategies_collection.find({"user_id": user_id}).sort("updated_at", -1).limit(1)
    strategies = await cursor.to_list(length=1)
    return strategies[0] if strategies else None


TokenCallback = Optional[Callable[[str], Awaitable[None]]]


async def call_openai_chat(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    *,
    stream: bool = True,
    persona: Optional[Dict[str, Any]] = None,
    goals: Optional[Sequence[str]] = None,
    temperature: float = 0.7,
    on_token: TokenCallback = None,
) -> str:
    """Invoke GPT-4o chat completions with persona injection and optional token streaming."""
    if not settings.OPENAI_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OpenAI API key not configured",
        )

    model = model or settings.OPENAI_MODEL
    headers = {
        "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    augmented_messages = _inject_persona_context(messages, persona=persona, goals=goals)
    payload: Dict[str, Any] = {
        "model": model,
        "messages": augmented_messages,
        "temperature": temperature,
    }
    if stream:
        payload["stream"] = True

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            if stream:
                data_text = await _stream_chat_completion(
                    client,
                    headers=headers,
                    payload=payload,
                    on_token=on_token,
                )
                return data_text

            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=f"OpenAI request failed: {exc.response.text}",
        ) from exc

    choices = data.get("choices", [])
    if not choices:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="OpenAI did not return any choices",
        )

    content = choices[0]["message"]["content"].strip()
    if on_token:
        await _invoke_callback(on_token, content)
    return content


def _inject_persona_context(
    messages: List[Dict[str, str]],
    *,
    persona: Optional[Dict[str, Any]],
    goals: Optional[Sequence[str]],
) -> List[Dict[str, str]]:
    """Augment the first system message with persona and goal descriptors."""
    if not persona and not goals:
        return list(messages)

    persona_lines: List[str] = []
    if persona:
        name = persona.get("name")
        tone = persona.get("tone")
        description = persona.get("description")
        if name:
            persona_lines.append(f"Persona Name: {name}")
        if tone:
            persona_lines.append(f"Desired tone: {tone}")
        if description:
            persona_lines.append(description)
        voice = persona.get("voice_id")
        if voice:
            persona_lines.append(f"Voice preference: {voice}")
    if goals:
        persona_lines.append("Goals: " + "; ".join(goals))

    if not persona_lines:
        return list(messages)

    injected_block = "\n".join(persona_lines)
    if messages and messages[0].get("role") == "system":
        merged = messages[0].copy()
        merged["content"] = f"{merged['content']}\n{injected_block}"
        return [merged] + [msg.copy() for msg in messages[1:]]

    return [{"role": "system", "content": injected_block}] + [msg.copy() for msg in messages]


async def _stream_chat_completion(
    client: httpx.AsyncClient,
    *,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    on_token: TokenCallback = None,
) -> str:
    """Stream GPT-4o deltas, yielding tokens to the provided callback."""
    url = "https://api.openai.com/v1/chat/completions"
    collected: List[str] = []
    async with client.stream("POST", url, headers=headers, json=payload) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line:
                continue
            if line.startswith("data: "):
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                deltas = chunk.get("choices", [])
                if not deltas:
                    continue
                delta = deltas[0].get("delta", {})
                if not delta:
                    continue
                content_piece = delta.get("content")
                if content_piece:
                    collected.append(content_piece)
                    if on_token:
                        await _invoke_callback(on_token, content_piece)
    return "".join(collected).strip()


async def _invoke_callback(callback: TokenCallback, token: str) -> None:
    """Safely await token callbacks without propagating exceptions."""
    try:
        result = callback(token)
        if inspect.isawaitable(result):
            await result
    except Exception:  # pragma: no cover - callback safety
        pass


async def generate_strategy_context(strategy_payload: Dict[str, Any]) -> str:
    """Flatten relevant strategy sections into a prompt-friendly string (core + user)."""
    merged = merge_core_and_user_strategy(strategy_payload or {})
    lines: List[str] = []
    scripts = merged.get("scripts") or {}

    # Scripts summary
    for field in _STRATEGY_FIELDS_PRIORITY:
        value = merged.get(field) or scripts.get(field)
        if not value:
            continue
        if isinstance(value, (list, tuple)):
            joined = "; ".join(str(item) for item in value)
            lines.append(f"{field.title()}: {joined}")
        elif isinstance(value, dict):
            pairs = "; ".join(f"{k}: {v}" for k, v in value.items())
            lines.append(f"{field.title()}: {pairs}")
        else:
            lines.append(f"{field.title()}: {value}")

    # Products summary
    prods = merged.get("product_details") or merged.get("products") or []
    if isinstance(prods, list) and prods:
        try:
            sample = prods[0]
            name = sample.get("name")
            value = sample.get("value") or sample.get("benefits")
            if name:
                lines.append(f"Product: {name}")
            if value:
                if isinstance(value, list):
                    lines.append("Benefits: " + "; ".join(map(str, value)))
                else:
                    lines.append(f"Value: {value}")
        except Exception:
            pass

    # Audience / business info
    audience = merged.get("target_audience") or {}
    if isinstance(audience, dict):
        pains = audience.get("pain_points")
        if pains:
            pains_str = ", ".join(map(str, pains)) if isinstance(pains, list) else str(pains)
            lines.append(f"Audience Pain Points: {pains_str}")
    biz = merged.get("business_info") or {}
    if isinstance(biz, dict) and biz.get("company_name"):
        lines.append(f"Company: {biz.get('company_name')} — {biz.get('tagline') or ''}")

    if not lines:
        lines.append("Use general sales best practices.")

    return "\n".join(lines)


async def synthesise_elevenlabs_voice(text: str, voice_id: Optional[str]) -> Dict[str, Any]:
    """Call ElevenLabs text-to-speech API and return metadata."""
    if not settings.ELEVENLABS_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ElevenLabs API key not configured",
        )

    voice_id = voice_id or settings.ELEVENLABS_DEFAULT_VOICE_ID
    if not voice_id:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="ElevenLabs voice_id is required. Provide it in the request or set ELEVENLABS_DEFAULT_VOICE_ID.",
        )

    url = f"{settings.ELEVENLABS_BASE_URL.rstrip('/')}/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": settings.ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }
    payload = {
        "text": text,
        "model_id": settings.ELEVENLABS_MODEL or "eleven_monolingual_v1",
        "voice_settings": {"stability": 0.4, "similarity_boost": 0.8},
    }

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            audio_bytes = response.content
    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code
        message = exc.response.text or exc.response.reason_phrase
        if status_code == 404:
            detail = (
                f"ElevenLabs voice '{voice_id}' not found. Use a valid voice_id or update ELEVENLABS_DEFAULT_VOICE_ID."
            )
        elif status_code == 401:
            detail = "ElevenLabs authentication failed. Check ELEVENLABS_API_KEY."
        else:
            detail = f"ElevenLabs request failed ({status_code}): {message}"
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=detail) from exc

    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    duration = max(len(text.split()) * 0.4, 1.0)

    return {"audio_base64": audio_base64, "duration": duration}


async def list_elevenlabs_voices() -> List[Dict[str, Any]]:
    if not settings.ELEVENLABS_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ElevenLabs API key not configured",
        )

    url = f"{settings.ELEVENLABS_BASE_URL.rstrip('/')}/v1/voices"
    headers = {
        "xi-api-key": settings.ELEVENLABS_API_KEY,
        "Accept": "application/json",
    }
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()

    return data.get("voices", [])


def get_twilio_client() -> Client:
    if not (settings.TWILIO_ACCOUNT_SID and settings.TWILIO_AUTH_TOKEN and settings.TWILIO_PHONE_NUMBER):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Twilio configuration is incomplete",
        )
    return Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)


async def create_twilio_call(client: Client, to_phone: str, greeting: str) -> Dict[str, Any]:
    """Attempt to start a Twilio call returning call metadata."""
    try:
        
        if settings.API_BASE_URL:
            webhook_url = f"{settings.API_BASE_URL.rstrip('/')}/api/v1/integrations/telephony/twilio/voice"
        # elif settings.API_BASE_URL:
        #     webhook_url = f"{settings.API_BASE_URL.rstrip('/')}/api/v1/integrations/telephony/twilio/voice"
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Twilio call webhook URL not configured",
            )
        logger.info(
            "Twilio call create request",
            extra={"to": to_phone, "from": settings.TWILIO_PHONE_NUMBER, "webhook": webhook_url},
        )

        call = await asyncio.to_thread(
            client.calls.create,
            to=to_phone,
            from_=settings.TWILIO_PHONE_NUMBER,
            url=webhook_url,
        )
        payload = {
            "sid": call.sid,
            "status": getattr(call, "status", CALL_STATUS_IN_PROGRESS),
            "direction": getattr(call, "direction", None),
            "to": getattr(call, "to", to_phone),
            "from": getattr(call, "from_", settings.TWILIO_PHONE_NUMBER),
        }
        logger.info("Twilio call created", extra=payload)
        return payload
    except TwilioRestException as exc:
        logger.error(
            "Twilio REST error during call create",
            exc_info=True,
            extra={"to": to_phone, "msg": str(exc), "code": exc.code},
        )
        return {"sid": None, "status": CALL_STATUS_FAILED, "error": str(exc)}


async def save_call_record(
    user_id: str,
    agent_id: str,
    lead_name: str,
    lead_phone: str,
    call_sid: Optional[str],
    status_label: str,
    lead_status: str,
    conversation: List[Dict[str, Any]],
    key_phrases: List[str],
    failure_reason: Optional[str],
    audio_urls: Optional[List[str]] = None,
) -> str:
    calls_collection = get_collection(COLLECTION_CALLS)
    call_entry = {
        "call_id": call_sid or f"manual-{datetime.utcnow().timestamp()}",
        "user_id": user_id,
        "agent_id": agent_id,
        "lead_name": lead_name,
        "lead_phone": lead_phone,
        "status": status_label,
        "lead_status": lead_status,
        "conversation": conversation,
        "key_phrases": key_phrases,
        "failure_reason": failure_reason,
        "audio_urls": audio_urls or [],
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }
    await calls_collection.insert_one(call_entry)
    return call_entry["call_id"]
