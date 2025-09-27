"""Service helpers for sales agent interactions."""
from __future__ import annotations

import asyncio
import base64
import inspect
import json
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

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

_STRATEGY_FIELDS_PRIORITY: Sequence[str] = (
    "greeting",
    "pitch",
    "faqs",
    "objections",
    "closing",
)


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
    stream: bool = False,
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
    """Flatten relevant strategy sections into a prompt-friendly string."""
    lines: List[str] = []
    scripts = strategy_payload.get("scripts") or {}

    for field in _STRATEGY_FIELDS_PRIORITY:
        value = strategy_payload.get(field) or scripts.get(field)
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
        webhook_url = settings.TWILIO_CALL_WEBHOOK_URL
        if not webhook_url:
            # Fallback to public base + Flask voice route if provided
            if settings.TWILIO_PUBLIC_BASE_URL:
                webhook_url = f"{settings.TWILIO_PUBLIC_BASE_URL.rstrip('/')}/twilio/voice"
            elif settings.API_BASE_URL:
                webhook_url = f"{settings.API_BASE_URL.rstrip('/')}/api/v1/integrations/telephony/twilio/voice"
            else:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Twilio call webhook URL not configured",
                )

        call = await asyncio.to_thread(
            client.calls.create,
            to=to_phone,
            from_=settings.TWILIO_PHONE_NUMBER,
            url=webhook_url,
        )
        return {"sid": call.sid, "status": getattr(call, "status", CALL_STATUS_IN_PROGRESS)}
    except TwilioRestException as exc:
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
