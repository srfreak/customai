"""Service helpers for sales agent interactions."""
from __future__ import annotations

import asyncio
import base64
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

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


async def call_openai_chat(messages: List[Dict[str, str]], model: Optional[str] = None) -> str:
    """Invoke OpenAI chat completion API and return the assistant message."""
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
    payload = {"model": model, "messages": messages, "temperature": 0.7}

    try:
        async with httpx.AsyncClient(timeout=40) as client:
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

    return choices[0]["message"]["content"].strip()


async def generate_strategy_context(strategy_payload: Dict[str, Any]) -> str:
    """Build a textual description of the strategy to feed into LLM."""
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
