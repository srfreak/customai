from __future__ import annotations

import base64
from datetime import datetime
from typing import List, Optional

import httpx

from apps.agents.sales.agent import SalesAgent
from core.config import settings
from .call_handler import ConversationTurn


async def run_simulated_dialogue(
    agent: SalesAgent,
    lead_name: str,
    voice_id: Optional[str],
    audio_urls: Optional[List[str]] = None,
    voice_failures: Optional[List[str]] = None,
) -> List[ConversationTurn]:
    """Generate a short scripted conversation using stored strategy."""

    conversation: List[ConversationTurn] = []
    audio_urls = audio_urls if audio_urls is not None else []
    voice_failures = voice_failures if voice_failures is not None else []
    generated_prompts = {
        "greeting": lead_name and f"Hi, this is {lead_name}." or "Hello!",
        "pitch": "Can you tell me more?",
        "faqs": "I have a quick question.",
        "objections": "I'm not sure this fits my budget.",
        "closing": "What happens next?",
    }

    for stage in ("greeting", "pitch", "faqs", "objections", "closing"):
        lead_input = generated_prompts.get(stage, "")
        if stage == "greeting" and not lead_input:
            lead_input = "Lead answers the call."

        sales_response = await agent.generate_sales_response(
            user_input=lead_input or "Start the conversation",
            stage=stage,
            lead_name=lead_name,
        )
        agent_reply = sales_response["text"]
        voice_payload = await agent.speak_response(agent_reply, voice_id=voice_id)

        audio_b64 = voice_payload.get("audio_base64")
        upload_endpoint = settings.AUDIO_UPLOAD_URL or (
            f"{settings.TWILIO_PUBLIC_BASE_URL.rstrip('/')}/audio/upload"
            if settings.TWILIO_PUBLIC_BASE_URL
            else ""
        )

        if audio_b64 and upload_endpoint:
            try:
                audio_bytes = base64.b64decode(audio_b64)
                async with httpx.AsyncClient(timeout=30) as client:
                    files = {
                        "file": (
                            f"clip_{datetime.utcnow().timestamp():.0f}.mp3",
                            audio_bytes,
                            "audio/mpeg",
                        )
                    }
                    upload_response = await client.post(upload_endpoint, files=files)
                    upload_response.raise_for_status()
                    upload_data = upload_response.json()
                    public_url = upload_data.get("url")
                    if public_url:
                        audio_urls.append(public_url)
                    else:
                        voice_failures.append("Audio upload succeeded without URL response")
            except (httpx.HTTPError, ValueError) as exc:
                voice_failures.append(f"Audio upload failed: {exc}")
        elif audio_b64:
            voice_failures.append("Audio upload endpoint not configured")

        conversation.append(
            ConversationTurn(
                stage=stage,
                lead_input=lead_input or "",
                agent_reply=agent_reply,
                audio_base64=voice_payload.get("audio_base64"),
                duration=voice_payload.get("duration"),
            )
        )

        if stage == "closing":
            break

    return conversation
