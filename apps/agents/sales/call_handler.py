from __future__ import annotations

import logging
import base64
from datetime import datetime
from typing import List, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from apps.agents.sales import services
from core.auth import RoleChecker
from core.config import settings
from core.database import get_collection
from shared.constants import (
    COLLECTION_MEMORY_LOGS,
    CALL_STATUS_COMPLETED,
    CALL_STATUS_FAILED,
    CALL_STATUS_IN_PROGRESS,
)
from shared.exceptions import TelephonyException
from utils.excel_logger import log_call_summary

router = APIRouter()

logger = logging.getLogger(__name__)


class ConversationTurn(BaseModel):
    stage: str
    lead_input: str
    agent_reply: str
    audio_base64: Optional[str] = None
    duration: Optional[float] = None


class StartCallRequest(BaseModel):
    lead_phone: str = Field(..., example="+911234567890")
    lead_name: str = Field(..., example="Ravi Kumar")
    agent_id: Optional[str] = None
    voice_id: Optional[str] = None


class StartCallResponse(BaseModel):
    call_id: str
    status: str
    lead_status: str
    lead_name: str
    lead_phone: str
    conversation: List[ConversationTurn]
    excel_log_path: Optional[str] = None
    debug: Optional[dict] = None
    audio_urls: List[str] = Field(default_factory=list)


async def _persist_memory(
    user_id: str,
    agent_id: Optional[str],
    call_id: str,
    conversation: List[ConversationTurn],
) -> None:
    memory_collection = get_collection(COLLECTION_MEMORY_LOGS)
    entry = {
        "memory_id": f"call-{call_id}",
        "user_id": user_id,
        "agent_id": agent_id,
        "data": {
            "type": "call_conversation",
            "call_id": call_id,
            "conversation": [turn.dict() for turn in conversation],
        },
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }
    await memory_collection.insert_one(entry)


def _extract_key_phrases(conversation: List[ConversationTurn]) -> List[str]:
    phrases: List[str] = []
    for turn in conversation:
        phrases.extend(word.strip() for word in turn.agent_reply.split(".") if word.strip())
    return phrases[:5]


@router.post("/start_call", response_model=StartCallResponse)
async def start_call(
    request: StartCallRequest,
    user: dict = Depends(RoleChecker(["user", "admin"])),
):
    """Initiate a Twilio call and orchestrate AI-driven conversation simulation."""
    strategy = await services.fetch_latest_strategy(user_id=user["user_id"])
    if not strategy:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No strategy found for user. Upload a strategy first.",
        )

    voice_id = request.voice_id or settings.ELEVENLABS_DEFAULT_VOICE_ID or None
    greeting = strategy.get("payload", {}).get("greeting") or "Hello, this is Scriza AI calling."

    twilio_call_sid = None
    call_status = CALL_STATUS_IN_PROGRESS

    failure_reason: Optional[str] = None

    try:
        client = services.get_twilio_client()
        call_meta = await services.create_twilio_call(
            client=client,
            to_phone=request.lead_phone,
            greeting=greeting,
        )
        twilio_call_sid = call_meta.get("sid")
        call_status = call_meta.get("status", CALL_STATUS_IN_PROGRESS)
        if call_meta.get("error"):
            failure_reason = call_meta.get("error")
            logger.warning("Twilio call initiation failed: %s", failure_reason)
            raise TelephonyException(call_meta["error"])
        logger.info(
            "Started Twilio call sid=%s status=%s lead=%s",
            twilio_call_sid,
            call_status,
            request.lead_phone,
        )
    except TelephonyException as exc:
        call_status = CALL_STATUS_FAILED
        failure_reason = str(exc)
        logger.error("TelephonyException while starting call: %s", failure_reason)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive logging
        call_status = CALL_STATUS_FAILED
        failure_reason = str(exc)
        logger.exception("Unexpected error when starting call")

    conversation: List[ConversationTurn] = []
    strategy_context: Optional[str] = None
    voice_failures: List[str] = []
    audio_urls: List[str] = []
    if call_status != CALL_STATUS_FAILED:
        current_stage = "greeting"
        lead_input = request.lead_name and f"Lead {request.lead_name} answers the call." or ""
        strategy_context = await services.generate_strategy_context(strategy.get("payload", {}))

        # Simulate a short conversational flow of up to 4 turns
        for _ in range(4):
            reply_response = await services.call_openai_chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are a friendly sales representative. Follow the provided strategy carefully.\n"
                            f"{strategy_context}"
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Stage: {current_stage}. Lead prompt: {lead_input or 'Start the conversation.'}",
                    },
                ]
            )

            try:
                voice_payload = await services.synthesise_elevenlabs_voice(
                    text=reply_response,
                    voice_id=voice_id,
                )
            except HTTPException as synth_exc:
                voice_failures.append(
                    synth_exc.detail if isinstance(synth_exc.detail, str) else str(synth_exc.detail)
                )
                logger.error("Voice synthesis failed: %s", voice_failures[-1])
                voice_payload = {"audio_base64": None, "duration": None}
            else:
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
                                logger.warning("Audio upload missing URL: %s", upload_data)
                    except (httpx.HTTPError, ValueError) as upload_exc:
                        logger.error("Audio upload failed: %s", upload_exc)
                        voice_failures.append(f"Audio upload failed: {upload_exc}")
                elif audio_b64:
                    message = "Audio upload endpoint not configured"
                    voice_failures.append(message)
                    logger.warning(message)
            conversation.append(
                ConversationTurn(
                    stage=current_stage,
                    lead_input=lead_input or "",
                    agent_reply=reply_response,
                    audio_base64=voice_payload["audio_base64"],
                    duration=voice_payload["duration"],
                )
            )
            if current_stage == "closing":
                break
            if current_stage == "greeting":
                current_stage = "pitch"
            elif current_stage == "pitch":
                current_stage = "faqs"
            elif current_stage == "faqs":
                current_stage = "objections"
            else:
                current_stage = "closing"
            lead_input = "Acknowledged."

    key_phrases = _extract_key_phrases(conversation)
    status_label = CALL_STATUS_COMPLETED if call_status != CALL_STATUS_FAILED else CALL_STATUS_FAILED
    lead_status = "hung_up" if call_status == CALL_STATUS_FAILED else "interested"
    if call_status == CALL_STATUS_FAILED and not failure_reason:
        failure_reason = "Unknown error"
    if voice_failures and not failure_reason:
        failure_reason = "; ".join(voice_failures)
    call_id = await services.save_call_record(
        user_id=user["user_id"],
        agent_id=request.agent_id or "default-agent",
        lead_name=request.lead_name,
        lead_phone=request.lead_phone,
        call_sid=twilio_call_sid,
        status_label=status_label,
        lead_status=lead_status,
        conversation=[turn.dict() for turn in conversation],
        key_phrases=key_phrases,
        failure_reason=failure_reason,
        audio_urls=audio_urls,
    )

    excel_path = log_call_summary(
        directory=settings.CALL_LOG_DIR,
        user_id=user["user_id"],
        agent_id=request.agent_id or "default-agent",
        lead_name=request.lead_name,
        lead_phone=request.lead_phone,
        call_status=status_label,
        lead_status=lead_status,
        failure_reason=failure_reason,
        key_phrases=key_phrases,
        call_id=call_id,
    )

    if twilio_call_sid and audio_urls:
        try:
            twiml = "<Response>" + "".join(
                f"<Play>{url}</Play>" for url in audio_urls
            ) + "</Response>"
            services.get_twilio_client().calls(twilio_call_sid).update(twiml=twiml)
        except Exception as update_exc:  # pragma: no cover
            logger.exception("Failed to enqueue audio playback on Twilio call: %s", update_exc)

    await _persist_memory(
        user_id=user["user_id"],
        agent_id=request.agent_id or "default-agent",
        call_id=call_id,
        conversation=conversation,
    )

    return StartCallResponse(
        call_id=call_id,
        status=status_label,
        lead_name=request.lead_name,
        lead_phone=request.lead_phone,
        conversation=conversation,
        excel_log_path=excel_path,
        lead_status=lead_status,
        debug={
            "twilio_sid": twilio_call_sid,
            "call_status": call_status,
            "failure_reason": failure_reason,
            "voice_failures": voice_failures,
        },
        audio_urls=audio_urls,
    )




@router.post("/handle_incoming_call")
async def handle_incoming_call(
    call_data: dict,
    user: dict = Depends(RoleChecker(["user", "admin"]))
):
    """
    Handle an incoming call
    
    Args:
        call_data: Data about the incoming call
        user: Authenticated user
        
    Returns:
        Dict with call handling instructions
    """
    try:
        # In a real implementation, this would handle the incoming call
        # For now, we'll just return a mock response
        return {
            "status": "success",
            "message": "Incoming call received",
            "call_data": call_data
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to handle incoming call: {str(e)}"
        )

@router.get("/call_status/{call_id}")
async def get_call_status(
    call_id: str,
    user: dict = Depends(RoleChecker(["user", "admin"]))
):
    """
    Get call status
    
    Args:
        call_id: Call ID
        user: Authenticated user
        
    Returns:
        Dict with call status
    """
    try:
        calls_collection = get_collection(COLLECTION_CALLS)
        call_record = await calls_collection.find_one({"call_id": call_id})
        
        if not call_record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Call not found"
            )
        
        return {
            "status": "success",
            "call_data": call_record
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get call status: {str(e)}"
        )
