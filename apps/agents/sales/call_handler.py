from __future__ import annotations

import logging
import base64
from datetime import datetime
from typing import List, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from apps.agents.sales import services
from apps.agents.sales.agent import SalesAgent, STAGE_SEQUENCE
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
    logger.info(
        "Persisted conversation transcript",
        extra={
            "call_id": call_id,
            "turns": len(conversation),
            "agent_id": agent_id,
        },
    )


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

    logger.info(
        "Fetched latest strategy",
        extra={
            "user_id": user["user_id"],
            "strategy_id": strategy.get("strategy_id"),
            "agent_id": request.agent_id,
        },
    )

    strategy_payload = strategy.get("payload", {})
    persona_payload = strategy_payload.get("persona") or {}
    agent_identifier = (
        request.agent_id
        or strategy.get("agent_id")
        or strategy_payload.get("agent_id")
        or "sales-agent"
    )
    agent_name = persona_payload.get("name") or strategy_payload.get("title") or "Scriza Sales Partner"
    sales_agent = SalesAgent(
        agent_id=agent_identifier,
        user_id=user["user_id"],
        name=agent_name,
        persona=persona_payload,
    )
    sales_agent.attach_strategy(strategy_payload)
    goals = strategy_payload.get("goals")
    if isinstance(goals, list):
        sales_agent.set_goals(goals)
    voice_id = request.voice_id or sales_agent.voice_id or settings.ELEVENLABS_DEFAULT_VOICE_ID or None
    greeting = strategy_payload.get("greeting") or "Hello, this is Scriza AI calling."

    twilio_call_sid = None
    call_status = CALL_STATUS_IN_PROGRESS

    failure_reason: Optional[str] = None
    call_reference = sales_agent.conversation_id
    logger.info(
        "start_call invoked",
        extra={
            "lead_phone": request.lead_phone,
            "lead_name": request.lead_name,
            "agent_id": agent_identifier,
        },
    )

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
        if twilio_call_sid:
            sales_agent.conversation_id = twilio_call_sid
            call_reference = twilio_call_sid
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
    if call_status != CALL_STATUS_FAILED and settings.SIMULATE_CALL_FLOW:
        strategy_context = await services.generate_strategy_context(strategy_payload)
        generated_lead_prompts = {
            "greeting": request.lead_name and f"Hi, this is {request.lead_name}." or "Hello!",
            "pitch": "Can you tell me more?",
            "faqs": "I have a quick question.",
            "objections": "I'm not sure this fits my budget.",
            "closing": "What happens next?",
        }

        for current_stage in STAGE_SEQUENCE:
            lead_input = generated_lead_prompts.get(current_stage, "")
            if current_stage == "greeting" and not lead_input:
                lead_input = "Lead answers the call."

            try:
                response_payload = await sales_agent.generate_sales_response(
                    user_input=lead_input or "Start the conversation",
                    stage=current_stage,
                    lead_name=request.lead_name,
                )
            except HTTPException as chat_exc:
                failure_reason = chat_exc.detail if isinstance(chat_exc.detail, str) else str(chat_exc.detail)
                logger.error("Failed to generate agent response: %s", failure_reason)
                call_status = CALL_STATUS_FAILED
                break

            agent_reply = response_payload["text"]
            logger.info(
                "Generated agent reply",
                extra={
                    "stage": current_stage,
                    "lead_input": lead_input,
                    "reply_excerpt": agent_reply[:120],
                    "memory_id": response_payload.get("memory_id"),
                },
            )

            try:
                logger.info(
                    "Submitting text to ElevenLabs",
                    extra={"stage": current_stage, "voice_id": voice_id},
                )
                voice_payload = await sales_agent.speak_response(agent_reply, voice_id=voice_id)
                logger.info(
                    "Received ElevenLabs response",
                    extra={
                        "stage": current_stage,
                        "has_audio": bool(voice_payload.get("audio_base64")),
                        "duration": voice_payload.get("duration"),
                    },
                )
            except HTTPException as synth_exc:
                detail = synth_exc.detail if isinstance(synth_exc.detail, str) else str(synth_exc.detail)
                voice_failures.append(detail)
                logger.error("Voice synthesis failed: %s", detail)
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
                                logger.info(
                                    "Uploaded audio clip",
                                    extra={
                                        "stage": current_stage,
                                        "public_url": public_url,
                                        "upload_endpoint": upload_endpoint,
                                    },
                                )
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
                    agent_reply=agent_reply,
                    audio_base64=voice_payload.get("audio_base64"),
                    duration=voice_payload.get("duration"),
                )
            )
            logger.info(
                "Appended conversation turn",
                extra={
                    "stage": current_stage,
                    "lead_input_excerpt": (lead_input or "")[:80],
                    "reply_excerpt": agent_reply[:120],
                },
            )

            if current_stage == "closing":
                break
    elif call_status != CALL_STATUS_FAILED:
        logger.info(
            "Live call mode active; waiting for Twilio stream events for lead input",
            extra={"call_sid": twilio_call_sid, "conversation_id": sales_agent.conversation_id},
        )

    key_phrases = _extract_key_phrases(conversation)
    logger.info(
        "Extracted key phrases",
        extra={"phrases": key_phrases, "count": len(key_phrases)},
    )
    status_label = CALL_STATUS_COMPLETED if call_status != CALL_STATUS_FAILED else CALL_STATUS_FAILED
    lead_status = "hung_up" if call_status == CALL_STATUS_FAILED else "interested"
    if call_status == CALL_STATUS_FAILED and not failure_reason:
        failure_reason = "Unknown error"
    if voice_failures and not failure_reason:
        failure_reason = "; ".join(voice_failures)
    call_id = await services.save_call_record(
        user_id=user["user_id"],
        agent_id=agent_identifier,
        lead_name=request.lead_name,
        lead_phone=request.lead_phone,
        call_sid=call_reference,
        status_label=status_label,
        lead_status=lead_status,
        conversation=[turn.dict() for turn in conversation],
        key_phrases=key_phrases,
        failure_reason=failure_reason,
        audio_urls=audio_urls,
    )

    logger.info(
        "Persisted call record",
        extra={
            "call_id": call_id,
            "status": status_label,
            "lead_status": lead_status,
            "conversation_turns": len(conversation),
            "audio_urls_count": len(audio_urls),
            "failure_reason": failure_reason,
        },
    )

    excel_path = log_call_summary(
        directory=settings.CALL_LOG_DIR,
        user_id=user["user_id"],
        agent_id=agent_identifier,
        lead_name=request.lead_name,
        lead_phone=request.lead_phone,
        call_status=status_label,
        lead_status=lead_status,
        failure_reason=failure_reason,
        key_phrases=key_phrases,
        call_id=call_id,
    )

    logger.info("Generated Excel log", extra={"excel_path": excel_path, "call_id": call_id})

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
        agent_id=agent_identifier,
        call_id=call_id,
        conversation=conversation,
    )

    logger.info(
        "start_call completed",
        extra={
            "call_id": call_id,
            "status": status_label,
            "final_voice_failures": voice_failures,
            "twilio_sid": twilio_call_sid,
        },
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
            "strategy_context": strategy_context,
            "conversation_id": sales_agent.conversation_id,
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
