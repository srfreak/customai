import asyncio
import base64
import json
import logging
import subprocess
from typing import Optional, Dict, Any, List
import time
import io
import wave
import audioop

from fastapi import APIRouter, HTTPException, status, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from pydantic import BaseModel
from core.auth import RoleChecker
from core.config import settings
from shared.exceptions import TelephonyException
from apps.agents.sales import services
from apps.agents.sales.agent import SalesAgent
import httpx
import binascii
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse
from core.database import get_collection
from shared.constants import COLLECTION_CALLS
from datetime import datetime

router = APIRouter()

logger = logging.getLogger(__name__)
# Ensure conversation logs appear in container logs even if root logger is WARNING
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)
logger.propagate = True

class CallRequest(BaseModel):
    """Call request model"""
    to_number: str
    from_number: Optional[str] = None
    agent_id: str
    callback_url: Optional[str] = None

class CallResponse(BaseModel):
    """Call response model"""
    call_sid: str
    status: str
    message: str

class TwilioService:
    """Twilio service for telephony operations"""
    
    def __init__(self):
        self.client = Client(
            settings.TWILIO_ACCOUNT_SID,
            settings.TWILIO_AUTH_TOKEN
        ) if settings.TWILIO_ACCOUNT_SID and settings.TWILIO_AUTH_TOKEN else None
    
    async def make_call(
        self,
        to_number: str,
        from_number: str,
        agent_id: str,
        callback_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Make a call using Twilio
        
        Args:
            to_number: Number to call
            from_number: Number to call from
            agent_id: Agent ID
            callback_url: Callback URL for call events
            
        Returns:
            Dict with call details
        """
        try:
            if not self.client:
                # Return mock response if Twilio is not configured
                return {
                    "call_sid": "mock_call_sid",
                    "status": "initiated",
                    "message": "Call initiated (mock)"
                }
            
            # Make the call
            # Determine webhook URL preference order
            if callback_url:
                webhook = f"{callback_url.rstrip('/')}/voice"
            elif settings.API_BASE_URL:
                webhook = f"{settings.API_BASE_URL.rstrip('/')}/api/v1/integrations/telephony/twilio/voice"
            elif settings.TWILIO_CALL_WEBHOOK_URL:
                webhook = settings.TWILIO_CALL_WEBHOOK_URL
            
            elif settings.TWILIO_PUBLIC_BASE_URL:
                webhook = f"{settings.TWILIO_PUBLIC_BASE_URL.rstrip('/')}/twilio/voice"
            
            else:
                raise TelephonyException("No webhook base configured for Twilio voice callback")

            call = self.client.calls.create(
                to=to_number,
                from_=from_number,
                url=webhook,
                method="POST",
            )
            
            return {
                "call_sid": call.sid,
                "status": call.status,
                "message": "Call initiated successfully"
            }
        except Exception as e:
            raise TelephonyException(f"Failed to make call: {str(e)}")
    
    async def send_sms(
        self,
        to_number: str,
        from_number: str,
        message: str
    ) -> Dict[str, Any]:
        """
        Send SMS using Twilio
        
        Args:
            to_number: Number to send SMS to
            from_number: Number to send SMS from
            message: SMS message
            
        Returns:
            Dict with SMS details
        """
        try:
            if not self.client:
                # Return mock response if Twilio is not configured
                return {
                    "message_sid": "mock_message_sid",
                    "status": "sent",
                    "message": "SMS sent (mock)"
                }
            
            # Send the SMS
            sms = self.client.messages.create(
                to=to_number,
                from_=from_number,
                body=message
            )
            
            return {
                "message_sid": sms.sid,
                "status": sms.status,
                "message": "SMS sent successfully"
            }
        except Exception as e:
            raise TelephonyException(f"Failed to send SMS: {str(e)}")

# Initialize Twilio service
twilio_service = TwilioService()

@router.post("/call", response_model=CallResponse)
async def make_call(
    call_request: CallRequest,
    user: dict = Depends(RoleChecker(["user", "admin"]))
):
    """
    Make a call using Twilio
    
    Args:
        call_request: Call request data
        user: Authenticated user
        
    Returns:
        CallResponse
    """
    try:
        result = await twilio_service.make_call(
            to_number=call_request.to_number,
            from_number=call_request.from_number or settings.TWILIO_PHONE_NUMBER,
            agent_id=call_request.agent_id,
            callback_url=call_request.callback_url
        )
        
        return CallResponse(
            call_sid=result["call_sid"],
            status=result["status"],
            message=result["message"]
        )
    except TelephonyException as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/sms")
async def send_sms(
    to_number: str,
    message: str,
    from_number: Optional[str] = None,
    user: dict = Depends(RoleChecker(["user", "admin"]))
):
    """
    Send SMS using Twilio
    
    Args:
        to_number: Number to send SMS to
        message: SMS message
        from_number: Number to send SMS from
        user: Authenticated user
        
    Returns:
        Dict with SMS result
    """
    try:
        result = await twilio_service.send_sms(
            to_number=to_number,
            from_number=from_number or settings.TWILIO_PHONE_NUMBER,
            message=message
        )
        
        return {
            "status": "success",
            "message_sid": result["message_sid"],
            "status": result["status"],
            "message": result["message"]
        }
    except TelephonyException as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

def _resolve_stream_url() -> str:
    if settings.TWILIO_STREAM_URL:
        return settings.TWILIO_STREAM_URL
    if not settings.TWILIO_PUBLIC_BASE_URL:
        raise TelephonyException("TWILIO_STREAM_URL or TWILIO_PUBLIC_BASE_URL must be configured")
    base = settings.TWILIO_PUBLIC_BASE_URL.rstrip("/")
    return f"{base}/api/v1/integrations/telephony/twilio/stream"


@router.post("/voice", include_in_schema=False)
async def handle_voice_call():
    """
    Handle incoming voice call
    
    Returns:
        TwiML response
    """
    try:
        # Create TwiML response
        response = VoiceResponse()
        connect = response.connect()
        stream_url = _resolve_stream_url()
        # Do not specify track to avoid 31941 errors; defaults to inbound audio
        connect.stream(url=stream_url)
        logger.info("Issued TwiML stream to %s", stream_url)
        return Response(content=str(response), media_type="application/xml")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to handle voice call: {str(e)}"
        )


def _mp3_to_mulaw_chunks(mp3_b64: str, chunk_ms: int = 20) -> List[str]:
    try:
        mp3_bytes = base64.b64decode(mp3_b64)
    except (base64.binascii.Error, ValueError) as exc:  # pragma: no cover - defensive
        raise TelephonyException(f"Invalid ElevenLabs audio payload: {exc}") from exc

    try:
        process = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                "pipe:0",
                "-ac",
                "1",
                "-ar",
                "8000",
                "-f",
                "mulaw",
                "pipe:1",
            ],
            input=mp3_bytes,
            stdout=subprocess.PIPE,
            check=True,
        )
    except FileNotFoundError as exc:
        raise TelephonyException("ffmpeg is required for audio conversion but was not found") from exc
    except subprocess.CalledProcessError as exc:  # pragma: no cover - logged for troubleshooting
        raise TelephonyException(f"ffmpeg failed to transcode audio: {exc}") from exc

    mulaw_data = process.stdout
    frame_size = int(8000 * chunk_ms / 1000)  # 160 bytes for 20 ms at 8 kHz
    chunks: List[str] = []
    for index in range(0, len(mulaw_data), frame_size):
        frame = mulaw_data[index:index + frame_size]
        if len(frame) == frame_size:
            chunks.append(base64.b64encode(frame).decode())
    return chunks


async def _send_audio_prompt(websocket: WebSocket, stream_sid: str, text: str, voice_id: Optional[str]) -> None:
    try:
        voice_payload = await services.synthesise_elevenlabs_voice(text=text, voice_id=voice_id)
    except HTTPException as exc:
        logger.error("Failed to synthesise greeting: %s", exc.detail)
        return

    try:
        chunks = _mp3_to_mulaw_chunks(voice_payload["audio_base64"])
    except TelephonyException as exc:
        logger.error("Audio conversion failed: %s", exc)
        return

    for chunk in chunks:
        await websocket.send_text(
            json.dumps(
                {
                    "event": "media",
                    "streamSid": stream_sid,
                    "media": {"payload": chunk},
                }
            )
        )
        await asyncio.sleep(0.02)

    await websocket.send_text(
        json.dumps(
            {
                "event": "mark",
                "streamSid": stream_sid,
                "mark": {"name": "greeting_complete"},
            }
        )
    )


def _mulaw_to_wav(mulaw_bytes: bytes) -> bytes:
    """Decode raw 8kHz μ-law mono into a WAV container (pure Python)."""
    # Convert μ-law to 16-bit linear PCM using audioop
    pcm16 = audioop.ulaw2lin(mulaw_bytes, 2)  # 2 bytes/sample
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(8000)
        wf.writeframes(pcm16)
    return buf.getvalue()


async def _transcribe_with_openai(wav_bytes: bytes) -> str:
    """Send a small WAV chunk to OpenAI Whisper for transcription."""
    if not settings.OPENAI_API_KEY:
        return ""
    headers = {"Authorization": f"Bearer {settings.OPENAI_API_KEY}"}
    files = {
        "file": ("chunk.wav", wav_bytes, "audio/wav"),
    }
    data = {
        "model": "whisper-1",
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.openai.com/v1/audio/transcriptions", headers=headers, files=files, data=data
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("text", "").strip()


async def _speak_and_stream(websocket: WebSocket, stream_sid: str, text: str, voice_id: Optional[str]) -> None:
    """TTS via ElevenLabs and stream μ-law frames back to Twilio."""
    try:
        tts = await services.synthesise_elevenlabs_voice(text=text, voice_id=voice_id)
        chunks = _mp3_to_mulaw_chunks(tts["audio_base64"])
    except HTTPException as exc:
        logger.error("TTS failed: %s", exc.detail)
        return
    for chunk in chunks:
        try:
            await websocket.send_text(
                json.dumps({"event": "media", "streamSid": stream_sid, "media": {"payload": chunk}})
            )
        except Exception:
            break
        await asyncio.sleep(0.02)


async def _stream_audio_b64(websocket: WebSocket, stream_sid: str, audio_b64: str) -> None:
    try:
        chunks = _mp3_to_mulaw_chunks(audio_b64)
    except TelephonyException as exc:
        logger.error("Audio conversion failed: %s", exc)
        return
    for chunk in chunks:
        try:
            await websocket.send_text(
                json.dumps({"event": "media", "streamSid": stream_sid, "media": {"payload": chunk}})
            )
        except Exception:
            break
        await asyncio.sleep(0.02)


@router.websocket("/stream")
async def media_stream_endpoint(websocket: WebSocket):
    await websocket.accept()
    stream_sid: Optional[str] = None
    greeting_task: Optional[asyncio.Task] = None
    # Simple rolling μ-law buffer and last-transcript tracking
    inbound_buffer = bytearray()
    last_transcript = ""
    processing = False
    # Collect conversational turns for end-of-call summary
    conversation: List[Dict[str, Any]] = []
    sales_agent: Optional[SalesAgent] = None
    lead_name: Optional[str] = None

    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            event = payload.get("event")

            if event == "start":
                stream_info = payload.get("start", {})
                stream_sid = stream_info.get("streamSid")
                call_sid = stream_info.get("callSid")
                logger.info("Twilio stream started: %s", stream_sid)
                # Initialize SalesAgent bound to this call (strategy/persona aware)
                try:
                    call_doc = None
                    if call_sid:
                        calls_col = get_collection(COLLECTION_CALLS)
                        call_doc = await calls_col.find_one({"call_id": call_sid})
                    user_id = (call_doc or {}).get("user_id") or "unknown-user"
                    agent_id = (call_doc or {}).get("agent_id") or "sales-agent"
                    lead_name = (call_doc or {}).get("lead_name")
                    strategy_doc = await services.fetch_latest_strategy(user_id=user_id)
                    strategy_payload = (strategy_doc or {}).get("payload", {})
                    persona_payload = strategy_payload.get("persona") or {}
                    sales_agent = SalesAgent(
                        agent_id=agent_id,
                        user_id=user_id,
                        name=persona_payload.get("name") or "Sales Agent",
                        persona=persona_payload,
                    )
                    sales_agent.attach_strategy(strategy_payload)
                    goals = strategy_payload.get("goals")
                    if isinstance(goals, list):
                        sales_agent.set_goals(goals)
                except Exception as exc:
                    logger.error("Failed to init SalesAgent for stream %s: %s", stream_sid, exc)
                greeting_text = "Hello! This is Scriza AI. Thanks for taking the call."  # TODO personalise
                greeting_task = asyncio.create_task(
                    _send_audio_prompt(
                        websocket=websocket,
                        stream_sid=stream_sid,
                        text=greeting_text,
                        voice_id=settings.ELEVENLABS_DEFAULT_VOICE_ID or None,
                    )
                )
            elif event == "media":
                media = payload.get("media", {})
                b64 = media.get("payload")
                if not b64:
                    continue
                # Ensure b64 is str, decode to bytes
                try:
                    if isinstance(b64, str):
                        decoded = base64.b64decode(b64, validate=False)
                    else:
                        # Unexpected type; skip
                        continue
                    if not isinstance(decoded, (bytes, bytearray)):
                        continue
                    inbound_buffer += decoded
                except (binascii.Error, ValueError) as exc:
                    # Corrupt frame; skip silently
                    continue

                # If we have ~1s of audio (8000 bytes) and not already processing, transcribe
                if not processing and len(inbound_buffer) >= 8000:
                    processing = True
                    # Take up to 2 seconds worth to reduce latency
                    take = min(len(inbound_buffer), 16000)
                    chunk = bytes(inbound_buffer[:take])
                    del inbound_buffer[:take]

                    # Decode μ-law to WAV and transcribe
                    try:
                        if isinstance(chunk, str):
                            # Shouldn't happen, but guard against wrong type
                            chunk = chunk.encode("latin1", errors="ignore")
                        wav = _mulaw_to_wav(chunk)
                        transcript = await _transcribe_with_openai(wav)
                    except Exception as exc:
                        logger.error("Transcription error: %s", exc)
                        transcript = ""

                    if transcript and transcript != last_transcript:
                        last_transcript = transcript
                        logger.info("Caller said: %s", transcript)
                        # Generate reply via SalesAgent (strategy/persona aware) with fallback
                        reply = ""
                        try:
                            if sales_agent:
                                gen = await sales_agent.generate_sales_response(
                                    user_input=transcript,
                                    stage=None,
                                    lead_name=lead_name,
                                )
                                reply = gen.get("text", "")
                            else:
                                reply = await services.call_openai_chat(
                                    [
                                        {"role": "system", "content": "You are Scriza Sales AI; reply concisely and helpfully."},
                                        {"role": "user", "content": transcript},
                                    ],
                                    model=settings.OPENAI_MODEL,
                                )
                        except HTTPException as exc:
                            logger.error("LLM error: %s", exc.detail)
                            reply = "Thanks for sharing. Could you repeat that?"

                        # Log and store this turn
                        turn = {
                            "ts": time.time(),
                            "stream_sid": stream_sid,
                            "call_sid": call_sid,
                            "user": transcript,
                            "agent": reply,
                        }
                        conversation.append(turn)
                        # Persist to calls collection
                        try:
                            if call_sid:
                                await get_collection(COLLECTION_CALLS).update_one(
                                    {"call_id": call_sid},
                                    {"$push": {"conversation": turn}, "$set": {"updated_at": datetime.utcnow()}},
                                )
                        except Exception:
                            pass
                        logger.info(
                            "Turn %d | User: %s | Agent: %s",
                            len(conversation),
                            transcript[:160],
                            reply[:160],
                        )

                        # Speak via SalesAgent (persona voice) when available
                        try:
                            if sales_agent:
                                tts = await sales_agent.speak_response(reply)
                                audio_b64 = tts.get("audio_base64")
                                if audio_b64:
                                    await _stream_audio_b64(websocket, stream_sid or "", audio_b64)
                            else:
                                await _speak_and_stream(
                                    websocket=websocket,
                                    stream_sid=stream_sid or "",
                                    text=reply,
                                    voice_id=settings.ELEVENLABS_DEFAULT_VOICE_ID or None,
                                )
                        except Exception as exc:
                            logger.error("TTS/stream error: %s", exc)

                    processing = False
            elif event == "stop":
                logger.info("Twilio stream stopped: %s", stream_sid)
                # Emit a final summary of the conversation turns
                logger.info("Call summary (%d turns) for stream %s:", len(conversation), stream_sid)
                for idx, t in enumerate(conversation, start=1):
                    logger.info(
                        "  #%d U: %s | A: %s",
                        idx,
                        (t.get("user") or "").strip()[:200],
                        (t.get("agent") or "").strip()[:200],
                    )
                # Mark call completed in DB
                try:
                    if 'call_sid' in locals() and call_sid:
                        await get_collection(COLLECTION_CALLS).update_one(
                            {"call_id": call_sid},
                            {"$set": {"status": "completed", "updated_at": datetime.utcnow()}},
                        )
                except Exception:
                    pass
                break
    except WebSocketDisconnect:
        logger.info("Twilio stream disconnected: %s", stream_sid)
    finally:
        if greeting_task:
            greeting_task.cancel()
        await websocket.close()


@router.websocket("/api/v1/integrations/telephony/twilio/stream")
async def media_stream_endpoint_direct(websocket: WebSocket):
    """Expose the websocket under its fully-qualified path for reverse proxies."""
    await media_stream_endpoint(websocket)
