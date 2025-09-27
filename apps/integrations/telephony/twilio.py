import asyncio
import base64
import json
import logging
import subprocess
from typing import Optional, Dict, Any, List

from fastapi import APIRouter, HTTPException, status, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from pydantic import BaseModel
from core.auth import RoleChecker
from core.config import settings
from shared.exceptions import TelephonyException
from apps.agents.sales import services
import httpx
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse

router = APIRouter()

logger = logging.getLogger(__name__)

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
            elif settings.TWILIO_CALL_WEBHOOK_URL:
                webhook = settings.TWILIO_CALL_WEBHOOK_URL
            elif settings.TWILIO_PUBLIC_BASE_URL:
                webhook = f"{settings.TWILIO_PUBLIC_BASE_URL.rstrip('/')}/twilio/voice"
            elif settings.API_BASE_URL:
                webhook = f"{settings.API_BASE_URL.rstrip('/')}/api/v1/integrations/telephony/twilio/voice"
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
        connect.stream(url=stream_url, track="both_tracks")
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


@router.websocket("/stream")
async def media_stream_endpoint(websocket: WebSocket):
    await websocket.accept()
    stream_sid: Optional[str] = None
    greeting_task: Optional[asyncio.Task] = None

    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            event = payload.get("event")

            if event == "start":
                stream_info = payload.get("start", {})
                stream_sid = stream_info.get("streamSid")
                logger.info("Twilio stream started: %s", stream_sid)
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
                pass  # Placeholder for streaming ASR pipeline
            elif event == "stop":
                logger.info("Twilio stream stopped: %s", stream_sid)
                break
    except WebSocketDisconnect:
        logger.info("Twilio stream disconnected: %s", stream_sid)
    finally:
        if greeting_task:
            greeting_task.cancel()
        await websocket.close()
