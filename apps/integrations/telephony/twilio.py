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
import traceback

from fastapi import APIRouter, HTTPException, status, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from pydantic import BaseModel
from core.auth import RoleChecker
from core.config import settings
from shared.exceptions import TelephonyException
from apps.agents.sales import services
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
        # Default to inbound audio only; omit track to avoid 31941 errors.
        # If you need both inbound & outbound frames from Twilio, set TWILIO_STREAM_TRACK=both_tracks.
        if settings.TWILIO_STREAM_TRACK:
            connect.stream(url=stream_url, track=settings.TWILIO_STREAM_TRACK)
        else:
            connect.stream(url=stream_url)
        logger.info("Issued TwiML stream to %s", stream_url)
        print(f"[twilio.voice] Issued TwiML stream to {stream_url}")
        return Response(content=str(response), media_type="application/xml")
    except Exception as e:
        print("[twilio.voice] Exception while building TwiML:", e)
        print(traceback.format_exc())
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
        print("[send_audio_prompt] TTS HTTPException:", exc.detail)
        print(traceback.format_exc())
        return

    try:
        chunks = _mp3_to_mulaw_chunks(voice_payload["audio_base64"])
    except TelephonyException as exc:
        logger.error("Audio conversion failed: %s", exc)
        print("[send_audio_prompt] Audio conversion failed:", exc)
        print(traceback.format_exc())
        return

    print(f"[send_audio_prompt] Streaming greeting: {len(text)} chars, {len(chunks)} chunks, streamSid={stream_sid}")
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
    print(f"[asr] Posting {len(wav_bytes)} bytes to Whisper")
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.openai.com/v1/audio/transcriptions", headers=headers, files=files, data=data
        )
        try:
            resp.raise_for_status()
        except Exception:
            print("[asr] Whisper HTTP error:", resp.status_code, resp.text[:500])
            print(traceback.format_exc())
            raise
        data = resp.json()
        text = data.get("text", "").strip()
        print(f"[asr] Whisper response: '{text}'")
        return text


async def _speak_and_stream(websocket: WebSocket, stream_sid: str, text: str, voice_id: Optional[str]) -> None:
    """TTS via ElevenLabs and stream μ-law frames back to Twilio."""
    try:
        tts = await services.synthesise_elevenlabs_voice(text=text, voice_id=voice_id)
        chunks = _mp3_to_mulaw_chunks(tts["audio_base64"])
    except HTTPException as exc:
        logger.error("TTS failed: %s", exc.detail)
        print("[tts] ElevenLabs HTTPException:", exc.detail)
        print(traceback.format_exc())
        return
    except Exception as exc:
        print("[tts] Unexpected error during TTS or conversion:", exc)
        print(traceback.format_exc())
        return
    print(f"[tts] Streaming reply: {len(text)} chars, {len(chunks)} chunks, streamSid={stream_sid}")
    for chunk in chunks:
        await websocket.send_text(
            json.dumps({"event": "media", "streamSid": stream_sid, "media": {"payload": chunk}})
        )
        await asyncio.sleep(0.02)


@router.websocket("/stream")
async def media_stream_endpoint(websocket: WebSocket):
    print("[ws] Accepting websocket…")
    await websocket.accept()
    print("[ws] Websocket accepted")
    stream_sid: Optional[str] = None
    greeting_task: Optional[asyncio.Task] = None
    nudge_task: Optional[asyncio.Task] = None
    # Rolling utterance buffer built by per-frame VAD
    utterance_buffer = bytearray()
    last_transcript = ""
    processing = False
    # Collect conversational turns for end-of-call summary
    conversation: List[Dict[str, Any]] = []
    last_user_activity = time.monotonic()
    # Gating flags and thresholds
    greet_in_progress = False
    speaking_out = False
    # Tuning knobs from environment, with sane defaults
    FRAME_DUR_MS = 20  # Twilio sends 20ms frames by default
    silence_threshold_ms = int(float(settings.CALL_SILENCE_THRESHOLD_SEC) * 1000)
    min_utter_ms = int(settings.CALL_MIN_BUFFER_MS)
    min_utter_bytes = 8 * min_utter_ms  # μ-law @ 8kHz ≈ 8 bytes/ms
    min_rms = int(settings.CALL_VAD_MIN_RMS)
    verbose = bool(settings.CALL_DEBUG_VERBOSE)
    frames_seen = 0
    in_speech = False
    trailing_silence_ms = 0

    try:
        while True:
            data = await websocket.receive_text()
            print(f"[ws] Received frame: {len(data)} chars")
            payload = json.loads(data)
            event = payload.get("event")
            print(f"[ws] Event: {event}")

            if event == "start":
                stream_info = payload.get("start", {})
                stream_sid = stream_info.get("streamSid")
                call_sid = stream_info.get("callSid")
                logger.info("Twilio stream started: %s", stream_sid)
                print(f"[ws] start | streamSid={stream_sid} callSid={call_sid}")
                if settings.CALL_FIRST_TURN_GREETING:
                    greeting_text = "Hello! This is Scriza AI. Thanks for taking the call."  # TODO personalise
                    greet_in_progress = True
                    if verbose:
                        logger.info("Greeting caller with first-turn prompt (%d chars)", len(greeting_text))
                    greeting_task = asyncio.create_task(
                        _send_audio_prompt(
                            websocket=websocket,
                            stream_sid=stream_sid,
                            text=greeting_text,
                            voice_id=settings.ELEVENLABS_DEFAULT_VOICE_ID or None,
                        )
                    )
                    def _greet_done(_):
                        nonlocal greet_in_progress
                        nonlocal nudge_task
                        greet_in_progress = False
                        if verbose:
                            logger.info("Greeting audio completed; ready to receive caller input…")
                        print("[ws] Greeting complete; ready for input")
                        # Optionally schedule a nudge if caller remains silent
                        if settings.CALL_NUDGE_AFTER_SEC > 0 and nudge_task is None:
                            async def _run_nudge():
                                await asyncio.sleep(settings.CALL_NUDGE_AFTER_SEC)
                                if (
                                    len(conversation) == 0
                                    and not greet_in_progress
                                    and not speaking_out
                                ):
                                    if verbose:
                                        logger.info("No user input detected; sending nudge prompt")
                                    await _speak_and_stream(
                                        websocket,
                                        stream_sid or "",
                                        "Hello? I’m here. How can I help you today?",
                                        settings.ELEVENLABS_DEFAULT_VOICE_ID or None,
                                    )
                            nudge_task = asyncio.create_task(_run_nudge())
                    greeting_task.add_done_callback(_greet_done)
                else:
                    if verbose:
                        logger.info("CALL_FIRST_TURN_GREETING disabled; waiting for user speech")
            elif event == "media":
                media = payload.get("media", {})
                b64 = media.get("payload")
                trk = media.get("track")
                try:
                    b64_len = len(b64) if isinstance(b64, str) else 0
                except Exception:
                    b64_len = 0
                print(f"[ws.media] track={trk} payload_len={b64_len}")
                if not b64:
                    print("[ws.media] Missing payload; skipping")
                    continue
                # Accept both Twilio variants for inbound
                if trk and trk not in ("inbound", "inbound_track"):
                    if verbose and trk:
                        logger.info("Skipping non-inbound media track: %s", trk)
                    print(f"[ws.media] Skipping non-inbound track: {trk}")
                    continue
                # Ensure b64 is str, decode to bytes
                try:
                    if isinstance(b64, str):
                        decoded = base64.b64decode(b64, validate=False)
                    else:
                        # Unexpected type; skip
                        print(f"[ws.media] Unexpected payload type: {type(b64)}")
                        continue
                    if not isinstance(decoded, (bytes, bytearray)):
                        print(f"[ws.media] Decoded payload is not bytes: {type(decoded)}")
                        continue
                    frames_seen += 1
                    # Per-frame VAD based on RMS energy; Twilio continuously sends frames,
                    # so we detect end-of-utterance by trailing silence duration, not gaps in frames.
                    if greet_in_progress or speaking_out:
                        if verbose and frames_seen % 50 == 0:
                            logger.info(
                                "Inbound frames received but gated (greet_in_progress=%s speaking_out=%s)",
                                greet_in_progress,
                                speaking_out,
                            )
                        continue

                    pcm16f = audioop.ulaw2lin(decoded, 2)
                    energy = audioop.rms(pcm16f, 2)
                    if frames_seen % 25 == 0:
                        print(f"[vad.frame] rms={energy} in_speech={in_speech} trail_sil={trailing_silence_ms}ms buf={len(utterance_buffer)} bytes")

                    if energy >= min_rms:
                        # Active speech
                        utterance_buffer += decoded
                        if not in_speech:
                            in_speech = True
                        trailing_silence_ms = 0
                    else:
                        # Silence frame
                        if in_speech:
                            trailing_silence_ms += FRAME_DUR_MS
                            # include short trailing silence to avoid truncation
                            if trailing_silence_ms <= silence_threshold_ms:
                                utterance_buffer += decoded

                            if (
                                not processing
                                and trailing_silence_ms >= silence_threshold_ms
                                and len(utterance_buffer) >= min_utter_bytes
                            ):
                                print(f"[vad] Utterance end | len={len(utterance_buffer)} bytes, trailing_silence={trailing_silence_ms}ms")
                                processing = True
                                chunk = bytes(utterance_buffer)
                                utterance_buffer.clear()
                                in_speech = False
                                trailing_silence_ms = 0

                                # Decode μ-law to WAV and transcribe
                                try:
                                    wav = _mulaw_to_wav(chunk)
                                    print(f"[vad] Built WAV: {len(wav)} bytes; sending to ASR…")
                                    transcript = await _transcribe_with_openai(wav)
                                except Exception as exc:
                                    logger.error("Transcription error: %s", exc)
                                    print("[asr] Transcription error:", exc)
                                    print(traceback.format_exc())
                                    transcript = ""

                                if transcript and transcript != last_transcript:
                                    last_transcript = transcript
                                    last_user_activity = time.monotonic()
                                    logger.info("Caller said: %s", transcript)
                                    print(f"[conv] Caller said: {transcript}")
                                    # Generate reply and speak it
                                    try:
                                        reply = await services.call_openai_chat(
                                            [
                                                {"role": "system", "content": "You are Scriza Sales AI; reply concisely and helpfully."},
                                                {"role": "user", "content": transcript},
                                            ],
                                            model=settings.OPENAI_MODEL,
                                        )
                                        print(f"[conv] Agent reply: {reply}")
                                    except HTTPException as exc:
                                        logger.error("LLM error: %s", exc.detail)
                                        print("[conv] LLM HTTPException:", exc.detail)
                                        print(traceback.format_exc())
                                        reply = "Thanks for sharing. Could you repeat that?"
                                    except Exception as exc:
                                        print("[conv] Unexpected LLM error:", exc)
                                        print(traceback.format_exc())
                                        reply = "Sorry, I missed that. Could you say it again?"

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
                                            res = await get_collection(COLLECTION_CALLS).update_one(
                                                {"call_id": call_sid},
                                                {"$push": {"conversation": turn}, "$set": {"updated_at": datetime.utcnow()}},
                                            )
                                            print(f"[db] Updated call record turns: matched={res.matched_count} modified={res.modified_count}")
                                    except Exception as exc:
                                        print("[db] Failed to append conversation turn:", exc)
                                        print(traceback.format_exc())
                                    logger.info(
                                        "Turn %d | User: %s | Agent: %s",
                                        len(conversation),
                                        transcript[:160],
                                        reply[:160],
                                    )

                                    speaking_out = True
                                    try:
                                        await _speak_and_stream(
                                            websocket=websocket,
                                            stream_sid=stream_sid or "",
                                            text=reply,
                                            voice_id=settings.ELEVENLABS_DEFAULT_VOICE_ID or None,
                                        )
                                    finally:
                                        speaking_out = False

                                processing = False
                except (binascii.Error, ValueError) as exc:
                    if verbose:
                        logger.info("Base64 decode error on media frame: %s", exc)
                    print("[ws.media] Base64 decode error:", exc)
                    print(traceback.format_exc())
                    continue

                # Utterance end and processing handled in per-frame VAD above
            elif event == "mark":
                # Twilio will echo marks we send; can be used to end greet_in_progress
                mark = payload.get("mark", {})
                name = mark.get("name")
                if name == "greeting_complete":
                    greet_in_progress = False
                    if verbose:
                        logger.info("Received mark: greeting_complete; ready to receive caller input…")
                    print("[ws] mark=greeting_complete; ready for input")
                    if settings.CALL_NUDGE_AFTER_SEC > 0 and nudge_task is None:
                        async def _run_nudge():
                            await asyncio.sleep(settings.CALL_NUDGE_AFTER_SEC)
                            if (
                                len(conversation) == 0
                                and not greet_in_progress
                                and not speaking_out
                            ):
                                if verbose:
                                    logger.info("No user input detected; sending nudge prompt")
                                await _speak_and_stream(
                                    websocket,
                                    stream_sid or "",
                                    "Hello? I’m here. How can I help you today?",
                                    settings.ELEVENLABS_DEFAULT_VOICE_ID or None,
                                )
                        nudge_task = asyncio.create_task(_run_nudge())
            elif event == "stop":
                logger.info("Twilio stream stopped: %s", stream_sid)
                print(f"[ws] stop | streamSid={stream_sid}")
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
                        res = await get_collection(COLLECTION_CALLS).update_one(
                            {"call_id": call_sid},
                            {"$set": {"status": "completed", "updated_at": datetime.utcnow()}},
                        )
                        print(f"[db] Marked call completed: matched={res.matched_count} modified={res.modified_count}")
                except Exception as exc:
                    print("[db] Failed to mark call completed:", exc)
                    print(traceback.format_exc())
                break
    except WebSocketDisconnect:
        logger.info("Twilio stream disconnected: %s", stream_sid)
        print(f"[ws] disconnected | streamSid={stream_sid}")
    finally:
        if greeting_task:
            greeting_task.cancel()
        if nudge_task:
            nudge_task.cancel()
        print("[ws] closing websocket")
        await websocket.close()


@router.websocket("/api/v1/integrations/telephony/twilio/stream")
async def media_stream_endpoint_direct(websocket: WebSocket):
    """Expose the websocket under its fully-qualified path for reverse proxies."""
    await media_stream_endpoint(websocket)
