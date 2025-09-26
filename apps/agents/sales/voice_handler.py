from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from apps.agents.sales import services
from core.auth import RoleChecker
from shared.exceptions import VoiceSynthesisException

router = APIRouter()

class VoiceRequest(BaseModel):
    text: str
    voice_id: Optional[str] = None
    agent_id: str

class VoiceResponse(BaseModel):
    audio_base64: str
    duration: float
    text: str

@router.post("/synthesize_voice", response_model=VoiceResponse)
async def synthesize_voice(
    voice_request: VoiceRequest,
    user: dict = Depends(RoleChecker(["user", "admin"]))
):
    """
    Synthesize voice from text using ElevenLabs API
    
    Args:
        voice_request: Voice synthesis request
        user: Authenticated user
        
    Returns:
        VoiceResponse with audio URL and metadata
    """
    try:
        payload = await services.synthesise_elevenlabs_voice(
            text=voice_request.text,
            voice_id=voice_request.voice_id,
        )
        return VoiceResponse(
            audio_base64=payload["audio_base64"],
            duration=payload["duration"],
            text=voice_request.text,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to synthesize voice: {str(e)}"
        )

@router.get("/voices")
async def list_voices(
    user: dict = Depends(RoleChecker(["user", "admin"]))
):
    """
    List available voices
    
    Args:
        user: Authenticated user
        
    Returns:
        Dict with available voices
    """
    try:
        voices = await services.list_elevenlabs_voices()
        return {"status": "success", "voices": voices}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch voices: {str(e)}"
        )
