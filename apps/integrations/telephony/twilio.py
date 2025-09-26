from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel
from typing import Optional, Dict, Any
from core.auth import RoleChecker
from core.config import settings
from shared.exceptions import TelephonyException
import httpx
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse

router = APIRouter()

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
            call = self.client.calls.create(
                to=to_number,
                from_=from_number,
                url=f"{callback_url}/voice" if callback_url else f"{settings.API_BASE_URL}/api/v1/integrations/telephony/twilio/voice",
                method="GET"
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

@router.post("/voice")
async def handle_voice_call():
    """
    Handle incoming voice call
    
    Returns:
        TwiML response
    """
    try:
        # Create TwiML response
        response = VoiceResponse()
        response.say("Hello! This is a Scriza AI agent. How can I help you today?")
        response.hangup()
        
        return str(response)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to handle voice call: {str(e)}"
        )
