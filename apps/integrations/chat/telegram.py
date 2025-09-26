from fastapi import APIRouter, HTTPException, status, Depends, Request
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from core.auth import RoleChecker
from core.config import settings
from shared.exceptions import TelephonyException
import httpx
import json

router = APIRouter()

class TelegramMessage(BaseModel):
    """Telegram message model"""
    chat_id: int
    text: str
    agent_id: str

class TelegramUpdate(BaseModel):
    """Telegram update model"""
    update_id: int
    message: Optional[Dict[str, Any]] = None

class TelegramService:
    """Telegram service for chat operations"""
    
    def __init__(self):
        self.token = settings.TELEGRAM_BOT_TOKEN
        self.base_url = f"https://api.telegram.org/bot{self.token}"
    
    async def send_message(
        self,
        chat_id: int,
        text: str
    ) -> Dict[str, Any]:
        """
        Send message to Telegram chat
        
        Args:
            chat_id: Telegram chat ID
            text: Message text
            
        Returns:
            Dict with message details
        """
        try:
            if not self.token:
                # Return mock response if Telegram is not configured
                return {
                    "message_id": 12345,
                    "status": "sent",
                    "message": "Message sent (mock)"
                }
            
            # Send the message
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/sendMessage",
                    json={
                        "chat_id": chat_id,
                        "text": text
                    }
                )
                
                if response.status_code != 200:
                    raise TelephonyException(f"Failed to send message: {response.text}")
                
                result = response.json()
                return {
                    "message_id": result["result"]["message_id"],
                    "status": "sent",
                    "message": "Message sent successfully"
                }
        except Exception as e:
            raise TelephonyException(f"Failed to send message: {str(e)}")
    
    async def get_updates(
        self,
        offset: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get updates from Telegram
        
        Args:
            offset: Update offset
            
        Returns:
            List of updates
        """
        try:
            if not self.token:
                # Return mock response if Telegram is not configured
                return []
            
            # Get updates
            async with httpx.AsyncClient() as client:
                params = {}
                if offset:
                    params["offset"] = offset
                
                response = await client.get(
                    f"{self.base_url}/getUpdates",
                    params=params
                )
                
                if response.status_code != 200:
                    raise TelephonyException(f"Failed to get updates: {response.text}")
                
                result = response.json()
                return result["result"]
        except Exception as e:
            raise TelephonyException(f"Failed to get updates: {str(e)}")

# Initialize Telegram service
telegram_service = TelegramService()

@router.post("/message")
async def send_message(
    message: TelegramMessage,
    user: dict = Depends(RoleChecker(["user", "admin"]))
):
    """
    Send message to Telegram chat
    
    Args:
        message: Message data
        user: Authenticated user
        
    Returns:
        Dict with message result
    """
    try:
        result = await telegram_service.send_message(
            chat_id=message.chat_id,
            text=message.text
        )
        
        return {
            "status": "success",
            "message_id": result["message_id"],
            "status": result["status"],
            "message": result["message"]
        }
    except TelephonyException as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/webhook")
async def telegram_webhook(
    request: Request
):
    """
    Handle Telegram webhook
    
    Args:
        request: HTTP request
        
    Returns:
        Dict with webhook result
    """
    try:
        # Get request body
        body = await request.json()
        
        # Process update
        update = TelegramUpdate(**body)
        
        # Handle message if present
        if update.message and update.message.get("text"):
            chat_id = update.message["chat"]["id"]
            text = update.message["text"]
            
            # TODO: Process message with AI agent
            # For now, we'll just echo the message
            response_text = f"You said: {text}"
            
            # Send response
            await telegram_service.send_message(
                chat_id=chat_id,
                text=response_text
            )
        
        return {
            "status": "success",
            "message": "Webhook processed successfully"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process webhook: {str(e)}"
        )

@router.get("/set_webhook")
async def set_webhook(
    url: str,
    user: dict = Depends(RoleChecker(["admin"]))
):
    """
    Set Telegram webhook
    
    Args:
        url: Webhook URL
        user: Authenticated user
        
    Returns:
        Dict with webhook result
    """
    try:
        if not telegram_service.token:
            return {
                "status": "success",
                "message": "Webhook set (mock)"
            }
        
        # Set webhook
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{telegram_service.base_url}/setWebhook",
                json={
                    "url": url
                }
            )
            
            if response.status_code != 200:
                raise TelephonyException(f"Failed to set webhook: {response.text}")
            
            result = response.json()
            return {
                "status": "success",
                "message": "Webhook set successfully",
                "result": result
            }
    except TelephonyException as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
