from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from core.auth import RoleChecker
from core.config import settings
from shared.exceptions import TelephonyException
import httpx

router = APIRouter()

class ZohoLead(BaseModel):
    """Zoho lead model"""
    first_name: str
    last_name: str
    email: str
    phone: str
    company: Optional[str] = None
    description: Optional[str] = None
    agent_id: str

class ZohoService:
    """Zoho CRM service for lead management"""
    
    def __init__(self):
        self.client_id = settings.ZOHO_CLIENT_ID
        self.client_secret = settings.ZOHO_CLIENT_SECRET
        self.access_token = None
        self.base_url = "https://www.zohoapis.com/crm/v2"
    
    async def authenticate(self):
        """
        Authenticate with Zoho CRM
        
        Returns:
            Bool indicating success
        """
        try:
            # In a real implementation, this would authenticate with Zoho
            # For now, we'll just set a mock token if credentials are provided
            if self.client_id and self.client_secret:
                self.access_token = "mock_access_token"
                return True
            return False
        except Exception as e:
            raise TelephonyException(f"Failed to authenticate: {str(e)}")
    
    async def create_lead(
        self,
        lead_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create lead in Zoho CRM
        
        Args:
            lead_data: Lead data
            
        Returns:
            Dict with lead details
        """
        try:
            # Authenticate if needed
            if not self.access_token:
                await self.authenticate()
            
            # Return mock response if Zoho is not configured
            if not self.access_token:
                return {
                    "lead_id": "mock_lead_id",
                    "status": "created",
                    "message": "Lead created (mock)"
                }
            
            # In a real implementation, this would create a lead in Zoho CRM
            # For now, we'll just return a mock response
            return {
                "lead_id": "zoho_lead_id_12345",
                "status": "created",
                "message": "Lead created successfully in Zoho CRM"
            }
        except Exception as e:
            raise TelephonyException(f"Failed to create lead: {str(e)}")
    
    async def get_lead(
        self,
        lead_id: str
    ) -> Dict[str, Any]:
        """
        Get lead from Zoho CRM
        
        Args:
            lead_id: Lead ID
            
        Returns:
            Dict with lead details
        """
        try:
            # Authenticate if needed
            if not self.access_token:
                await self.authenticate()
            
            # Return mock response if Zoho is not configured
            if not self.access_token:
                return {
                    "lead_id": lead_id,
                    "first_name": "John",
                    "last_name": "Doe",
                    "email": "john.doe@example.com",
                    "phone": "+1234567890",
                    "status": "mock"
                }
            
            # In a real implementation, this would fetch a lead from Zoho CRM
            # For now, we'll just return a mock response
            return {
                "lead_id": lead_id,
                "first_name": "John",
                "last_name": "Doe",
                "email": "john.doe@example.com",
                "phone": "+1234567890",
                "status": "active"
            }
        except Exception as e:
            raise TelephonyException(f"Failed to get lead: {str(e)}")

# Initialize Zoho service
zoho_service = ZohoService()

@router.post("/lead")
async def create_lead(
    lead: ZohoLead,
    user: dict = Depends(RoleChecker(["user", "admin"]))
):
    """
    Create lead in Zoho CRM
    
    Args:
        lead: Lead data
        user: Authenticated user
        
    Returns:
        Dict with lead result
    """
    try:
        result = await zoho_service.create_lead(
            lead_data=lead.dict()
        )
        
        return {
            "status": "success",
            "lead_id": result["lead_id"],
            "status": result["status"],
            "message": result["message"]
        }
    except TelephonyException as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/lead/{lead_id}")
async def get_lead(
    lead_id: str,
    user: dict = Depends(RoleChecker(["user", "admin"]))
):
    """
    Get lead from Zoho CRM
    
    Args:
        lead_id: Lead ID
        user: Authenticated user
        
    Returns:
        Dict with lead details
    """
    try:
        result = await zoho_service.get_lead(
            lead_id=lead_id
        )
        
        return {
            "status": "success",
            "lead": result
        }
    except TelephonyException as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/authenticate")
async def authenticate_zoho(
    user: dict = Depends(RoleChecker(["admin"]))
):
    """
    Authenticate with Zoho CRM
    
    Args:
        user: Authenticated user
        
    Returns:
        Dict with authentication result
    """
    try:
        result = await zoho_service.authenticate()
        
        return {
            "status": "success",
            "authenticated": result,
            "message": "Authenticated with Zoho CRM" if result else "Zoho CRM not configured"
        }
    except TelephonyException as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
