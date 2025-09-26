from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel
from typing import List, Optional
from core.auth import RoleChecker
from core.database import get_collection
from shared.constants import COLLECTION_USERS
from datetime import datetime, timedelta
import uuid

router = APIRouter()

class SubscriptionPlan(BaseModel):
    """Subscription plan model"""
    plan_id: str
    name: str
    description: str
    price: float
    call_limit: int  # Number of calls allowed per month
    agent_limit: int  # Number of agents allowed

class Subscription(BaseModel):
    """Subscription model"""
    subscription_id: str
    user_id: str
    plan_id: str
    start_date: datetime
    end_date: datetime
    is_active: bool
    call_count: int = 0
    agent_count: int = 0

class UsageRecord(BaseModel):
    """Usage record model"""
    usage_id: str
    user_id: str
    resource: str  # "calls", "agents", etc.
    count: int
    timestamp: datetime

@router.get("/plans")
async def get_subscription_plans(
    user: dict = Depends(RoleChecker(["user", "admin"]))
):
    """Get available subscription plans"""
    # In a real implementation, this would fetch plans from a database
    # For now, we'll return mock plans
    
    mock_plans = [
        {
            "plan_id": "free",
            "name": "Free Plan",
            "description": "Basic plan for getting started",
            "price": 0.0,
            "call_limit": 100,
            "agent_limit": 1
        },
        {
            "plan_id": "starter",
            "name": "Starter Plan",
            "description": "Perfect for small businesses",
            "price": 49.0,
            "call_limit": 1000,
            "agent_limit": 5
        },
        {
            "plan_id": "professional",
            "name": "Professional Plan",
            "description": "For growing businesses",
            "price": 99.0,
            "call_limit": 5000,
            "agent_limit": 20
        },
        {
            "plan_id": "enterprise",
            "name": "Enterprise Plan",
            "description": "For large organizations",
            "price": 299.0,
            "call_limit": 50000,
            "agent_limit": 100
        }
    ]
    
    return {
        "status": "success",
        "plans": mock_plans
    }

@router.post("/subscribe/{plan_id}")
async def subscribe_to_plan(
    plan_id: str,
    user: dict = Depends(RoleChecker(["user", "admin"]))
):
    """Subscribe to a plan"""
    # In a real implementation, this would integrate with a payment processor
    # For now, we'll just simulate the subscription process
    
    # Check if plan exists
    plans = await get_subscription_plans(user)
    plan_exists = any(plan["plan_id"] == plan_id for plan in plans["plans"])
    
    if not plan_exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Plan not found"
        )
    
    # Create subscription
    subscription = {
        "subscription_id": str(uuid.uuid4()),
        "user_id": user["user_id"],
        "plan_id": plan_id,
        "start_date": datetime.utcnow(),
        "end_date": datetime.utcnow() + timedelta(days=30),  # 30-day subscription
        "is_active": True,
        "call_count": 0,
        "agent_count": 0
    }
    
    # Store subscription in database
    # TODO: Implement actual database storage
    
    return {
        "status": "success",
        "message": f"Subscribed to {plan_id} plan successfully",
        "subscription": subscription
    }

@router.get("/usage")
async def get_usage(
    user: dict = Depends(RoleChecker(["user", "admin"]))
):
    """Get usage statistics"""
    # In a real implementation, this would fetch actual usage data
    # For now, we'll return mock usage data
    
    return {
        "status": "success",
        "usage": {
            "call_count": 45,
            "agent_count": 3,
            "storage_used": "2.4 GB",
            "api_calls": 1240
        }
    }

@router.get("/billing")
async def get_billing_info(
    user: dict = Depends(RoleChecker(["user", "admin"]))
):
    """Get billing information"""
    # In a real implementation, this would fetch actual billing data
    # For now, we'll return mock billing data
    
    return {
        "status": "success",
        "billing": {
            "current_plan": "starter",
            "next_billing_date": "2023-12-01",
            "amount_due": 49.00,
            "payment_method": "Visa ending in 1234"
        }
    }
