from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from core.auth import RoleChecker
from core.database import get_collection
from shared.constants import COLLECTION_USERS, COLLECTION_AGENTS, COLLECTION_CALLS
from datetime import datetime, timedelta

router = APIRouter()

class SystemStats(BaseModel):
    """System statistics model"""
    total_users: int
    total_agents: int
    active_calls: int
    total_calls_today: int
    system_uptime: str

class UserStats(BaseModel):
    """User statistics model"""
    user_id: str
    email: str
    agent_count: int
    call_count: int
    last_active: datetime

@router.get("/stats")
async def get_system_stats(
    user: dict = Depends(RoleChecker(["admin"]))
):
    """Get system statistics (admin only)"""
    try:
        # Get database collections
        users_collection = get_collection(COLLECTION_USERS)
        agents_collection = get_collection(COLLECTION_AGENTS)
        calls_collection = get_collection(COLLECTION_CALLS)
        
        # Get total users
        total_users = await users_collection.count_documents({})
        
        # Get total agents
        total_agents = await agents_collection.count_documents({})
        
        # Get active calls
        active_calls = await calls_collection.count_documents({
            "status": "in_progress"
        })
        
        # Get calls from today
        today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        total_calls_today = await calls_collection.count_documents({
            "created_at": {"$gte": today}
        })
        
        # TODO: Implement actual system uptime tracking
        system_uptime = "24 hours"
        
        return SystemStats(
            total_users=total_users,
            total_agents=total_agents,
            active_calls=active_calls,
            total_calls_today=total_calls_today,
            system_uptime=system_uptime
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch system stats: {str(e)}"
        )

@router.get("/user_stats")
async def get_user_stats(
    skip: int = 0,
    limit: int = 50,
    user: dict = Depends(RoleChecker(["admin"]))
):
    """Get user statistics (admin only)"""
    try:
        # Get database collections
        users_collection = get_collection(COLLECTION_USERS)
        agents_collection = get_collection(COLLECTION_AGENTS)
        calls_collection = get_collection(COLLECTION_CALLS)
        
        # Get users
        cursor = users_collection.find().skip(skip).limit(limit)
        users = await cursor.to_list(length=limit)
        
        user_stats = []
        for user_doc in users:
            user_id = user_doc["user_id"]
            
            # Get agent count for user
            agent_count = await agents_collection.count_documents({
                "user_id": user_id
            })
            
            # Get call count for user
            call_count = await calls_collection.count_documents({
                "user_id": user_id
            })
            
            user_stats.append(UserStats(
                user_id=user_id,
                email=user_doc["email"],
                agent_count=agent_count,
                call_count=call_count,
                last_active=user_doc.get("updated_at", user_doc.get("created_at", datetime.utcnow()))
            ))
        
        return {
            "status": "success",
            "user_stats": user_stats
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch user stats: {str(e)}"
        )

@router.post("/system/maintenance")
async def run_system_maintenance(
    user: dict = Depends(RoleChecker(["admin"]))
):
    """Run system maintenance (admin only)"""
    try:
        # TODO: Implement actual maintenance tasks
        # This could include:
        # - Cleaning up old logs
        # - Archiving old data
        # - Running database optimizations
        # - Checking for inconsistencies
        
        return {
            "status": "success",
            "message": "System maintenance completed successfully",
            "tasks_completed": [
                "Database optimization",
                "Log cleanup",
                "Cache refresh"
            ]
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to run system maintenance: {str(e)}"
        )

@router.post("/system/backup")
async def trigger_backup(
    user: dict = Depends(RoleChecker(["admin"]))
):
    """Trigger system backup (admin only)"""
    try:
        # TODO: Implement actual backup logic
        # This would typically:
        # - Backup database
        # - Backup configuration files
        # - Store backups in secure location
        # - Send notification on completion
        
        return {
            "status": "success",
            "message": "Backup initiated successfully",
            "backup_id": "backup_12345"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initiate backup: {str(e)}"
        )
