from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel
from typing import List, Optional
from core.auth import RoleChecker
from core.database import get_collection
from shared.constants import COLLECTION_STRATEGIES
from datetime import datetime
from typing import Dict, Any

router = APIRouter()

class TrainingLogEntry(BaseModel):
    """Training log entry model"""
    log_id: str
    user_id: str
    agent_id: str
    strategy_id: str
    action: str  # "trained", "updated", "deleted"
    timestamp: datetime
    details: Dict[str, Any]

class TrainingLogResponse(BaseModel):
    """Training log response model"""
    logs: List[TrainingLogEntry]
    total: int
    page: int
    size: int

@router.get("/training_logs", response_model=TrainingLogResponse)
async def get_training_logs(
    skip: int = 0,
    limit: int = 50,
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    user: dict = Depends(RoleChecker(["admin"]))
):
    """Get training logs (admin only)"""
    try:
        strategies_collection = get_collection(COLLECTION_STRATEGIES)
        
        # Build query
        query = {}
        if user_id:
            query["user_id"] = user_id
        if agent_id:
            query["agent_id"] = agent_id
        
        # Get total count
        total = await strategies_collection.count_documents(query)
        
        # Get logs
        cursor = strategies_collection.find(query).sort("created_at", -1).skip(skip).limit(limit)
        logs = await cursor.to_list(length=limit)
        
        # Transform logs to TrainingLogEntry format
        log_entries = []
        for log in logs:
            log_entries.append(TrainingLogEntry(
                log_id=str(log.get("_id", "")),
                user_id=log.get("user_id", ""),
                agent_id=log.get("agent_id", ""),
                strategy_id=log.get("strategy_id", ""),
                action="trained",
                timestamp=log.get("created_at", datetime.utcnow()),
                details=log.get("payload", {})
            ))
        
        return TrainingLogResponse(
            logs=log_entries,
            total=total,
            page=skip // limit + 1,
            size=len(log_entries)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch training logs: {str(e)}"
        )

@router.get("/training_logs/{log_id}")
async def get_training_log_detail(
    log_id: str,
    user: dict = Depends(RoleChecker(["admin"]))
):
    """Get detailed training log (admin only)"""
    try:
        strategies_collection = get_collection(COLLECTION_STRATEGIES)
        log = await strategies_collection.find_one({"_id": log_id})
        
        if not log:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Training log not found"
            )
        
        return {
            "status": "success",
            "log": log
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch training log: {str(e)}"
        )
