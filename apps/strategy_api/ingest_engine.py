from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status, Depends
from pydantic import BaseModel
from typing import Dict, Any, Optional
import json
from core.database import get_collection
from core.auth import RoleChecker
from shared.constants import COLLECTION_STRATEGIES
from shared.exceptions import InvalidStrategyException
import uuid
from datetime import datetime

router = APIRouter()

class StrategyIngestRequest(BaseModel):
    """Strategy ingest request model"""
    title: str
    description: str
    scripts: Dict[str, Any] = {}
    products: list = []
    objections: Dict[str, str] = {}
    closing_techniques: list = []
    source: str = "manual"  # "manual", "file", "link"

async def ingest_strategy_data(
    user_id: str,
    strategy_data: Dict[str, Any],
    source: str = "manual"
) -> str:
    """
    Ingest strategy data into the database
    
    Args:
        user_id: User ID
        strategy_data: Strategy data
        source: Source of the strategy data
        
    Returns:
        Strategy ID
    """
    try:
        # Validate strategy data
        if not strategy_data or not isinstance(strategy_data, dict):
            raise InvalidStrategyException("Invalid strategy data")
        
        # Add metadata
        strategy_data["source"] = source
        strategy_data["ingested_at"] = datetime.utcnow()
        
        # Store strategy in database
        strategies_collection = get_collection(COLLECTION_STRATEGIES)
        strategy_entry = {
            "strategy_id": str(uuid.uuid4()),
            "user_id": user_id,
            "payload": strategy_data,
            "source": source,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        result = await strategies_collection.insert_one(strategy_entry)
        
        return strategy_entry["strategy_id"]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest strategy: {str(e)}"
        )

@router.post("/ingest")
async def ingest_strategy(
    payload: StrategyIngestRequest,
    user: dict = Depends(RoleChecker(["user", "admin"]))
):
    """
    Ingest strategy JSON for agent training
    
    Args:
        payload: Strategy data
        user: Authenticated user
        
    Returns:
        Dict with ingestion result
    """
    try:
        strategy_id = await ingest_strategy_data(
            user_id=user["user_id"],
            strategy_data=payload.dict(),
            source="manual"
        )
        
        return {
            "status": "success",
            "message": "Strategy ingested successfully",
            "strategy_id": strategy_id
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest strategy: {str(e)}"
        )

@router.post("/ingest_file")
async def ingest_strategy_file(
    file: UploadFile = File(...),
    user: dict = Depends(RoleChecker(["user", "admin"]))
):
    """
    Ingest strategy from uploaded file (JSON/PDF)
    
    Args:
        file: Uploaded file
        user: Authenticated user
        
    Returns:
        Dict with ingestion result
    """
    try:
        # Check file type
        if file.content_type == "application/json":
            content = await file.read()
            strategy_data = json.loads(content)
        else:
            # For PDF or other file types, we would extract text
            # This is a simplified version for now
            content = await file.read()
            strategy_data = {
                "title": file.filename,
                "description": f"Strategy extracted from {file.filename}",
                "content": content.decode('utf-8', errors='ignore')
            }
        
        strategy_id = await ingest_strategy_data(
            user_id=user["user_id"],
            strategy_data=strategy_data,
            source="file"
        )
        
        return {
            "status": "success",
            "message": "Strategy file ingested successfully",
            "strategy_id": strategy_id
        }
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid JSON file"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest strategy file: {str(e)}"
        )

@router.post("/ingest_link")
async def ingest_strategy_link(
    link: str = Form(...),
    user: dict = Depends(RoleChecker(["user", "admin"]))
):
    """
    Ingest strategy from a link
    
    Args:
        link: Link to strategy content
        user: Authenticated user
        
    Returns:
        Dict with ingestion result
    """
    try:
        # In a real implementation, this would fetch content from the link
        # For now, we'll just simulate the ingestion
        
        strategy_data = {
            "title": f"Strategy from {link}",
            "description": f"Strategy ingested from {link}",
            "link": link
        }
        
        strategy_id = await ingest_strategy_data(
            user_id=user["user_id"],
            strategy_data=strategy_data,
            source="link"
        )
        
        return {
            "status": "success",
            "message": "Strategy link ingested successfully",
            "strategy_id": strategy_id
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest strategy link: {str(e)}"
        )
