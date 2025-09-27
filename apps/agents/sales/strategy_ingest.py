from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status, Depends
from typing import Dict, Any, Optional
import json
from pydantic import BaseModel
from core.database import get_collection
from core.auth import RoleChecker
from shared.constants import COLLECTION_STRATEGIES
from shared.exceptions import InvalidStrategyException
import uuid
from datetime import datetime

router = APIRouter()

class PersonaConfig(BaseModel):
    name: Optional[str] = None
    tone: Optional[str] = "friendly, empathetic, persuasive"
    description: Optional[str] = None
    voice_id: Optional[str] = None
    locale: Optional[str] = "en-IN"


class StrategyPayload(BaseModel):
    """Strategy payload model"""
    title: str
    description: str
    scripts: Dict[str, Any] = {}
    products: list = []
    objections: Dict[str, str] = {}
    closing_techniques: list = []
    persona: PersonaConfig = PersonaConfig()
    fallback_scenarios: Dict[str, str] = {}

@router.post("/ingest_strategy")
async def ingest_strategy(
    payload: StrategyPayload,
    user: dict = Depends(RoleChecker(["user", "admin"]))
):
    """
    Ingest strategy JSON for sales agent training
    
    Args:
        payload: Strategy data
        user: Authenticated user
        
    Returns:
        Dict with ingestion result
    """
    try:
        # Store strategy in database
        strategies_collection = get_collection(COLLECTION_STRATEGIES)
        strategy_entry = {
            "strategy_id": str(uuid.uuid4()),
            "user_id": user["user_id"],
            "payload": payload.dict(),
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        result = await strategies_collection.insert_one(strategy_entry)
        
        return {
            "status": "success",
            "message": "Strategy ingested successfully",
            "strategy_id": strategy_entry["strategy_id"]
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest strategy: {str(e)}"
        )

@router.post("/ingest_strategy_file")
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
        
        # Store strategy in database
        strategies_collection = get_collection(COLLECTION_STRATEGIES)
        strategy_entry = {
            "strategy_id": str(uuid.uuid4()),
            "user_id": user["user_id"],
            "payload": strategy_data,
            "file_name": file.filename,
            "content_type": file.content_type,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        result = await strategies_collection.insert_one(strategy_entry)
        
        return {
            "status": "success",
            "message": "Strategy file ingested successfully",
            "strategy_id": strategy_entry["strategy_id"]
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
