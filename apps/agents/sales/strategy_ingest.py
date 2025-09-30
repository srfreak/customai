from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status, Depends
from typing import Dict, Any, Optional, List
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
    goals: list = []
    products: list = []
    objections: Dict[str, str] = {}
    closing_techniques: list = []
    persona: PersonaConfig = PersonaConfig()
    fallback_scenarios: Dict[str, str] = {}
    # New business-aligned fields (optional)
    agent_profile: Optional[Dict[str, Any]] = None
    business_info: Optional[Dict[str, Any]] = None
    product_details: Optional[List[Dict[str, Any]]] = None
    target_audience: Optional[Dict[str, Any]] = None
    common_objections: Optional[Dict[str, str]] = None
    pitch_examples: Optional[Dict[str, Any]] = None
    call_to_actions: Optional[List[str]] = None


def _normalize_strategy_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Map business-facing keys into the internal structure used by the agent.

    - agent_profile → persona
    - pitch_examples.intro → scripts.greeting
    - pitch_examples.elevator_pitch → scripts.pitch
    - common_objections → objections
    - call_to_actions → closing_techniques and scripts.closing
    - product_details → products
    - business_info/target_audience retained for reference
    """
    normalized: Dict[str, Any] = dict(payload)  # shallow copy

    # Ensure sub dicts exist
    scripts = dict(normalized.get("scripts") or {})
    persona = dict((normalized.get("persona") or {}))

    # agent_profile → persona
    agent_profile = normalized.get("agent_profile") or {}
    if isinstance(agent_profile, dict):
        persona.setdefault("name", agent_profile.get("name"))
        # Combine tone/style
        tone = agent_profile.get("tone") or persona.get("tone")
        style = agent_profile.get("style")
        if tone and style:
            persona["tone"] = f"{tone}; style: {style}"
        elif tone:
            persona["tone"] = tone
        # If style provided and no description, capture it
        if style and not persona.get("description"):
            persona["description"] = f"Conversational style: {style}"

    # pitch_examples → scripts
    pitch_examples = normalized.get("pitch_examples") or {}
    if isinstance(pitch_examples, dict):
        intro = pitch_examples.get("intro") or pitch_examples.get("sample_intro")
        if intro:
            scripts["greeting"] = intro
        elevator = pitch_examples.get("elevator_pitch") or pitch_examples.get("pitch")
        if elevator:
            scripts["pitch"] = elevator

    # call_to_actions → closing_techniques + scripts.closing
    ctas = normalized.get("call_to_actions") or []
    if isinstance(ctas, list) and ctas:
        normalized["closing_techniques"] = list(ctas)
        scripts["closing"] = list(ctas)

    # common_objections → objections
    common_obj = normalized.get("common_objections") or {}
    if isinstance(common_obj, dict) and common_obj:
        normalized["objections"] = {**(normalized.get("objections") or {}), **common_obj}

    # product_details → products
    prod_details = normalized.get("product_details")
    if isinstance(prod_details, list) and prod_details:
        normalized["products"] = prod_details

    # Persist updated sub-objects back
    if scripts:
        normalized["scripts"] = scripts
    if persona:
        normalized["persona"] = persona

    # Keep business_info and target_audience as-is for reference/analytics
    return normalized

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
        # Normalize user-friendly fields into internal schema
        normalized_payload = _normalize_strategy_payload(payload.dict())

        # Store strategy in database
        strategies_collection = get_collection(COLLECTION_STRATEGIES)
        strategy_entry = {
            "strategy_id": str(uuid.uuid4()),
            "user_id": user["user_id"],
            "payload": normalized_payload,
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
            raw = json.loads(content)
            # Allow users to upload the business-aligned schema; normalize it
            strategy_data = _normalize_strategy_payload(raw if isinstance(raw, dict) else {})
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
