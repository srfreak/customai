from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status, Depends
from typing import Dict, Any, Optional, List
import json
import logging
from pydantic import BaseModel
from core.database import get_collection
from core.auth import RoleChecker
from shared.constants import COLLECTION_STRATEGIES
from shared.exceptions import InvalidStrategyException
import uuid
from datetime import datetime
from apps.agents.sales import services

router = APIRouter()
logger = logging.getLogger(__name__)

class PersonaConfig(BaseModel):
    name: Optional[str] = None
    tone: Optional[str] = "friendly, empathetic, persuasive"
    description: Optional[str] = None
    voice_id: Optional[str] = None
    locale: Optional[str] = "en-IN"
    tone_override: Optional[str] = None


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
    fallback_policies: Optional[List[Dict[str, Any]]] = None
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


def _extract_json_block(text: str) -> str:
    """Best-effort extraction of JSON from LLM responses."""
    if not text:
        return text
    text = text.strip()
    if text.startswith("```"):
        # Strip code fences
        fence_end = text.find("```", 3)
        if fence_end != -1:
            inner = text[3:fence_end].strip()
            # remove optional language hint, e.g., json\n
            if "\n" in inner:
                language_hint, remainder = inner.split("\n", 1)
                if language_hint.lower() in {"json", "javascript"}:
                    inner = remainder
            text = inner
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text

class StrategyAIDraftRequest(BaseModel):
    """Request payload for LLM-generated strategies."""
    purpose: str
    buyer_persona: Optional[str] = None
    offerings: Optional[str] = None
    tone: Optional[str] = None
    language: Optional[str] = None
    # New optional hints for preset overlay
    industry: Optional[str] = None
    use_case: Optional[str] = None


class StrategyAIDraftResponse(BaseModel):
    """Response wrapper for AI drafted strategies."""
    strategy: Dict[str, Any]

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

@router.post("/strategy/ai_draft", response_model=StrategyAIDraftResponse)
async def generate_strategy_ai_draft(
    request: StrategyAIDraftRequest,
    user: dict = Depends(RoleChecker(["user", "admin"]))
):
    """
    Let GPT-4o draft a strategy JSON from natural language instructions.

    Args:
        request: High level notes and hints.
        user: Authenticated user.

    Returns:
        Machine-readable strategy payload.
    """
    try:
        # Attempt to resolve a preset first
        from shared.presets import find_preset
        preset = find_preset(request.industry, request.use_case)
        purpose = request.purpose.strip()
        if not purpose:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Purpose must not be empty"
            )

        system_prompt = (
            "You are Scriza AI's sales enablement architect. "
            "Given user instructions, produce a JSON object that matches the StrategyPayload model:\n"
            "{\n"
            '  "title": string,\n'
            '  "description": string,\n'
            '  "scripts": {\n'
            '      "greeting": string,\n'
            '      "pitch": string,\n'
            '      "faqs": {string: string},\n'
            '      "objections": {string: string},\n'
            '      "closing": [string]\n'
            "  },\n"
            '  "goals": [string],\n'
            '  "persona": {\n'
            '      "name": string,\n'
            '      "tone": string,\n'
            '      "description": string,\n'
            '      "voice_id": string,\n'
            '      "locale": string,\n'
            '      "tone_override": string\n'
            "  },\n"
            '  "objections": {string: string},\n'
            '  "closing_techniques": [string],\n'
            '  "products": [ { "name": string, "value": string } ],\n'
            '  "fallback_scenarios": {string: string},\n'
            '  "fallback_policies": [ { "trigger": string, "action": string } ]\n'
            "}\n"
            "Always ensure the JSON is valid and contains realistic sales enablement details. "
            "If the user does not provide certain hints, infer reasonable defaults."
        )

        user_prompt_parts = [
            f"Purpose / context: {purpose}"
        ]
        if request.buyer_persona:
            user_prompt_parts.append(f"Buyer persona hints: {request.buyer_persona}")
        if request.offerings:
            user_prompt_parts.append(f"Product / offering details: {request.offerings}")
        if request.tone:
            user_prompt_parts.append(f"Desired tone: {request.tone}")
        if request.language:
            user_prompt_parts.append(f"Preferred language/locale: {request.language}")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n".join(user_prompt_parts)},
        ]
        draft: Dict[str, Any] = {}
        # If there are meaningful hints or no preset, invoke LLM for a draft
        have_hints = any([request.buyer_persona, request.offerings, request.tone, request.language])
        if (preset is None) or have_hints:
            completion = await services.call_openai_chat(
                messages,
                stream=False,
                persona=None,
                goals=None,
                temperature=0.7,
            )
            try:
                draft_raw = _extract_json_block(completion)
                draft = json.loads(draft_raw)
            except json.JSONDecodeError as exc:
                logger.warning("AI draft JSON decode failed. Raw completion: %s", completion)
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Model returned invalid JSON: {exc}"
                ) from exc

        # Normalize and then overlay preset if available
        normalized = _normalize_strategy_payload(draft or {})
        if preset:
            persona = normalized.get("persona") or {}
            p_persona = preset.get("persona") or {}
            if p_persona:
                persona = {**p_persona, **persona}
                normalized["persona"] = persona
            scripts = normalized.get("scripts") or {}
            if preset.get("greeting"):
                scripts["greeting"] = preset["greeting"]
            if preset.get("pitch"):
                scripts["pitch"] = preset["pitch"]
            if preset.get("faqs") and isinstance(preset["faqs"], dict):
                faqs = dict(scripts.get("faqs") or {})
                faqs.update({k: str(v) for k, v in preset["faqs"].items()})
                scripts["faqs"] = faqs
            if preset.get("objections") and isinstance(preset["objections"], dict):
                objections = dict(normalized.get("objections") or {})
                objections.update({k: str(v) for k, v in preset["objections"].items()})
                normalized["objections"] = objections
            normalized["scripts"] = scripts
            # goals union
            goals = list(dict.fromkeys([*(normalized.get("goals") or []), *list(preset.get("goals") or [])]))
            if goals:
                normalized["goals"] = goals
            # Store selected industry/use_case for analytics
            if request.industry:
                normalized.setdefault("business_info", {})["industry"] = request.industry
            if request.use_case:
                normalized.setdefault("business_info", {})["use_case"] = request.use_case

        normalized.setdefault("title", "Untitled Strategy")
        normalized.setdefault("description", purpose[:120])

        return StrategyAIDraftResponse(strategy=normalized)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate strategy: {str(e)}"
        )
