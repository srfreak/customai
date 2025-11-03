from __future__ import annotations

from typing import Any, Dict, Optional, List

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from core.auth import RoleChecker
from shared.presets import find_preset, load_presets, upsert_preset, delete_preset
from core.config import settings
from core.database import get_collection
from shared.constants import COLLECTION_AGENT_TEMPLATES
from datetime import datetime

router = APIRouter()


@router.get("/templates/preset")
async def get_preset(
    industry: str = Query(..., description="Industry label, case-insensitive"),
    use_case: str = Query(..., description="Use case label, case-insensitive"),
    user: Optional[dict] = Depends(RoleChecker(["user", "admin"]))
) -> Dict[str, Any]:
    preset = find_preset(industry, use_case)
    source = "preset"
    if not preset:
        # Build a gentle default instead of returning 404 to keep UX smooth
        persona_name = "Scrappy Agent"
        tone = "friendly, helpful"
        voice_id = getattr(settings, "ELEVENLABS_DEFAULT_VOICE_ID", "") or None
        locale = getattr(settings, "DEFAULT_PERSONA_LOCALE", "en")
        goal_key = use_case.strip().lower().replace(" ", "_") if use_case else "assist"
        preset = {
            "persona": {
                "name": persona_name,
                "tone": tone,
                "voice_id": voice_id,
                "locale": locale,
            },
            "greeting": f"Hi, this is {persona_name}.",
            "goals": [goal_key],
        }
        source = "default"

    # Log usage for analytics if collection exists
    try:
        coll = get_collection(COLLECTION_AGENT_TEMPLATES)
        await coll.insert_one(
            {
                "industry": industry,
                "use_case": use_case,
                "user_id": (user or {}).get("user_id") if isinstance(user, dict) else None,
                "created_at": datetime.utcnow(),
            }
        )
    except Exception:
        # Non-fatal
        pass

    return {"industry": industry, "use_case": use_case, "preset": preset, "source": source}


@router.get("/templates/presets")
async def list_presets(
    user: Optional[dict] = Depends(RoleChecker(["user", "admin"]))
) -> Dict[str, Any]:
    """Return the full preset tree for UI consumption."""
    return {"templates": load_presets()}


class PresetPayload(BaseModel):
    industry: str = Field(..., description="Industry label")
    use_case: str = Field(..., description="Use case label")
    preset: Dict[str, Any] = Field(..., description="Preset payload (persona, greeting, pitch, faqs, objections, goals)")


@router.post("/templates/preset", status_code=status.HTTP_201_CREATED)
async def upsert_preset_admin(
    payload: PresetPayload,
    _: dict = Depends(RoleChecker(["admin"]))
) -> Dict[str, Any]:
    """Create or update a preset (admin only)."""
    stored = upsert_preset(payload.industry, payload.use_case, payload.preset)
    return {"status": "ok", "industry": payload.industry, "use_case": payload.use_case, "preset": stored}


@router.delete("/templates/preset", status_code=status.HTTP_200_OK)
async def delete_preset_admin(
    industry: str = Query(...),
    use_case: str = Query(...),
    _: dict = Depends(RoleChecker(["admin"]))
) -> Dict[str, Any]:
    ok = delete_preset(industry, use_case)
    if not ok:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Preset not found")
    return {"status": "ok", "industry": industry, "use_case": use_case}
