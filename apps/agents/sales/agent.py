from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, HttpUrl

from apps.agents.agent_base import BaseAgent
from apps.agents.sales import services
from apps.agents.sales.business_agent_service import (
    draft_strategy_from_context,
    scrape_website_text,
    summarise_website,
)
from apps.agents.sales.catalog import (
    INDUSTRY_OPTIONS,
    USE_CASE_OPTIONS,
    resolve_industry_label,
    resolve_use_case_label,
    resolve_use_case_tone,
)
from apps.agents.sales.strategy_ingest import _normalize_strategy_payload, _extract_json_block
from core.auth import RoleChecker
from core.database import get_collection
from shared.constants import (
    COLLECTION_AGENTS,
    COLLECTION_STRATEGIES,
)
from shared.exceptions import AgentNotFoundException, InvalidStrategyException

class SalesAgent(BaseAgent):
    """Production sales specialist built on Scrappy Singh's base capabilities."""

    def __init__(self, agent_id: str, user_id: str, name: str, persona: Optional[Dict[str, Any]] = None):
        super().__init__(agent_id, user_id, name, persona=persona)
        self.agent_type = "sales"
        self.is_trained = False
        self.objection_map: Dict[str, str] = {}
        self.rapport_phrases: List[str] = []
        self.close_phrases: List[str] = []
        self.fallback_policies: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Strategy / persona loading
    # ------------------------------------------------------------------
    async def train(self, strategy_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Extend base training with sales-specific assets and persistence."""
        if not strategy_payload:
            raise InvalidStrategyException("Strategy payload is required")

        await super().train(strategy_payload)
        self._load_fallback_policies(strategy_payload)
        self.objection_map = strategy_payload.get("objections", {})
        self.rapport_phrases = strategy_payload.get("rapport", [
            "It's great speaking with you today!",
            "I appreciate you sharing that."
        ])
        self.close_phrases = strategy_payload.get("closing_techniques", [])
        fallback = strategy_payload.get("fallback_scenarios", {})
        if fallback:
            self.register_subagent("fallback_bot", FallbackAgent(self.agent_id, self.user_id, f"{self.name}-fallback", fallback))

        # Persist the fact the agent is trained
        strategies_collection = get_collection(COLLECTION_STRATEGIES)
        strategy_entry = {
            "strategy_id": strategy_payload.get("strategy_id", str(uuid.uuid4())),
            "agent_id": self.agent_id,
            "user_id": self.user_id,
            "payload": strategy_payload,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
        await strategies_collection.insert_one(strategy_entry)

        agents_collection = get_collection(COLLECTION_AGENTS)
        await agents_collection.update_one(
            {"agent_id": self.agent_id},
            {
                "$set": {
                    "user_id": self.user_id,
                    "name": self.name,
                    "persona": self.persona,
                    "is_trained": True,
                    "strategy_id": strategy_entry["strategy_id"],
                    "updated_at": datetime.utcnow(),
                },
                "$setOnInsert": {
                    "created_at": datetime.utcnow(),
                },
            },
            upsert=True,
        )

        self.is_trained = True
        return {
            "status": "success",
            "message": f"Sales agent {self.name} trained successfully",
            "agent_id": self.agent_id,
            "persona": self.persona,
            "goals": self.goals,
        }

    def _load_fallback_policies(self, payload: Dict[str, Any]) -> None:
        policies = payload.get("fallback_policies") if isinstance(payload, dict) else None
        if isinstance(policies, list):
            self.fallback_policies = [p for p in policies if isinstance(p, dict)]
        else:
            self.fallback_policies = []

    def attach_strategy(self, strategy: Dict[str, Any]) -> None:
        super().attach_strategy(strategy)
        payload = strategy.get("payload") if isinstance(strategy, dict) and "payload" in strategy else strategy
        if isinstance(payload, dict):
            self._load_fallback_policies(payload)

    # ------------------------------------------------------------------
    # Conversational hooks
    # ------------------------------------------------------------------
    async def handle_objection(self, user_input: str) -> Optional[str]:
        """Return a tailored objection handler if a matching keyword is present."""
        for keyword, response in self.objection_map.items():
            if keyword.lower() in user_input.lower():
                return response
        return None

    def build_rapport(self, caller_name: Optional[str] = None) -> str:
        """Generate a warm rapport-building line that mirrors the caller."""
        base = self.rapport_phrases[0] if self.rapport_phrases else "I’m really glad we could connect."
        if caller_name:
            return f"{base} {caller_name}, tell me a little more about your goals."
        return base

    def assumptive_close(self) -> str:
        """Offer an assumptive close phrase to move the conversation towards booking."""
        if self.close_phrases:
            return self.close_phrases[0]
        return "Let’s lock in a quick demo so you can experience this first hand — what works better, tomorrow morning or afternoon?"

    async def follow_up(self, lead_name: str, channel: str = "sms") -> Dict[str, Any]:
        """Craft a post-call follow-up message ready to push to downstream channels."""
        message = f"Hi {lead_name}, thanks for the time today. I’ll send over the details we discussed."
        return {"channel": channel, "message": message, "status": "queued"}

    async def generate_sales_response(self, user_input: str, stage: Optional[str] = None, lead_name: Optional[str] = None) -> Dict[str, Any]:
        """Create a context-aware sales reply and persist the exchange to memory."""
        objection_reply = await self.handle_objection(user_input)
        if objection_reply:
            assistant_text = objection_reply
        else:
            # Use critical model for in-call sales replies
            from core.config import settings
            response = await self.generate_response(
                user_input,
                stage=stage,
                lead_name=lead_name,
                strategy_context=self.strategy_context,
                model=getattr(settings, "OPENAI_MODEL_CRITICAL", getattr(settings, "OPENAI_MODEL", "gpt-4o")),
            )
            assistant_text = response["text"]

        dest_mem = await self.log_turn(
            agent_text=assistant_text,
            user_text=user_input,
            extra={"stage": stage, "lead_name": lead_name},
        )
        return {"text": assistant_text, "memory_id": dest_mem}

    async def respond_to_prompt(self, prompt: str) -> str:
        """Convenience wrapper used by tests and tooling to fetch a raw reply."""
        result = await self.generate_sales_response(user_input=prompt)
        return result["text"]


class FallbackAgent(BaseAgent):
    """Keyword-driven backup agent for safety nets and escalations."""
    def __init__(self, agent_id: str, user_id: str, name: str, fallback_map: Dict[str, str]):
        super().__init__(agent_id, user_id, name)
        self.fallback_map = fallback_map

    async def train(self, strategy_payload: Dict[str, Any]) -> Dict[str, Any]:  # pragma: no cover - fallback is static
        return {"status": "noop"}

    async def respond_to_prompt(self, prompt: str) -> str:
        for keyword, response in self.fallback_map.items():
            if keyword.lower() in prompt.lower():
                return response
        return "Thanks for the question! I'll get a success manager to share the precise details right after this call."


router = APIRouter()


class AgentSummary(BaseModel):
    agent_id: str
    name: Optional[str]
    is_trained: bool = False
    strategy_id: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class CreateAgentRequest(BaseModel):
    name: str
    agent_id: Optional[str] = None
    persona: Optional[Dict[str, Any]] = None


class CreateAgentResponse(BaseModel):
    status: str
    agent: AgentSummary


class BusinessAgentRequest(BaseModel):
    industry: str
    use_case: str
    name: str
    main_goal: str
    website_url: Optional[HttpUrl] = None
    ai_generate: bool = True
    strategy_notes: Optional[str] = None
    persona_voice_id: Optional[str] = None
    persona_locale: Optional[str] = None


class BusinessAgentResponse(BaseModel):
    status: str
    agent_id: str
    strategy: Dict[str, Any]
    knowledge_summary: Optional[str] = None
    industry: str
    use_case: str


class GenerateReplyRequest(BaseModel):
    lead_input: str
    previous_stage: Optional[str] = None
    agent_name: Optional[str] = None


class GenerateReplyResponse(BaseModel):
    reply: str
    next_stage: str
    used_strategy: Dict[str, Any]


STAGE_SEQUENCE = ["greeting", "pitch", "faqs", "objections", "closing"]


def _determine_next_stage(previous_stage: Optional[str]) -> str:
    if not previous_stage:
        return STAGE_SEQUENCE[0]
    try:
        idx = STAGE_SEQUENCE.index(previous_stage)
    except ValueError:
        return STAGE_SEQUENCE[0]
    return STAGE_SEQUENCE[min(idx + 1, len(STAGE_SEQUENCE) - 1)]


@router.post("/generate_reply", response_model=GenerateReplyResponse)
async def generate_reply(
    request: GenerateReplyRequest,
    user: dict = Depends(RoleChecker(["user", "admin"])),
):
    """Generate the agent's reply using OpenAI, conditioned on stored strategy."""
    strategy_doc = await services.fetch_latest_strategy(user_id=user["user_id"])
    if not strategy_doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No strategy found for user",
        )

    strategy_payload = strategy_doc.get("payload", {})
    context = await services.generate_strategy_context(strategy_payload)
    agent_name = request.agent_name or strategy_payload.get("title") or "Sales AI Assistant"
    messages = [
        {
            "role": "system",
            "content": (
                f"You are {agent_name}, a persuasive sales representative. Use the following strategy to respond succinctly.\n"
                f"Strategy:\n{context}\nRespond in plain English without mentioning the strategy explicitly."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Previous stage: {request.previous_stage or 'none'}\n"
                f"Lead said: {request.lead_input}\n"
                "Provide the next spoken reply."
            ),
        },
    ]

    reply = await services.call_openai_chat(messages)
    next_stage = _determine_next_stage(request.previous_stage)

    return GenerateReplyResponse(
        reply=reply,
        next_stage=next_stage,
        used_strategy=strategy_payload,
    )


@router.get("/agents", response_model=List[AgentSummary])
async def list_sales_agents(
    user: dict = Depends(RoleChecker(["user", "admin"]))
):
    agents_collection = get_collection(COLLECTION_AGENTS)
    cursor = agents_collection.find({"user_id": user["user_id"]}).sort("updated_at", -1)
    agents = await cursor.to_list(length=200)
    summaries: List[AgentSummary] = []
    for doc in agents:
        summaries.append(
            AgentSummary(
                agent_id=doc.get("agent_id"),
                name=doc.get("name"),
                is_trained=bool(doc.get("is_trained")),
                strategy_id=doc.get("strategy_id"),
                created_at=doc.get("created_at"),
                updated_at=doc.get("updated_at"),
            )
        )
    return summaries


@router.post("/agents", response_model=CreateAgentResponse, status_code=status.HTTP_201_CREATED)
async def create_sales_agent(
    payload: CreateAgentRequest,
    user: dict = Depends(RoleChecker(["user", "admin"]))
):
    agents_collection = get_collection(COLLECTION_AGENTS)
    agent_id = payload.agent_id or str(uuid.uuid4())
    existing = await agents_collection.find_one({"agent_id": agent_id})
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Agent with this ID already exists",
        )

    now = datetime.utcnow()
    agent_doc = {
        "agent_id": agent_id,
        "user_id": user["user_id"],
        "name": payload.name,
        "persona": payload.persona or {},
        "is_trained": False,
        "strategy_id": None,
        "created_at": now,
        "updated_at": now,
    }

    await agents_collection.insert_one(agent_doc)

    summary = AgentSummary(
        agent_id=agent_id,
        name=payload.name,
        is_trained=False,
        strategy_id=None,
        created_at=now,
        updated_at=now,
    )

    return CreateAgentResponse(status="success", agent=summary)


@router.get("/agents/metadata")
async def get_agent_metadata(
    user: dict = Depends(RoleChecker(["user", "admin"]))
):
    # Simple endpoint to keep front-end options in sync
    return {
        "industries": INDUSTRY_OPTIONS,
        "use_cases": USE_CASE_OPTIONS,
    }


@router.post("/agents/business", response_model=BusinessAgentResponse, status_code=status.HTTP_201_CREATED)
async def create_business_agent(
    payload: BusinessAgentRequest,
    user: dict = Depends(RoleChecker(["user", "admin"]))
):
    industry_ids = {item["id"] for item in INDUSTRY_OPTIONS}
    use_case_ids = {item["id"] for item in USE_CASE_OPTIONS}
    if payload.industry not in industry_ids:
        raise HTTPException(status_code=400, detail="Invalid industry selection")
    if payload.use_case not in use_case_ids:
        raise HTTPException(status_code=400, detail="Invalid use case selection")

    knowledge_summary: Optional[str] = None
    if payload.website_url:
        raw_text = await scrape_website_text(str(payload.website_url))
        knowledge_summary = await summarise_website(raw_text, payload.industry, payload.use_case)

    strategy_json: Dict[str, Any]
    if payload.ai_generate:
        completion_text = await draft_strategy_from_context(
            industry_id=payload.industry,
            use_case_id=payload.use_case,
            goal=payload.main_goal,
            persona_name=payload.name,
            notes=payload.strategy_notes,
            knowledge_summary=knowledge_summary,
        )
        try:
            draft_raw = _extract_json_block(completion_text)
            strategy_json = json.loads(draft_raw)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Failed to parse AI strategy output: {exc}"
            ) from exc
    else:
        industry_label = resolve_industry_label(payload.industry)
        use_case_label = resolve_use_case_label(payload.use_case)
        tone_hint = resolve_use_case_tone(payload.use_case)
        strategy_json = {
            "title": f"{industry_label} {use_case_label} Playbook",
            "description": payload.strategy_notes or payload.main_goal,
            "goals": [payload.main_goal],
            "persona": {
                "name": payload.name,
                "tone": tone_hint,
                "description": f"Represents a {use_case_label.lower()} specialist for {industry_label} clients.",
                "tone_override": tone_hint,
            },
            "scripts": {
                "greeting": f"Hi, this is {payload.name}. I'm here to help you with {industry_label.lower()} {use_case_label.lower()} needs.",
                "pitch": f"We work with {industry_label.lower()} teams to achieve {payload.main_goal}.",
            },
            "objections": {},
            "closing_techniques": [
                "Would you like to schedule the next step now or should I send follow-up options?"
            ],
            "fallback_scenarios": {},
            "fallback_policies": [],
        }

    persona = strategy_json.setdefault("persona", {})
    persona.setdefault("name", payload.name)
    persona.setdefault("tone_override", resolve_use_case_tone(payload.use_case))
    if payload.persona_locale:
        persona.setdefault("locale", payload.persona_locale)
    if payload.persona_voice_id:
        persona.setdefault("voice_id", payload.persona_voice_id)

    goals_value = strategy_json.get("goals")
    if isinstance(goals_value, dict):
        goals = [str(v) for v in goals_value.values() if v]
    elif isinstance(goals_value, list):
        goals = [str(g) for g in goals_value if g]
    elif isinstance(goals_value, str):
        goals = [goals_value]
    else:
        goals = []
    if payload.main_goal and payload.main_goal not in goals:
        goals.append(payload.main_goal)
    strategy_json["goals"] = goals

    strategy_json.setdefault("description", payload.main_goal)
    strategy_json.setdefault("title", f"{payload.name} Strategy")

    strategy_id = strategy_json.setdefault("strategy_id", str(uuid.uuid4()))
    normalized_strategy = _normalize_strategy_payload(strategy_json)
    normalized_strategy["strategy_id"] = strategy_id
    normalized_strategy.setdefault("persona", {})["name"] = payload.name

    agent_id = str(uuid.uuid4())
    sales_agent = SalesAgent(
        agent_id=agent_id,
        user_id=user["user_id"],
        name=payload.name,
        persona=normalized_strategy.get("persona"),
    )
    await sales_agent.train(normalized_strategy)

    agents_collection = get_collection(COLLECTION_AGENTS)
    await agents_collection.update_one(
        {"agent_id": sales_agent.agent_id},
        {
            "$set": {
                "user_id": user["user_id"],
                "industry_id": payload.industry,
                "industry": resolve_industry_label(payload.industry),
                "use_case_id": payload.use_case,
                "use_case": resolve_use_case_label(payload.use_case),
                "goal": payload.main_goal,
                "website_url": str(payload.website_url) if payload.website_url else None,
                "knowledge_summary": knowledge_summary,
                "strategy_notes": payload.strategy_notes,
                "pretrained": payload.ai_generate,
                "updated_at": datetime.utcnow(),
            },
            "$setOnInsert": {
                "created_at": datetime.utcnow(),
            },
        },
        upsert=True,
    )

    return BusinessAgentResponse(
        status="success",
        agent_id=sales_agent.agent_id,
        strategy=normalized_strategy,
        knowledge_summary=knowledge_summary,
        industry=resolve_industry_label(payload.industry),
        use_case=resolve_use_case_label(payload.use_case),
    )
