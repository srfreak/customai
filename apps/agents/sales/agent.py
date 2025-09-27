from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from apps.agents.agent_base import BaseAgent
from apps.agents.sales import services
from core.auth import RoleChecker
from core.database import get_collection
from shared.constants import (
    COLLECTION_AGENTS,
    COLLECTION_STRATEGIES,
)
from shared.exceptions import AgentNotFoundException, InvalidStrategyException
from datetime import datetime
import uuid

class SalesAgent(BaseAgent):
    """Production sales specialist built on Scrappy Singh's base capabilities."""

    def __init__(self, agent_id: str, user_id: str, name: str, persona: Optional[Dict[str, Any]] = None):
        super().__init__(agent_id, user_id, name, persona=persona)
        self.agent_type = "sales"
        self.is_trained = False
        self.objection_map: Dict[str, str] = {}
        self.rapport_phrases: List[str] = []
        self.close_phrases: List[str] = []

    # ------------------------------------------------------------------
    # Strategy / persona loading
    # ------------------------------------------------------------------
    async def train(self, strategy_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Extend base training with sales-specific assets and persistence."""
        if not strategy_payload:
            raise InvalidStrategyException("Strategy payload is required")

        await super().train(strategy_payload)
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
            {"$set": {"is_trained": True, "strategy_id": strategy_entry["strategy_id"], "updated_at": datetime.utcnow()}},
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
            response = await self.generate_response(
                user_input,
                stage=stage,
                lead_name=lead_name,
                strategy_context=self.strategy_context,
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
