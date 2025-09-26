from typing import Dict, Any, List, Optional

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

class SalesAgent(BaseAgent):
    """Sales agent implementation"""
    
    def __init__(self, agent_id: str, user_id: str, name: str):
        super().__init__(agent_id, user_id, name)
        self.agent_type = "sales"
        self.strategy = None
        self.is_trained = False
        
    async def train(self, strategy_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train the sales agent with provided strategy
        
        Args:
            strategy_payload: Strategy data to train the agent
            
        Returns:
            Dict with training result
        """
        # Validate strategy payload
        if not strategy_payload or not isinstance(strategy_payload, dict):
            raise InvalidStrategyException("Invalid strategy payload")
        
        # Store strategy in database
        strategies_collection = get_collection(COLLECTION_STRATEGIES)
        strategy_entry = {
            "strategy_id": strategy_payload.get("strategy_id", ""),
            "agent_id": self.agent_id,
            "user_id": self.user_id,
            "payload": strategy_payload,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
        
        await strategies_collection.insert_one(strategy_entry)
        
        # Update agent with strategy
        self.strategy = strategy_payload
        self.is_trained = True
        
        # Update agent in database
        agents_collection = get_collection(COLLECTION_AGENTS)
        await agents_collection.update_one(
            {"agent_id": self.agent_id},
            {
                "$set": {
                    "is_trained": True,
                    "strategy_id": strategy_payload.get("strategy_id", ""),
                    "updated_at": self.updated_at
                }
            }
        )
        
        return {
            "status": "success",
            "message": f"Sales agent {self.name} trained successfully",
            "agent_id": self.agent_id
        }
    
    async def respond_to_prompt(self, prompt: str) -> str:
        """
        Generate sales response to a given prompt
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated sales response
        """
        # TODO: Implement actual AI response generation
        # This is a mock implementation for now
        
        if not self.is_trained:
            return "I'm still learning. Please provide more information about your products or services."
        
        # Simple rule-based response for demo purposes
        prompt_lower = prompt.lower()
        
        if "price" in prompt_lower or "cost" in prompt_lower:
            return "Our pricing is competitive and tailored to your specific needs. Can you tell me more about your requirements?"
        elif "product" in prompt_lower or "service" in prompt_lower:
            return "We offer a range of solutions designed to meet your business needs. What specific challenges are you looking to solve?"
        elif "demo" in prompt_lower or "example" in prompt_lower:
            return "I'd be happy to provide a demonstration. When would be a convenient time for you?"
        elif "contact" in prompt_lower or "speak" in prompt_lower:
            return "I can connect you with our specialist team. Would you prefer a call or email follow-up?"
        else:
            return "Thank you for your inquiry. Our team will follow up with you shortly to discuss how we can help."
    
    async def place_call(self, phone_number: str) -> Dict[str, Any]:
        """
        Place a sales call (mock implementation)
        
        Args:
            phone_number: Phone number to call
            
        Returns:
            Dict with call result
        """
        # TODO: Integrate with Twilio for real call placement
        return {
            "status": "mocked",
            "message": f"Call to {phone_number} would be placed here",
            "call_id": "mock_call_id"
        }
    
    async def synthesize_voice(self, text: str) -> bytes:
        """
        Synthesize voice from text (mock implementation)
        
        Args:
            text: Text to synthesize
            
        Returns:
            Audio bytes
        """
        # TODO: Integrate with ElevenLabs for real voice synthesis
        return b"mock_audio_data"


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
