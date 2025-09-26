from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from core.database import get_collection
from core.auth import RoleChecker
from shared.constants import COLLECTION_STRATEGIES, COLLECTION_AGENTS
from shared.exceptions import StrategyNotFoundException, AgentNotFoundException
import uuid
from datetime import datetime

router = APIRouter()

class StrategyTransformRequest(BaseModel):
    """Strategy transform request model"""
    strategy_id: str
    agent_id: str
    transform_options: Dict[str, Any] = {}

class TransformedStrategy(BaseModel):
    """Transformed strategy model"""
    transformed_id: str
    strategy_id: str
    agent_id: str
    transformed_data: Dict[str, Any]
    created_at: datetime

class StrategyTransformer:
    """Transform strategy data into agent behavior"""
    
    @staticmethod
    async def transform_strategy_for_agent(
        strategy_id: str,
        agent_id: str,
        transform_options: Dict[str, Any] = {}
    ) -> TransformedStrategy:
        """
        Transform strategy data into agent behavior
        
        Args:
            strategy_id: Strategy ID
            agent_id: Agent ID
            transform_options: Transformation options
            
        Returns:
            TransformedStrategy
        """
        try:
            # Get strategy from database
            strategies_collection = get_collection(COLLECTION_STRATEGIES)
            strategy_doc = await strategies_collection.find_one({"strategy_id": strategy_id})
            
            if not strategy_doc:
                raise StrategyNotFoundException(f"Strategy {strategy_id} not found")
            
            # Get agent from database
            agents_collection = get_collection(COLLECTION_AGENTS)
            agent_doc = await agents_collection.find_one({"agent_id": agent_id})
            
            if not agent_doc:
                raise AgentNotFoundException(f"Agent {agent_id} not found")
            
            # Transform strategy data
            transformed_data = StrategyTransformer._apply_transformations(
                strategy_doc["payload"],
                transform_options
            )
            
            # Create transformed strategy entry
            transformed_strategy = TransformedStrategy(
                transformed_id=str(uuid.uuid4()),
                strategy_id=strategy_id,
                agent_id=agent_id,
                transformed_data=transformed_data,
                created_at=datetime.utcnow()
            )
            
            # Store transformed strategy in database
            # TODO: Implement actual database storage for transformed strategies
            
            return transformed_strategy
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to transform strategy: {str(e)}"
            )
    
    @staticmethod
    def _apply_transformations(
        strategy_data: Dict[str, Any],
        transform_options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply transformations to strategy data
        
        Args:
            strategy_data: Original strategy data
            transform_options: Transformation options
            
        Returns:
            Transformed strategy data
        """
        # Apply default transformations
        transformed_data = strategy_data.copy()
        
        # Add agent-specific adaptations
        if transform_options.get("personalize", False):
            transformed_data["personalized"] = True
            transformed_data["personalization_date"] = datetime.utcnow()
        
        # Apply tone adjustments
        tone = transform_options.get("tone", "professional")
        transformed_data["tone"] = tone
        
        # Apply complexity adjustments
        complexity = transform_options.get("complexity", "medium")
        transformed_data["complexity"] = complexity
        
        # Add transformation metadata
        transformed_data["transformed_at"] = datetime.utcnow()
        transformed_data["transform_version"] = "1.0"
        
        return transformed_data

@router.post("/transform", response_model=TransformedStrategy)
async def transform_strategy(
    transform_request: StrategyTransformRequest,
    user: dict = Depends(RoleChecker(["user", "admin"]))
):
    """
    Transform strategy data into agent behavior
    
    Args:
        transform_request: Transformation request
        user: Authenticated user
        
    Returns:
        TransformedStrategy
    """
    try:
        # Verify that the user owns the agent
        agents_collection = get_collection(COLLECTION_AGENTS)
        agent_doc = await agents_collection.find_one({
            "agent_id": transform_request.agent_id,
            "user_id": user["user_id"]
        })
        
        if not agent_doc:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to transform strategy for this agent"
            )
        
        # Transform strategy
        transformed_strategy = await StrategyTransformer.transform_strategy_for_agent(
            strategy_id=transform_request.strategy_id,
            agent_id=transform_request.agent_id,
            transform_options=transform_request.transform_options
        )
        
        return transformed_strategy
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to transform strategy: {str(e)}"
        )

@router.get("/transformed/{transformed_id}", response_model=TransformedStrategy)
async def get_transformed_strategy(
    transformed_id: str,
    user: dict = Depends(RoleChecker(["user", "admin"]))
):
    """
    Get transformed strategy
    
    Args:
        transformed_id: Transformed strategy ID
        user: Authenticated user
        
    Returns:
        TransformedStrategy
    """
    try:
        # In a real implementation, this would fetch from database
        # For now, we'll return a mock response
        return TransformedStrategy(
            transformed_id=transformed_id,
            strategy_id="mock_strategy_id",
            agent_id="mock_agent_id",
            transformed_data={
                "title": "Mock Transformed Strategy",
                "description": "This is a mock transformed strategy",
                "transformed_at": datetime.utcnow()
            },
            created_at=datetime.utcnow()
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch transformed strategy: {str(e)}"
        )

@router.get("/transform_options")
async def get_transform_options(
    user: dict = Depends(RoleChecker(["user", "admin"]))
):
    """
    Get available transformation options
    
    Args:
        user: Authenticated user
        
    Returns:
        Dict with transformation options
    """
    return {
        "status": "success",
        "transform_options": {
            "personalize": {
                "type": "boolean",
                "description": "Personalize strategy for specific agent"
            },
            "tone": {
                "type": "string",
                "options": ["professional", "friendly", "assertive", "consultative"],
                "description": "Adjust tone of the strategy"
            },
            "complexity": {
                "type": "string",
                "options": ["simple", "medium", "complex"],
                "description": "Adjust complexity of the strategy"
            },
            "industry": {
                "type": "string",
                "options": ["technology", "healthcare", "finance", "retail", "real_estate"],
                "description": "Optimize for specific industry"
            }
        }
    }
