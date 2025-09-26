from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from core.database import get_collection
from shared.constants import COLLECTION_MEMORY_LOGS
import uuid
from datetime import datetime

class BaseAgent(ABC):
    """Base class for all AI agents"""
    
    def __init__(self, agent_id: str, user_id: str, name: str):
        self.agent_id = agent_id
        self.user_id = user_id
        self.name = name
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        
    @abstractmethod
    async def train(self, strategy_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train the agent with provided strategy
        
        Args:
            strategy_payload: Strategy data to train the agent
            
        Returns:
            Dict with training result
        """
        pass
    
    @abstractmethod
    async def respond_to_prompt(self, prompt: str) -> str:
        """
        Generate response to a given prompt
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated response
        """
        pass
    
    async def store_memory(self, memory_data: Dict[str, Any]) -> str:
        """
        Store agent memory
        
        Args:
            memory_data: Memory data to store
            
        Returns:
            Memory ID
        """
        memory_collection = get_collection(COLLECTION_MEMORY_LOGS)
        memory_entry = {
            "memory_id": str(uuid.uuid4()),
            "agent_id": self.agent_id,
            "user_id": self.user_id,
            "data": memory_data,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        result = await memory_collection.insert_one(memory_entry)
        return str(result.inserted_id)
    
    async def clear_memory(self, memory_id: Optional[str] = None) -> bool:
        """
        Clear agent memory
        
        Args:
            memory_id: Specific memory ID to clear, if None clears all
            
        Returns:
            True if successful
        """
        memory_collection = get_collection(COLLECTION_MEMORY_LOGS)
        
        if memory_id:
            result = await memory_collection.delete_one(
                {"memory_id": memory_id, "agent_id": self.agent_id}
            )
            return result.deleted_count > 0
        else:
            # Clear all memory for this agent
            result = await memory_collection.delete_many(
                {"agent_id": self.agent_id}
            )
            return True
    
    async def get_memory(self, memory_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve agent memory
        
        Args:
            memory_id: Specific memory ID to retrieve, if None retrieves all
            
        Returns:
            List of memory entries
        """
        memory_collection = get_collection(COLLECTION_MEMORY_LOGS)
        
        if memory_id:
            memory_entry = await memory_collection.find_one(
                {"memory_id": memory_id, "agent_id": self.agent_id}
            )
            return [memory_entry] if memory_entry else []
        else:
            # Get all memory for this agent
            cursor = memory_collection.find({"agent_id": self.agent_id})
            memories = await cursor.to_list(length=100)  # Limit to 100 for performance
            return memories
