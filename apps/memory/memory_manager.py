from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from core.auth import RoleChecker
from core.database import get_collection
from shared.constants import COLLECTION_MEMORY_LOGS, COLLECTION_AGENTS
from shared.exceptions import MemoryOperationException
import uuid
from datetime import datetime

router = APIRouter()

class MemoryEntry(BaseModel):
    """Memory entry model"""
    memory_id: str
    agent_id: str
    user_id: str
    data: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

class MemoryOperationRequest(BaseModel):
    """Memory operation request model"""
    agent_id: str
    operation: str  # "store", "retrieve", "clear"
    data: Optional[Dict[str, Any]] = None
    memory_id: Optional[str] = None

class MemoryService:
    """Memory service for agent memory operations"""
    
    @staticmethod
    async def store_memory(
        user_id: str,
        agent_id: str,
        data: Dict[str, Any]
    ) -> str:
        """
        Store memory for an agent
        
        Args:
            user_id: User ID
            agent_id: Agent ID
            data: Memory data
            
        Returns:
            Memory ID
        """
        try:
            memory_collection = get_collection(COLLECTION_MEMORY_LOGS)
            memory_entry = {
                "memory_id": str(uuid.uuid4()),
                "user_id": user_id,
                "agent_id": agent_id,
                "data": data,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            result = await memory_collection.insert_one(memory_entry)
            return memory_entry["memory_id"]
        except Exception as e:
            raise MemoryOperationException(f"Failed to store memory: {str(e)}")
    
    @staticmethod
    async def retrieve_memory(
        user_id: str,
        agent_id: str,
        memory_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve memory for an agent
        
        Args:
            user_id: User ID
            agent_id: Agent ID
            memory_id: Specific memory ID to retrieve (optional)
            
        Returns:
            List of memory entries
        """
        try:
            memory_collection = get_collection(COLLECTION_MEMORY_LOGS)
            
            if memory_id:
                # Retrieve specific memory
                memory_entry = await memory_collection.find_one({
                    "memory_id": memory_id,
                    "user_id": user_id,
                    "agent_id": agent_id
                })
                
                if not memory_entry:
                    return []
                if "_id" in memory_entry:
                    memory_entry["_id"] = str(memory_entry["_id"])
                return [memory_entry]
            else:
                # Retrieve all memory for agent
                cursor = memory_collection.find({
                    "user_id": user_id,
                    "agent_id": agent_id
                }).sort("created_at", -1)
                
                memories = await cursor.to_list(length=100)  # Limit to 100 for performance
                for item in memories:
                    if "_id" in item:
                        item["_id"] = str(item["_id"])
                return memories
        except Exception as e:
            raise MemoryOperationException(f"Failed to retrieve memory: {str(e)}")
    
    @staticmethod
    async def clear_memory(
        user_id: str,
        agent_id: str,
        memory_id: Optional[str] = None
    ) -> bool:
        """
        Clear memory for an agent
        
        Args:
            user_id: User ID
            agent_id: Agent ID
            memory_id: Specific memory ID to clear (optional)
            
        Returns:
            Bool indicating success
        """
        try:
            memory_collection = get_collection(COLLECTION_MEMORY_LOGS)
            
            if memory_id:
                # Clear specific memory
                result = await memory_collection.delete_one({
                    "memory_id": memory_id,
                    "user_id": user_id,
                    "agent_id": agent_id
                })
                
                return result.deleted_count > 0
            else:
                # Clear all memory for agent
                result = await memory_collection.delete_many({
                    "user_id": user_id,
                    "agent_id": agent_id
                })
                
                return True
        except Exception as e:
            raise MemoryOperationException(f"Failed to clear memory: {str(e)}")

# Initialize memory service
memory_service = MemoryService()

@router.post("/operate")
async def operate_memory(
    operation_request: MemoryOperationRequest,
    user: dict = Depends(RoleChecker(["user", "admin"]))
):
    """
    Perform memory operation (store, retrieve, clear)
    
    Args:
        operation_request: Memory operation request
        user: Authenticated user
        
    Returns:
        Dict with operation result
    """
    try:
        if operation_request.operation == "store":
            if not operation_request.data:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Data is required for store operation"
                )
            
            memory_id = await memory_service.store_memory(
                user_id=user["user_id"],
                agent_id=operation_request.agent_id,
                data=operation_request.data
            )
            
            return {
                "status": "success",
                "operation": "store",
                "memory_id": memory_id,
                "message": "Memory stored successfully"
            }
        
        elif operation_request.operation == "retrieve":
            memories = await memory_service.retrieve_memory(
                user_id=user["user_id"],
                agent_id=operation_request.agent_id,
                memory_id=operation_request.memory_id
            )
            
            return {
                "status": "success",
                "operation": "retrieve",
                "memories": memories,
                "count": len(memories)
            }
        
        elif operation_request.operation == "clear":
            result = await memory_service.clear_memory(
                user_id=user["user_id"],
                agent_id=operation_request.agent_id,
                memory_id=operation_request.memory_id
            )
            
            if result:
                return {
                    "status": "success",
                    "operation": "clear",
                    "message": "Memory cleared successfully"
                }
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Memory not found"
                )
        
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid operation: {operation_request.operation}"
            )
    except MemoryOperationException as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/list/{agent_id}")
async def list_memories(
    agent_id: str,
    user: dict = Depends(RoleChecker(["user", "admin"]))
):
    """
    List all memories for an agent
    
    Args:
        agent_id: Agent ID
        user: Authenticated user
        
    Returns:
        Dict with memory list
    """
    try:
        memories = await memory_service.retrieve_memory(
            user_id=user["user_id"],
            agent_id=agent_id
        )
        
        return {
            "status": "success",
            "memories": memories,
            "count": len(memories)
        }
    except MemoryOperationException as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.delete("/delete/{agent_id}")
async def delete_all_memories(
    agent_id: str,
    user: dict = Depends(RoleChecker(["user", "admin"]))
):
    """
    Delete all memories for an agent (GDPR compliant)
    
    Args:
        agent_id: Agent ID
        user: Authenticated user
        
    Returns:
        Dict with deletion result
    """
    try:
        result = await memory_service.clear_memory(
            user_id=user["user_id"],
            agent_id=agent_id
        )
        
        if result:
            return {
                "status": "success",
                "message": "All memories deleted successfully"
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No memories found for this agent"
            )
    except MemoryOperationException as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/{agent_name}")
async def get_call_memories_by_agent_name(
    agent_name: str,
    user: dict = Depends(RoleChecker(["user", "admin"])),
):
    """Fetch stored call memories for an agent identified by name."""
    agents_collection = get_collection(COLLECTION_AGENTS)
    agent_doc = await agents_collection.find_one(
        {"name": agent_name, "user_id": user["user_id"]}
    )
    if not agent_doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found for user",
        )

    memories = await memory_service.retrieve_memory(
        user_id=user["user_id"],
        agent_id=agent_doc.get("agent_id"),
    )
    return {
        "status": "success",
        "agent_id": agent_doc.get("agent_id"),
        "memories": memories,
        "count": len(memories),
    }
