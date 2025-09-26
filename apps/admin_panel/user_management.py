from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel
from typing import List, Optional
from core.auth import RoleChecker
from core.database import get_collection
from apps.users.models import UserInResponse, UserRole
from shared.constants import COLLECTION_USERS
from datetime import datetime

router = APIRouter()

class UserUpdateRequest(BaseModel):
    """User update request model"""
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    company: Optional[str] = None
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None

@router.get("/users", response_model=List[UserInResponse])
async def list_users(
    skip: int = 0,
    limit: int = 100,
    user: dict = Depends(RoleChecker(["admin"]))
):
    """List all users (admin only)"""
    try:
        users_collection = get_collection(COLLECTION_USERS)
        cursor = users_collection.find().skip(skip).limit(limit)
        users = await cursor.to_list(length=limit)
        
        return [UserInResponse(**user) for user in users]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch users: {str(e)}"
        )

@router.get("/users/{user_id}", response_model=UserInResponse)
async def get_user(
    user_id: str,
    user: dict = Depends(RoleChecker(["admin"]))
):
    """Get a specific user (admin only)"""
    try:
        users_collection = get_collection(COLLECTION_USERS)
        user_doc = await users_collection.find_one({"user_id": user_id})
        
        if not user_doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return UserInResponse(**user_doc)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch user: {str(e)}"
        )

@router.put("/users/{user_id}")
async def update_user(
    user_id: str,
    user_update: UserUpdateRequest,
    user: dict = Depends(RoleChecker(["admin"]))
):
    """Update a user (admin only)"""
    try:
        users_collection = get_collection(COLLECTION_USERS)
        
        # Build update document
        update_doc = {"updated_at": datetime.utcnow()}
        if user_update.first_name is not None:
            update_doc["first_name"] = user_update.first_name
        if user_update.last_name is not None:
            update_doc["last_name"] = user_update.last_name
        if user_update.company is not None:
            update_doc["company"] = user_update.company
        if user_update.role is not None:
            update_doc["role"] = user_update.role.value
        if user_update.is_active is not None:
            update_doc["is_active"] = user_update.is_active
        
        # Update user
        result = await users_collection.update_one(
            {"user_id": user_id},
            {"$set": update_doc}
        )
        
        if result.matched_count == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return {"status": "success", "message": "User updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update user: {str(e)}"
        )

@router.delete("/users/{user_id}")
async def delete_user(
    user_id: str,
    user: dict = Depends(RoleChecker(["admin"]))
):
    """Delete a user (admin only)"""
    try:
        users_collection = get_collection(COLLECTION_USERS)
        result = await users_collection.delete_one({"user_id": user_id})
        
        if result.deleted_count == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return {"status": "success", "message": "User deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete user: {str(e)}"
        )
