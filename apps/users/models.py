from typing import Optional
from pydantic import BaseModel, EmailStr
from enum import Enum
from datetime import datetime
from typing import List

class UserRole(str, Enum):
    ADMIN = "admin"
    USER = "user"

class UserRegistration(BaseModel):
    """User registration model"""
    email: str
    password: str
    first_name: str
    last_name: str
    company: Optional[str] = None

class UserInDB(BaseModel):
    """User model for database storage"""
    user_id: str
    email: str
    hashed_password: str
    first_name: str
    last_name: str
    company: Optional[str] = None
    role: UserRole = UserRole.USER
    is_active: bool = True
    created_at: datetime
    updated_at: datetime

class UserInResponse(BaseModel):
    """User model for API responses"""
    user_id: str
    email: str
    first_name: str
    last_name: str
    company: Optional[str] = None
    role: UserRole
    is_active: bool
    created_at: datetime
    updated_at: datetime
