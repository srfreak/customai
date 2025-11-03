from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel
from typing import List, Optional
from core.auth import create_access_token, verify_token, RoleChecker
from core.database import get_collection
from apps.users.models import UserRegistration, UserInDB, UserInResponse, UserRole
from shared.constants import COLLECTION_USERS
from shared.utils import is_valid_email
import uuid
from datetime import datetime, timedelta
from passlib.context import CryptContext

router = APIRouter()


pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    user_id: str
    role: str

class LoginRequest(BaseModel):
    email: str
    password: str

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

async def get_user_by_email(email: str):
    """Get user by email"""
    users_collection = get_collection(COLLECTION_USERS)
    user_doc = await users_collection.find_one({"email": email})
    if user_doc:
        return UserInDB(**user_doc)
    return None

async def authenticate_user(email: str, password: str):
    """Authenticate user"""
    user = await get_user_by_email(email)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

@router.post("/register", response_model=UserInResponse)
async def register_user(user_data: UserRegistration):
    """Register a new user"""
    # Validate email
    if not is_valid_email(user_data.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid email address"
        )
    
    # Check if user already exists
    existing_user = await get_user_by_email(user_data.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this email already exists"
        )
    
    # Hash password
    hashed_password = get_password_hash(user_data.password)
    
    # Create user document
    user_id = str(uuid.uuid4())
    user_doc = {
        "user_id": user_id,
        "email": user_data.email,
        "hashed_password": hashed_password,
        "first_name": user_data.first_name,
        "last_name": user_data.last_name,
        "company": user_data.company,
        "role": UserRole.USER.value,
        "is_active": True,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    
    # Store user in database
    users_collection = get_collection(COLLECTION_USERS)
    await users_collection.insert_one(user_doc)
    
    # Return user response
    return UserInResponse(
        user_id=user_id,
        email=user_doc["email"],
        first_name=user_doc["first_name"],
        last_name=user_doc["last_name"],
        company=user_doc["company"],
        role=UserRole(user_doc["role"]),
        is_active=user_doc["is_active"],
        created_at=user_doc["created_at"],
        updated_at=user_doc["updated_at"]
    )

@router.post("/login", response_model=Token)
async def login_user(login_request: LoginRequest):
    """Login user and return access token"""
    user = await authenticate_user(login_request.email, login_request.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=30)
    # Ensure enum is encoded as its string value
    role_value = user.role.value if hasattr(user.role, "value") else str(user.role)
    access_token = create_access_token(
        data={"sub": user.user_id, "role": role_value},
        expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

async def _get_user_by_id(user_id: str) -> Optional[UserInDB]:
    users_collection = get_collection(COLLECTION_USERS)
    doc = await users_collection.find_one({"user_id": user_id})
    return UserInDB(**doc) if doc else None


@router.get("/profile", response_model=UserInResponse)
async def get_user_profile(user: dict = Depends(verify_token)):
    """Get user profile"""
    user_doc = await _get_user_by_id(user["user_id"])  # lookup by user_id from JWT
    if not user_doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return UserInResponse(
        user_id=user_doc.user_id,
        email=user_doc.email,
        first_name=user_doc.first_name,
        last_name=user_doc.last_name,
        company=user_doc.company,
        role=UserRole(user_doc.role),
        is_active=user_doc.is_active,
        created_at=user_doc.created_at,
        updated_at=user_doc.updated_at
    )

@router.put("/profile", response_model=UserInResponse)
async def update_user_profile(
    user_data: dict,
    user: dict = Depends(verify_token)
):
    """Update user profile"""
    # Update user in database
    users_collection = get_collection(COLLECTION_USERS)
    result = await users_collection.update_one(
        {"user_id": user["user_id"]},
        {
            "$set": {
                "first_name": user_data.get("first_name", ""),
                "last_name": user_data.get("last_name", ""),
                "company": user_data.get("company"),
                "updated_at": datetime.utcnow()
            }
        }
    )
    
    if result.matched_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Get updated user
    user_doc = await users_collection.find_one({"user_id": user["user_id"]})
    
    return UserInResponse(
        user_id=user_doc["user_id"],
        email=user_doc["email"],
        first_name=user_doc["first_name"],
        last_name=user_doc["last_name"],
        company=user_doc["company"],
        role=UserRole(user_doc["role"]),
        is_active=user_doc["is_active"],
        created_at=user_doc["created_at"],
        updated_at=user_doc["updated_at"]
    )
