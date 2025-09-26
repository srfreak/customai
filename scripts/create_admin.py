#!/usr/bin/env python3
"""
Admin user creation script for Scriza AI Platform
"""
import asyncio
import uuid
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorClient
from core.config import settings
from apps.users.models import UserRole
from passlib.context import CryptContext

# Password hashing
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

def get_password_hash(password):
    return pwd_context.hash(password)

async def create_admin_user():
    """Create an admin user"""
    try:
        # Connect to MongoDB
        client = AsyncIOMotorClient(settings.MONGODB_URL)
        db = client[settings.DATABASE_NAME]
        
        # Check if admin already exists
        existing_admin = await db.users.find_one({"email": "karnveer@scriza.in"})
        if existing_admin:
            print("Admin user already exists!")
            return
        
        # Create admin user
        admin_user = {
            "user_id": str(uuid.uuid4()),
            "email": "karnveer@scriza.in",
            "hashed_password": get_password_hash("admin123"),
            "first_name": "System",
            "last_name": "Administrator",
            "company": "Scriza AI Platform",
            "role": UserRole.ADMIN.value,
            "is_active": True,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        # Insert admin user
        result = await db.users.insert_one(admin_user)
        
        print("Admin user created successfully!")
        print("Email: karnveer@scriza.in")
        print("Password: admin123")
        print("Role: Admin")
        
    except Exception as e:
        print(f"Error creating admin user: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    asyncio.run(create_admin_user())
