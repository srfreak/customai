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
from urllib.parse import urlparse, urlunparse

# Password hashing
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

def get_password_hash(password):
    return pwd_context.hash(password)

async def create_admin_user():
    """Create an admin user"""
    try:
        # Connect to MongoDB (fast-fail + localhost fallback)
        uri = settings.MONGODB_URL
        client = AsyncIOMotorClient(uri, serverSelectionTimeoutMS=5000)
        try:
            await client.admin.command("ping")
        except Exception:
            parsed = urlparse(uri)
            if parsed.hostname == "mongodb":
                fallback = parsed._replace(netloc=f"localhost:{parsed.port or 27017}")
                fallback_uri = urlunparse(fallback)
                client = AsyncIOMotorClient(fallback_uri, serverSelectionTimeoutMS=4000)
                await client.admin.command("ping")
        db = client[settings.DATABASE_NAME]
        
        # Check if admin already exists
        existing_admin = await db.users.find_one({"email": "karnveer@scriza.co"})
        if existing_admin:
            print("Admin user already exists!")
            return
        
        # Create admin user
        admin_user = {
            "user_id": str(uuid.uuid4()),
            "email": "karnveer@scriza.co",
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
        print("Email: karnveer@scriza.co")
        print("Password: admin123")
        print("Role: Admin")
        
    except Exception as e:
        print(f"Error creating admin user: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    asyncio.run(create_admin_user())
