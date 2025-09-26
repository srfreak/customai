from motor.motor_asyncio import AsyncIOMotorClient
from core.config import settings
from typing import Optional

# Global MongoDB client
client: Optional[AsyncIOMotorClient] = None

# Database instance
db = None

async def connect_to_mongo():
    """Connect to MongoDB"""
    global client, db
    client = AsyncIOMotorClient(settings.MONGODB_URL)
    db = client[settings.DATABASE_NAME]
    print("Connected to MongoDB")

async def close_mongo_connection():
    """Close MongoDB connection"""
    global client
    if client:
        client.close()
        print("Closed MongoDB connection")

def get_database():
    """Get database instance"""
    global db
    return db

def get_collection(collection_name: str):
    """Get collection from database"""
    global db
    if db is None:
        raise Exception("Database not initialized")
    return db[collection_name]
