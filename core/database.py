from motor.motor_asyncio import AsyncIOMotorClient
from core.config import settings
from typing import Optional
from urllib.parse import urlparse, urlunparse

# Global MongoDB client
client: Optional[AsyncIOMotorClient] = None

# Database instance
db = None

async def connect_to_mongo():
    """Connect to MongoDB with a fast fail and a local fallback for dev."""
    global client, db
    uri = settings.MONGODB_URL
    # Faster selection timeout for dev ergonomics
    client = AsyncIOMotorClient(uri, serverSelectionTimeoutMS=5000)
    try:
        # Force connection attempt now to fail fast if unreachable
        await client.admin.command("ping")
        db = client[settings.DATABASE_NAME]
        print(f"Connected to MongoDB at {uri}")
        return
    except Exception as exc:
        # If using docker-compose hostname 'mongodb' locally, fall back to localhost
        try:
            parsed = urlparse(uri)
            if parsed.hostname == "mongodb":
                fallback = parsed._replace(netloc=f"localhost:{parsed.port or 27017}")
                fallback_uri = urlunparse(fallback)
                fallback_client = AsyncIOMotorClient(fallback_uri, serverSelectionTimeoutMS=4000)
                await fallback_client.admin.command("ping")
                client = fallback_client
                db = client[settings.DATABASE_NAME]
                print(f"Connected to MongoDB via fallback {fallback_uri}")
                return
        except Exception:
            pass
        # Surface an actionable message
        raise RuntimeError(
            "Failed to connect to MongoDB. Check MONGODB_URL in your environment. "
            "For local dev without Docker, set MONGODB_URL=mongodb://127.0.0.1:27017."
        ) from exc

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
