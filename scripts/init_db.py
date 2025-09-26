#!/usr/bin/env python3
"""
Database initialization script
"""
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from core.config import settings

async def init_database():
    """Initialize the database with required collections and indexes"""
    try:
        # Connect to MongoDB
        client = AsyncIOMotorClient(settings.MONGODB_URL)
        db = client[settings.DATABASE_NAME]
        
        # Create collections if they don't exist
        collections = await db.list_collection_names()
        
        required_collections = [
            "users",
            "agents",
            "strategies",
            "memory_logs",
            "calls",
            "subscriptions"
        ]
        
        for collection_name in required_collections:
            if collection_name not in collections:
                await db.create_collection(collection_name)
                print(f"Created collection: {collection_name}")
        
        # Create indexes
        # Users collection indexes
        await db.users.create_index("email", unique=True)
        await db.users.create_index("user_id", unique=True)
        print("Created indexes for users collection")
        
        # Agents collection indexes
        await db.agents.create_index("agent_id", unique=True)
        await db.agents.create_index("user_id")
        print("Created indexes for agents collection")
        
        # Strategies collection indexes
        await db.strategies.create_index("strategy_id", unique=True)
        await db.strategies.create_index("user_id")
        await db.strategies.create_index("agent_id")
        print("Created indexes for strategies collection")
        
        # Memory logs collection indexes
        await db.memory_logs.create_index("memory_id", unique=True)
        await db.memory_logs.create_index("user_id")
        await db.memory_logs.create_index("agent_id")
        await db.memory_logs.create_index("created_at")
        print("Created indexes for memory_logs collection")
        
        # Calls collection indexes
        await db.calls.create_index("call_id", unique=True)
        await db.calls.create_index("user_id")
        await db.calls.create_index("agent_id")
        await db.calls.create_index("created_at")
        print("Created indexes for calls collection")
        
        # Subscriptions collection indexes
        await db.subscriptions.create_index("user_id")
        print("Created indexes for subscriptions collection")
        
        print("Database initialization completed successfully!")
        
    except Exception as e:
        print(f"Error initializing database: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    asyncio.run(init_database())
