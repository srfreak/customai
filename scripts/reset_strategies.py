#!/usr/bin/env python3
"""Utility script to wipe all strategy documents or reassign to a specific user."""
import argparse
import asyncio
import sys
from datetime import datetime
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient

from core.config import settings

COLLECTION_STRATEGIES = "strategies"
COLLECTION_USERS = "users"


async def reset_strategies(user_email: Optional[str] = None) -> None:
    client = AsyncIOMotorClient(settings.MONGODB_URL)
    db = client[settings.DATABASE_NAME]
    try:
        if user_email:
            user = await db[COLLECTION_USERS].find_one({"email": user_email})
            if not user:
                print(f"No user found with email {user_email}")
                return
            user_id = user.get("user_id")
            if not user_id:
                print(f"User document for {user_email} lacks user_id field")
                return
            result = await db[COLLECTION_STRATEGIES].update_many(
                {},
                {
                    "$set": {
                        "user_id": user_id,
                        "updated_at": datetime.utcnow(),
                    }
                }
            )
            print(f"Updated user_id for {result.modified_count} strategy documents.")
        else:
            confirmation = input(
                "This will delete ALL strategies. Type 'DELETE' to continue: "
            )
            if confirmation != "DELETE":
                print("Aborted.")
                return
            result = await db[COLLECTION_STRATEGIES].delete_many({})
            print(f"Deleted {result.deleted_count} strategy documents.")
    finally:
        client.close()


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="Reset or reassign strategies.")
    parser.add_argument(
        "--user-email",
        help="If provided, assigns all strategies to the user with this email instead of deleting them.",
    )
    args = parser.parse_args(argv)

    asyncio.run(reset_strategies(user_email=args.user_email))


if __name__ == "__main__":
    main(sys.argv[1:])
