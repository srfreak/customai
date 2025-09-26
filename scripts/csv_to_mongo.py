#!/usr/bin/env python3
"""
CSV to MongoDB import script for Scriza AI Platform
Reads readme.csv and imports data into MongoDB
"""
import asyncio
import pandas as pd
from motor.motor_asyncio import AsyncIOMotorClient
from core.config import settings
from datetime import datetime
import uuid

async def import_csv_to_mongodb():
    """Import CSV data to MongoDB"""
    try:
        # Connect to MongoDB
        client = AsyncIOMotorClient(settings.MONGODB_URL)
        db = client[settings.DATABASE_NAME]
        
        # Read CSV file
        df = pd.read_csv('readme.csv')
        print(f"Loaded {len(df)} records from readme.csv")
        
        # Create contacts collection if it doesn't exist
        if 'contacts' not in await db.list_collection_names():
            await db.create_collection('contacts')
        
        # Clear existing data
        await db.contacts.delete_many({})
        
        # Import data
        contacts = []
        for _, row in df.iterrows():
            contact = {
                'contact_id': str(uuid.uuid4()),
                'id': str(row['ID']),
                'name': str(row['Name']),
                'address': str(row['Address']),
                'city': str(row['City']),
                'state': str(row['State']),
                'zip': str(row['Zip']),
                'mobile': str(row['Mobile']),
                'created_at': datetime.utcnow(),
                'updated_at': datetime.utcnow()
            }
            contacts.append(contact)
        
        # Insert all contacts
        if contacts:
            result = await db.contacts.insert_many(contacts)
            print(f"Successfully imported {len(result.inserted_ids)} contacts")
        
        # Create indexes
        await db.contacts.create_index('contact_id', unique=True)
        await db.contacts.create_index('id')
        await db.contacts.create_index('mobile')
        
        print("CSV import completed successfully!")
        
    except Exception as e:
        print(f"Error importing CSV: {e}")
    finally:
        client.close()

async def dump_contacts_to_csv():
    """Export contacts from MongoDB to CSV"""
    try:
        # Connect to MongoDB
        client = AsyncIOMotorClient(settings.MONGODB_URL)
        db = client[settings.DATABASE_NAME]
        
        # Get all contacts
        cursor = db.contacts.find({})
        contacts = await cursor.to_list(length=None)
        
        if not contacts:
            print("No contacts found in database")
            return
        
        # Convert to DataFrame
        data = []
        for contact in contacts:
            data.append({
                'ID': contact.get('id', ''),
                'Name': contact.get('name', ''),
                'Address': contact.get('address', ''),
                'City': contact.get('city', ''),
                'State': contact.get('state', ''),
                'Zip': contact.get('zip', ''),
                'Mobile': contact.get('mobile', '')
            })
        
        df = pd.DataFrame(data)
        
        # Save to CSV
        df.to_csv('dump.csv', index=False)
        print(f"Successfully exported {len(df)} contacts to dump.csv")
        
    except Exception as e:
        print(f"Error dumping contacts: {e}")
    finally:
        client.close()

async def main():
    """Main function to import and dump"""
    await import_csv_to_mongodb()
    await dump_contacts_to_csv()

if __name__ == "__main__":
    asyncio.run(main())
