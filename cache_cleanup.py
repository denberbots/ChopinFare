#!/usr/bin/env python3
"""
Complete cache wipe script
Deletes ALL flight data and statistics to start fresh
Prevents any corrupted data from causing false deals
"""
import os
from pymongo import MongoClient

print("🧹 Starting COMPLETE cache cleanup...")
print("⚠️ This will delete ALL cached flight data and statistics")

# Connect to your database  
MONGO_URI = os.getenv('MONGODB_CONNECTION_STRING')
if not MONGO_URI:
    print("❌ MONGODB_CONNECTION_STRING not found")
    exit(1)

client = MongoClient(MONGO_URI)
db = client['flight_bot_db']

print("\n🗑️ Deleting all collections...")

# Delete ALL flight data
result1 = db.flight_data.delete_many({})
print(f"✅ Deleted {result1.deleted_count} total flight entries")

# Delete ALL destination statistics  
result2 = db.destination_stats.delete_many({})
print(f"✅ Deleted {result2.deleted_count} destination statistics")

# Delete ALL deal alerts (optional - clears alert history)
result3 = db.deal_alerts.delete_many({})
print(f"✅ Deleted {result3.deleted_count} deal alert records")

# Show remaining collections (should be empty or minimal)
collections = db.list_collection_names()
print(f"\n📊 Remaining collections: {collections}")

for collection_name in collections:
    count = db[collection_name].count_documents({})
    print(f"  - {collection_name}: {count} documents")

print("\n🎯 COMPLETE cache wipe finished!")
print("🚀 Next bot run will rebuild ALL data from scratch")
print("✨ No more corrupted data - only fresh, clean prices!")
