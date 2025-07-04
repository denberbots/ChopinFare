#!/usr/bin/env python3
"""
One-time Paris cleanup script
Run this once to clear corrupted Paris data
"""
import os
from pymongo import MongoClient

print("🧹 Starting Paris data cleanup...")

# Connect to your database  
MONGO_URI = os.getenv('MONGODB_CONNECTION_STRING')
if not MONGO_URI:
    print("❌ MONGODB_CONNECTION_STRING not found")
    exit(1)

client = MongoClient(MONGO_URI)
db = client['flight_bot_db']

# Delete all Paris flight data
result1 = db.flight_data.delete_many({'destination': 'CDG'})
print(f"✅ Deleted {result1.deleted_count} corrupted Paris flight entries")

# Delete Paris statistics
result2 = db.destination_stats.delete_one({'destination': 'CDG'})
if result2.deleted_count > 0:
    print("✅ Deleted corrupted Paris statistics")
else:
    print("ℹ️ No Paris statistics to delete")

print("🎯 Paris cleanup complete!")
print("🚀 Next bot run will rebuild clean Paris data")
