#!/usr/bin/env python3
"""
Combined Cache Management Script
- Check cache status
- Clean cache if needed
- Verify cleaning worked
"""

import os
import pymongo
from datetime import datetime

def check_cache_status():
    """Check if MongoDB cache is actually empty"""
    
    mongodb_uri = os.getenv('MONGODB_CONNECTION_STRING')
    if not mongodb_uri:
        print("❌ No MongoDB connection string found")
        return False, {}
    
    try:
        client = pymongo.MongoClient(mongodb_uri)
        db = client['flight_deals']
        
        print("🔍 CHECKING CACHE STATUS...")
        print("=" * 50)
        
        # Check collections
        stats_count = db.statistics.count_documents({})
        flights_count = db.flights.count_documents({})
        deals_count = db.deals.count_documents({})
        
        print(f"📊 Statistics collection: {stats_count} documents")
        print(f"✈️ Flights collection: {flights_count} documents") 
        print(f"🎯 Deals collection: {deals_count} documents")
        
        # Check if cache is truly empty
        is_empty = (stats_count == 0 and flights_count == 0 and deals_count == 0)
        
        if is_empty:
            print("\n✅ CACHE IS COMPLETELY EMPTY!")
            print("✅ Ready for fresh data collection")
        else:
            print(f"\n⚠️ CACHE IS NOT EMPTY!")
            
            if stats_count > 0:
                print(f"\n📊 Sample statistics found:")
                for stat in db.statistics.find().limit(5):
                    print(f"   {stat['destination']}: median={stat.get('median_price', 0):.0f} PLN, samples={stat.get('sample_size', 0)}")
            
            if flights_count > 0:
                print(f"\n✈️ Sample flights found:")
                for flight in db.flights.find().limit(3):
                    print(f"   {flight.get('origin', 'N/A')}→{flight.get('destination', 'N/A')}: {flight.get('value', 0):.0f} PLN")
        
        # Show latest cache update times
        if stats_count > 0:
            latest_stat = db.statistics.find().sort("updated_at", -1).limit(1)
            for stat in latest_stat:
                if 'updated_at' in stat:
                    print(f"\n🕒 Latest statistics update: {stat['updated_at']}")
        
        counts = {
            'statistics': stats_count,
            'flights': flights_count, 
            'deals': deals_count
        }
        
        client.close()
        return is_empty, counts
        
    except Exception as e:
        print(f"❌ Error checking cache: {e}")
        return False, {}

def clean_cache():
    """Clean all cache collections"""
    
    mongodb_uri = os.getenv('MONGODB_CONNECTION_STRING')
    if not mongodb_uri:
        print("❌ No MongoDB connection string found")
        return False
    
    try:
        client = pymongo.MongoClient(mongodb_uri)
        db = client['flight_deals']
        
        print("\n🧹 CLEANING CACHE...")
        print("=" * 50)
        
        # Delete all documents from each collection
        stats_result = db.statistics.delete_many({})
        flights_result = db.flights.delete_many({})
        deals_result = db.deals.delete_many({})
        
        print(f"🗑️ Deleted {stats_result.deleted_count} statistics")
        print(f"🗑️ Deleted {flights_result.deleted_count} flights")
        print(f"🗑️ Deleted {deals_result.deleted_count} deals")
        
        total_deleted = stats_result.deleted_count + flights_result.deleted_count + deals_result.deleted_count
        print(f"\n✅ Total documents deleted: {total_deleted}")
        
        client.close()
        return True
        
    except Exception as e:
        print(f"❌ Error cleaning cache: {e}")
        return False

def main():
    """Main function - check, clean if needed, verify"""
    
    print("🚀 MONGODB CACHE MANAGEMENT")
    print("=" * 60)
    
    # Step 1: Check current cache status
    is_empty, counts = check_cache_status()
    
    if is_empty:
        print("\n🎉 Cache is already empty - no cleaning needed!")
        return
    
    # Step 2: Ask user if they want to clean
    total_docs = sum(counts.values())
    print(f"\n📋 Found {total_docs} total documents in cache")
    
    response = input("\n❓ Do you want to clean the cache? (y/n): ").lower().strip()
    
    if response != 'y':
        print("⏭️ Skipping cache cleaning")
        return
    
    # Step 3: Clean the cache
    if clean_cache():
        print("\n🔄 Verifying cache was cleaned...")
        
        # Step 4: Verify cleaning worked
        is_empty_after, counts_after = check_cache_status()
        
        if is_empty_after:
            print("\n🎉 SUCCESS! Cache has been completely cleaned!")
            print("✅ Your bot will now collect fresh data on next run")
        else:
            print("\n⚠️ WARNING: Some data may still remain")
            print("📋 You may need to run the cleaning script again")
    else:
        print("\n❌ Cache cleaning failed!")

if __name__ == "__main__":
    main()
