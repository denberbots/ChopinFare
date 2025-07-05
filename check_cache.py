#!/usr/bin/env python3
"""
Quick Cache Check - Verify European destination medians
"""
import os
import pymongo

def quick_cache_check():
    """Check key European destinations for realistic medians"""
    
    mongodb_uri = os.getenv('MONGODB_CONNECTION_STRING')
    client = pymongo.MongoClient(mongodb_uri)
    db = client['flight_deals']
    
    # Key European destinations to check
    european_destinations = ['BCN', 'CDG', 'FCO', 'AMS', 'LHR', 'MUC', 'VIE', 'BRU', 'MAD']
    
    print("🔍 CHECKING KEY EUROPEAN DESTINATIONS:")
    print("=" * 50)
    
    realistic_count = 0
    total_checked = 0
    
    for dest in european_destinations:
        stat = db.statistics.find_one({'destination': dest})
        total_checked += 1
        
        if stat:
            median = stat['median_price']
            samples = stat['sample_size']
            
            # Check if median is realistic (should be 300-800 PLN for Europe)
            is_realistic = 300 <= median <= 800
            status = "✅ REALISTIC" if is_realistic else "❌ INFLATED"
            
            if is_realistic:
                realistic_count += 1
                
            print(f"{dest}: {median:.0f} PLN (samples: {samples}) {status}")
        else:
            print(f"{dest}: No data ⚠️")
    
    print("=" * 50)
    print(f"📊 SUMMARY: {realistic_count}/{total_checked} destinations have realistic medians")
    
    if realistic_count >= 7:
        print("✅ Cache looks GOOD! Ready for deal detection.")
    else:
        print("❌ Cache still has issues. Consider cleaning again.")
    
    client.close()

if __name__ == "__main__":
    quick_cache_check()
