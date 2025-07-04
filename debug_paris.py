#!/usr/bin/env python3
"""
Quick debug script to inspect Paris (CDG) cache data
Run this to see what's in your MongoDB cache right now
"""

import os
from pymongo import MongoClient
import statistics

# Get MongoDB connection
mongodb_connection = os.getenv('MONGODB_CONNECTION_STRING')
client = MongoClient(mongodb_connection)
db = client['flight_bot_db']

def debug_paris_cache():
    print("🔍 DEBUGGING PARIS (CDG) CACHE DATA")
    print("=" * 50)
    
    # Get all Paris price data
    paris_data = list(db.flight_data.find(
        {'destination': 'CDG'}, 
        {'price': 1, 'cached_date': 1, 'verified_deal': 1, '_id': 0}
    ))
    
    if not paris_data:
        print("❌ No data found for CDG")
        return
    
    prices = [doc['price'] for doc in paris_data]
    
    print(f"📊 Total CDG entries: {len(paris_data)}")
    print(f"💰 Price range: {min(prices):.0f} - {max(prices):.0f} PLN")
    print(f"📈 Raw median: {statistics.median(prices):.0f} PLN")
    print(f"📊 Raw average: {sum(prices)/len(prices):.0f} PLN")
    
    # Check price distribution
    price_buckets = {
        "Under 300": len([p for p in prices if p < 300]),
        "300-500": len([p for p in prices if 300 <= p < 500]),
        "500-800": len([p for p in prices if 500 <= p < 800]), 
        "800-1200": len([p for p in prices if 800 <= p < 1200]),
        "Over 1200": len([p for p in prices if p >= 1200])
    }
    
    print("\n📊 Price Distribution:")
    for bucket, count in price_buckets.items():
        percentage = (count / len(prices)) * 100
        print(f"   {bucket}: {count} entries ({percentage:.1f}%)")
    
    # Check for extremely high prices
    high_prices = [p for p in prices if p > 1000]
    print(f"\n⚠️ Prices over 1000 PLN: {len(high_prices)} entries")
    if high_prices:
        print(f"   Highest prices: {sorted(high_prices, reverse=True)[:10]}")
    
    # Check verification status
    verified_count = len([doc for doc in paris_data if doc.get('verified_deal')])
    print(f"\n✅ Verified deals cached: {verified_count}")
    
    # Check dates
    dates = list(set([doc['cached_date'] for doc in paris_data]))
    print(f"📅 Cache dates: {sorted(dates)}")
    
    # Get current statistics
    stats = db.destination_stats.find_one({'destination': 'CDG'})
    if stats:
        print(f"\n📊 Current Statistics:")
        print(f"   Median: {stats['median_price']:.0f} PLN")
        print(f"   Std Dev: {stats['std_dev']:.0f} PLN")
        print(f"   Min: {stats['min_price']:.0f} PLN") 
        print(f"   Max: {stats['max_price']:.0f} PLN")
        print(f"   Sample size: {stats['sample_size']}")
        print(f"   Outliers removed: {stats.get('outliers_removed', 'N/A')}")

if __name__ == "__main__":
    debug_paris_cache()
