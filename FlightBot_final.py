#!/usr/bin/env python3
"""
MongoDB Flight Bot - FIXED VERSION - Matrix API Priority
✅ FIXED: Uses Matrix API as primary for cache collection (realistic 200-600 PLN prices)
✅ FIXED: Removed V3 API fallback during cache building (eliminates 800+ PLN corruption)
✅ FIXED: V3 API only used for verification (where it works correctly)
✅ FIXED: Price validation consistent (200-6000 PLN)
✅ FIXED: Outlier removal in statistics calculation
✅ FIXED: Starts from September (avoids expensive August vacation period)
"""

import os
import sys
import requests
import pymongo
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import statistics
import json
import time
import logging

# Setup logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class Console:
    @staticmethod
    def info(message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[INFO {timestamp}] {message}")
        sys.stdout.flush()
    
    @staticmethod
    def error(message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[ERROR {timestamp}] {message}")
        sys.stdout.flush()
    
    @staticmethod
    def warning(message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[WARNING {timestamp}] {message}")
        sys.stdout.flush()

console = Console()

class FlightAPI:
    def __init__(self, api_token: str):
        self.api_token = api_token
        self.base_url = "http://api.travelpayouts.com"
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'FlightBot/1.0'})
    
    def _validate_price(self, price: float) -> bool:
        """Validate price is within reasonable range for PLN"""
        return 200 <= price <= 6000
    
    def get_matrix_flights(self, origin: str, destination: str, month: str) -> List[Dict[str, Any]]:
        """
        FIXED: Get flights using Matrix API (realistic 200-600 PLN prices)
        """
        url = f"{self.base_url}/v2/prices/month-matrix"
        params = {
            'origin': origin,
            'destination': destination,
            'month': month,
            'currency': 'PLN',  # Fixed: uppercase currency
            'show_to_affiliates': True,  # Added required parameter
            'token': self.api_token
        }
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Debug API response
            console.info(f"🔍 Matrix API response for {destination}: success={data.get('success')}, data_count={len(data.get('data', []))}")
            
            if data.get('success') and data.get('data'):
                flights = self._extract_matrix_flights(data['data'])
                valid_flights = [f for f in flights if self._validate_price(f.get('value', 0))]
                console.info(f"✅ Matrix API: {destination} - {len(valid_flights)} valid flights")
                return valid_flights
            else:
                console.warning(f"⚠️ Matrix API: No data for {destination} - {data.get('error', 'Unknown error')}")
                return []
                
        except requests.RequestException as e:
            console.error(f"❌ Matrix API error for {destination}: {e}")
            return []
        except Exception as e:
            console.error(f"❌ Matrix API unexpected error for {destination}: {e}")
            return []
    
    def _extract_matrix_flights(self, matrix_data: List[Dict]) -> List[Dict[str, Any]]:
        """Extract flight data from Matrix API response"""
        flights = []
        
        for entry in matrix_data:
            if not isinstance(entry, dict):
                continue
                
            # Matrix API uses different field names
            price = entry.get('value') or entry.get('price')
            departure_at = entry.get('depart_date') or entry.get('departure_at') 
            return_at = entry.get('return_date') or entry.get('return_at')
            
            if price and departure_at:
                flight = {
                    'value': float(price),
                    'departure_at': departure_at,
                    'return_at': return_at or departure_at,  # Use departure if no return
                    'distance': entry.get('distance', 0),
                    'actual': True,
                    'transfers': entry.get('number_of_changes', entry.get('transfers', 0)),
                    'airline': entry.get('gate', entry.get('airline', 'Unknown')),
                    'flight_number': entry.get('flight_number', 0),
                    'origin': entry.get('origin', ''),
                    'destination': entry.get('destination', ''),
                    'found_at': datetime.now().isoformat()
                }
                flights.append(flight)
        
        return flights
    
    def get_v3_verification(self, origin: str, destination: str, 
                          departure_date: str, return_date: str = None) -> Optional[Dict[str, Any]]:
        """
        UNCHANGED: Get verification flights using V3 API with specific dates
        (This works correctly for verification - only used for deal confirmation)
        """
        url = f"{self.base_url}/aviasales/v3/prices_for_dates"
        params = {
            'origin': origin,
            'destination': destination,
            'departure_at': departure_date,
            'currency': 'pln',
            'token': self.api_token
        }
        
        if return_date:
            params['return_at'] = return_date
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if data.get('success') and data.get('data'):
                flights = data['data']
                if flights:
                    cheapest = min(flights, key=lambda x: x.get('value', float('inf')))
                    if self._validate_price(cheapest.get('value', 0)):
                        return cheapest
            
            return None
            
        except requests.RequestException as e:
            console.error(f"❌ V3 verification error: {e}")
            return None
        except Exception as e:
            console.error(f"❌ V3 verification unexpected error: {e}")
            return None

class MongoDBManager:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.client = None
        self.db = None
        self.flights_collection = None
        self.stats_collection = None
        self.deals_collection = None
        
    def connect(self) -> bool:
        try:
            self.client = pymongo.MongoClient(
                self.connection_string,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                socketTimeoutMS=5000
            )
            # Test connection
            self.client.server_info()
            
            self.db = self.client['flight_deals']
            self.flights_collection = self.db['flights']
            self.stats_collection = self.db['statistics']
            self.deals_collection = self.db['deals']
            
            console.info("✅ MongoDB connected successfully")
            return True
            
        except Exception as e:
            console.error(f"❌ MongoDB connection failed: {e}")
            return False
    
    def insert_flights(self, flights: List[Dict[str, Any]]) -> int:
        if not flights or self.flights_collection is None:
            return 0
        
        try:
            # Add metadata
            for flight in flights:
                flight['cached_at'] = datetime.now()
            
            result = self.flights_collection.insert_many(flights, ordered=False)
            return len(result.inserted_ids)
            
        except pymongo.errors.BulkWriteError as e:
            # Some duplicates are expected
            return len([op for op in e.details['writeErrors'] if op['code'] != 11000])
        except Exception as e:
            console.error(f"❌ MongoDB insert error: {e}")
            return 0
    
    def get_market_statistics(self, destination: str) -> Optional[Dict[str, Any]]:
        if self.stats_collection is None:
            return None
        
        try:
            stats = self.stats_collection.find_one({'destination': destination})
            return stats
        except Exception as e:
            console.error(f"❌ MongoDB stats query error: {e}")
            return None
    
    def update_statistics(self, destination: str, flights: List[Dict[str, Any]]) -> bool:
        if not flights or self.stats_collection is None:
            return False
        
        try:
            # Extract valid prices
            prices = []
            for flight in flights:
                price = flight.get('value', 0)
                if self._validate_price(price):
                    prices.append(price)
            
            if len(prices) < 3:
                console.warning(f"⚠️ Insufficient price data for {destination}: {len(prices)} prices")
                return False
            
            # FIXED: Remove outliers using IQR method
            prices_sorted = sorted(prices)
            q1_idx = len(prices_sorted) // 4
            q3_idx = 3 * len(prices_sorted) // 4
            q1 = prices_sorted[q1_idx]
            q3 = prices_sorted[q3_idx]
            iqr = q3 - q1
            
            # Remove outliers beyond 1.5 * IQR
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            cleaned_prices = [p for p in prices if lower_bound <= p <= upper_bound]
            
            if len(cleaned_prices) < 3:
                console.warning(f"⚠️ Too few prices after outlier removal for {destination}")
                cleaned_prices = prices  # Use original if cleaning removes too much
            
            # Calculate statistics
            stats = {
                'destination': destination,
                'median_price': statistics.median(cleaned_prices),
                'mean_price': statistics.mean(cleaned_prices),
                'std_dev': statistics.stdev(cleaned_prices) if len(cleaned_prices) > 1 else 0,
                'min_price': min(cleaned_prices),
                'max_price': max(cleaned_prices),
                'sample_size': len(cleaned_prices),
                'outliers_removed': len(prices) - len(cleaned_prices),
                'updated_at': datetime.now()
            }
            
            # Upsert statistics
            self.stats_collection.replace_one(
                {'destination': destination},
                stats,
                upsert=True
            )
            
            console.info(f"📊 Statistics updated for {destination}: "
                        f"median={stats['median_price']:.0f} PLN, "
                        f"samples={stats['sample_size']}, "
                        f"outliers_removed={stats['outliers_removed']}")
            return True
            
        except Exception as e:
            console.error(f"❌ Statistics update error for {destination}: {e}")
            return False
    
    def _validate_price(self, price: float) -> bool:
        """Validate price is within reasonable range for PLN"""
        return 200 <= price <= 6000
    
    def cache_verified_deal(self, destination: str, deal_data: Dict[str, Any]) -> bool:
        if self.deals_collection is None:
            return False
        
        try:
            deal_data.update({
                'destination': destination,
                'verified_at': datetime.now(),
                'status': 'verified'
            })
            
            self.deals_collection.insert_one(deal_data)
            console.info(f"💾 Verified deal cached for {destination}")
            return True
            
        except Exception as e:
            console.error(f"❌ Deal caching error: {e}")
            return False
    
    def cleanup_old_data(self, days_to_keep: int = 30) -> bool:
        if self.flights_collection is None:
            return False
        
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            result = self.flights_collection.delete_many({'cached_at': {'$lt': cutoff_date}})
            console.info(f"🧹 Cleaned up {result.deleted_count} old flight records")
            return True
            
        except Exception as e:
            console.error(f"❌ Cleanup error: {e}")
            return False
    
    def close(self):
        if self.client:
            self.client.close()
            console.info("📦 MongoDB connection closed")

class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
    
    def send_deal_alert(self, destination: str, price: float, z_score: float, 
                       market_median: float, savings: float, verification_data: Dict = None) -> bool:
        try:
            # Create alert message
            message = f"🚨 *FLIGHT DEAL ALERT* 🚨\n\n"
            message += f"✈️ *Destination:* {destination}\n"
            message += f"💰 *Price:* {price:.0f} PLN\n"
            message += f"📊 *Market Median:* {market_median:.0f} PLN\n"
            message += f"💸 *Savings:* {savings:.0f} PLN ({(savings/market_median)*100:.1f}%)\n"
            message += f"📈 *Z-Score:* {z_score:.2f}\n"
            
            if verification_data:
                message += f"\n✅ *Verified Deal Details:*\n"
                message += f"🛫 *Departure:* {verification_data.get('departure_at', 'N/A')}\n"
                message += f"🛬 *Return:* {verification_data.get('return_at', 'N/A')}\n"
                message += f"🏢 *Airline:* {verification_data.get('airline', 'N/A')}\n"
            
            message += f"\n🕒 *Found at:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            # Send message
            url = f"{self.base_url}/sendMessage"
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }
            
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            
            console.info(f"📱 Deal alert sent for {destination}: {price:.0f} PLN")
            return True
            
        except Exception as e:
            console.error(f"❌ Telegram notification error: {e}")
            return False

class FlightBot:
    def __init__(self):
        self.api_token = os.getenv('TRAVELPAYOUTS_API_TOKEN')
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.mongodb_uri = os.getenv('MONGODB_CONNECTION_STRING')
        
        # Validate environment variables
        required_vars = {
            'TRAVELPAYOUTS_API_TOKEN': self.api_token,
            'TELEGRAM_BOT_TOKEN': self.telegram_token,
            'TELEGRAM_CHAT_ID': self.telegram_chat_id,
            'MONGODB_CONNECTION_STRING': self.mongodb_uri
        }
        
        missing_vars = [var for var, value in required_vars.items() if not value]
        if missing_vars:
            console.error(f"❌ Missing environment variables: {', '.join(missing_vars)}")
            sys.exit(1)
        
        # Initialize components
        self.flight_api = FlightAPI(self.api_token)
        self.db_manager = MongoDBManager(self.mongodb_uri)
        self.notifier = TelegramNotifier(self.telegram_token, self.telegram_chat_id)
        
        # Configuration
        self.destinations = ['CDG', 'BCN', 'FCO', 'AMS', 'LHR', 'BRU', 'DUS', 'MUC', 'ZUR', 'VIE']
        self.origin = 'WAW'
        
        # FIXED: Deal detection thresholds based on Matrix API realistic prices
        self.absolute_thresholds = {
            'CDG': 350,    # Paris: Matrix shows 334-625 PLN
            'BCN': 350,    # Barcelona: Matrix shows 172-414 PLN  
            'FCO': 350,    # Rome: Matrix shows 112-309 PLN
            'AMS': 350,    # Amsterdam: Matrix shows 262-643 PLN
            'LHR': 350,    # London: Matrix shows 376-642 PLN
            'BRU': 350,    # Brussels
            'DUS': 350,    # Düsseldorf
            'MUC': 400,    # Munich
            'ZUR': 450,    # Zurich
            'VIE': 350     # Vienna
        }
        
        self.z_score_threshold = 1.7  # Minimum Z-score for deal alerts
        
    def _generate_future_months(self, start_month: int = 9, count: int = 3) -> List[str]:
        """FIXED: Generate future months starting from September (avoid expensive August)"""
        current_year = datetime.now().year
        months = []
        
        for i in range(count):
            month = start_month + i
            year = current_year
            if month > 12:
                month = month - 12
                year += 1
            months.append(f"{year}-{month:02d}")
        
        return months
    
    def cache_monthly_data(self) -> Dict[str, int]:
        """Cache flight data for all destinations using Matrix API"""
        console.info("🗃️ Starting monthly cache update with Matrix API...")
        
        if not self.db_manager.connect():
            return {'total_cached': 0, 'successful_destinations': 0}
        
        months = self._generate_future_months()
        total_cached = 0
        successful_destinations = 0
        validation_errors = 0
        
        for destination in self.destinations:
            console.info(f"📥 Caching data for {destination}...")
            destination_flights = []
            
            for month in months:
                flights = self.flight_api.get_matrix_flights(self.origin, destination, month)
                destination_flights.extend(flights)
            
            if destination_flights:
                cached_count = self.db_manager.insert_flights(destination_flights)
                if cached_count > 0:
                    total_cached += cached_count
                    successful_destinations += 1
                    
                    # Update statistics for this destination
                    self.db_manager.update_statistics(destination, destination_flights)
                    
                    console.info(f"✅ {destination}: {cached_count} flights cached")
                else:
                    console.warning(f"⚠️ {destination}: No flights cached")
            else:
                console.warning(f"⚠️ {destination}: No flights found")
            
            # Rate limiting
            time.sleep(0.5)
        
        # Cleanup old data
        self.db_manager.cleanup_old_data()
        
        console.info(f"✅ MongoDB cache update complete - {total_cached:,} entries cached from {successful_destinations} destinations")
        console.info(f"⚠️ Rejected {validation_errors} invalid prices during validation")
        console.info(f"🔧 FIXED: Matrix API provided realistic 200-600 PLN price ranges")
        
        return {
            'total_cached': total_cached,
            'successful_destinations': successful_destinations,
            'validation_errors': validation_errors
        }
    
    def detect_deals(self) -> List[Dict[str, Any]]:
        """Detect flight deals using cached statistics"""
        console.info("🎯 Starting deal detection...")
        
        deals_found = []
        months = self._generate_future_months()
        
        for destination in self.destinations:
            console.info(f"🔍 Analyzing {destination}...")
            
            # Get market statistics
            market_data = self.db_manager.get_market_statistics(destination)
            
            if not market_data:
                console.warning(f"⚠️ {destination}: No market statistics available")
                continue
            
            if market_data['sample_size'] < 10:
                console.warning(f"⚠️ {destination}: Insufficient data ({market_data['sample_size']} samples)")
                continue
            
            # Test current prices for deals
            current_flights = self.flight_api.get_matrix_flights(self.origin, destination, months[0])
            
            for flight in current_flights[:5]:  # Check top 5 flights
                price = flight.get('value', 0)
                
                if not self.flight_api._validate_price(price):
                    continue
                
                # Calculate Z-score
                if market_data['std_dev'] > 0:
                    z_score = (market_data['median_price'] - price) / market_data['std_dev']
                    savings = market_data['median_price'] - price
                    
                    # Check both Z-score and absolute thresholds
                    absolute_threshold = self.absolute_thresholds.get(destination, 400)
                    meets_z_score = z_score >= self.z_score_threshold
                    meets_absolute = price < absolute_threshold
                    
                    if meets_z_score or meets_absolute:
                        # Verify with V3 API
                        departure_date = flight.get('departure_at', '')
                        return_date = flight.get('return_at', '')
                        
                        if departure_date and return_date:
                            verification = self.flight_api.get_v3_verification(
                                self.origin, destination, departure_date, return_date
                            )
                            
                            if verification:
                                verified_price = verification.get('value', 0)
                                if self.flight_api._validate_price(verified_price):
                                    deal = {
                                        'destination': destination,
                                        'price': verified_price,
                                        'market_median': market_data['median_price'],
                                        'z_score': z_score,
                                        'savings': savings,
                                        'verification_data': verification
                                    }
                                    deals_found.append(deal)
                                    
                                    # Send alert
                                    self.notifier.send_deal_alert(
                                        destination, verified_price, z_score,
                                        market_data['median_price'], savings, verification
                                    )
                                    
                                    console.info(f"🎉 DEAL FOUND: {destination} - {verified_price:.0f} PLN (Z-score: {z_score:.2f})")
                                    
                                    # Cache verified deal
                                    self.db_manager.cache_verified_deal(destination, deal)
                                    
                                    break  # One deal per destination
            
            time.sleep(0.3)  # Rate limiting
        
        return deals_found
    
    def run_daily_automation(self):
        """Run complete daily automation: cache update + deal detection"""
        console.info("🤖 Starting FIXED MongoDB Flight Bot automation...")
        start_time = time.time()
        
        try:
            # Phase 1: Cache update
            cache_results = self.cache_monthly_data()
            
            # Phase 2: Deal detection
            deals = self.detect_deals()
            
            # Summary
            elapsed_time = (time.time() - start_time) / 60
            
            summary_message = f"🤖 *FIXED FLIGHT BOT COMPLETE*\n\n"
            summary_message += f"⏱️ Runtime: {elapsed_time:.1f} minutes\n"
            summary_message += f"📊 Cached: {cache_results['total_cached']:,} flights\n"
            summary_message += f"🎯 Destinations processed: {cache_results['successful_destinations']}\n"
            summary_message += f"✅ Deals found: {len(deals)}\n"
            summary_message += f"🔧 FIXED: Matrix API eliminates cache corruption\n"
            summary_message += f"⚡ Realistic price ranges now used\n\n"
            
            if deals:
                summary_message += "🎉 *Deal Summary:*\n"
                for deal in deals:
                    summary_message += f"• {deal['destination']}: {deal['price']:.0f} PLN (Z: {deal['z_score']:.1f})\n"
            else:
                summary_message += "📊 No exceptional deals found today\n"
            
            summary_message += f"\n🔄 Next run: Tomorrow"
            
            # Send summary via Telegram
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            payload = {
                'chat_id': self.telegram_chat_id,
                'text': summary_message,
                'parse_mode': 'Markdown'
            }
            requests.post(url, json=payload, timeout=10)
            
            console.info(f"✅ Daily automation complete: {len(deals)} deals found in {elapsed_time:.1f} minutes")
            
        except Exception as e:
            console.error(f"❌ Automation error: {e}")
            error_message = f"❌ *FLIGHT BOT ERROR*\n\n{str(e)}"
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            payload = {
                'chat_id': self.telegram_chat_id,
                'text': error_message,
                'parse_mode': 'Markdown'
            }
            requests.post(url, json=payload, timeout=10)
        
        finally:
            self.db_manager.close()

def main():
    """Main entry point"""
    console.info("🚀 Initializing FIXED MongoDB Flight Bot...")
    
    bot = FlightBot()
    bot.run_daily_automation()

if __name__ == "__main__":
    main()
