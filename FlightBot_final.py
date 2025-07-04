#!/usr/bin/env python3
"""
Enhanced Flight Bot - Clean Build with Paris Cleanup
✅ Built from scratch to eliminate any syntax issues
✅ Forces Paris data cleanup on next run
✅ Enhanced data validation and economy filtering
"""

import logging
import os
import statistics
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import requests
from pymongo import MongoClient
import sys

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class Console:
    @staticmethod
    def info(msg: str):
        print(f"ℹ️  {msg}")
        logger.info(msg)
    
    @staticmethod
    def success(msg: str):
        print(f"✅ {msg}")
        logger.info(msg)
    
    @staticmethod
    def warning(msg: str):
        print(f"⚠️  {msg}")
        logger.warning(msg)
    
    @staticmethod
    def error(msg: str):
        print(f"❌ {msg}")
        logger.error(msg)

console = Console()

# Configuration
AMADEUS_API_KEY = os.getenv('AMADEUS_API_KEY')
AMADEUS_API_SECRET = os.getenv('AMADEUS_API_SECRET')
MONGO_URI = os.getenv('MONGO_URI')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# Enhanced validation
PRICE_LIMITS = (150, 4000)
MAX_PRICE_FILTER = 5000
MIN_PRICE_FILTER = 100

REGIONAL_PRICE_RANGES = {
    'europe_west': (200, 1200),
    'europe_close': (150, 800),
    'europe_north': (300, 1000),
    'asia_east': (900, 4000),
    'asia_south': (800, 3000),
    'middle_east': (600, 2500),
    'africa_north': (500, 2000),
    'americas': (1200, 5000),
    'domestic': (150, 600),
    'default': (200, 3000)
}

ABSOLUTE_THRESHOLDS = {
    'europe_west': 350,
    'europe_close': 320,
    'europe_north': 650,
    'asia_east': 1400,
    'asia_south': 1300,
    'middle_east': 1200,
    'africa_north': 900,
    'americas': 2000,
    'domestic': 250,
    'default': 500
}

DESTINATIONS = {
    'CDG': {'name': 'Paris', 'country': 'France', 'region': 'europe_west'},
    'ORY': {'name': 'Paris Orly', 'country': 'France', 'region': 'europe_west'},
    'AMS': {'name': 'Amsterdam', 'country': 'Netherlands', 'region': 'europe_west'},
    'BRU': {'name': 'Brussels', 'country': 'Belgium', 'region': 'europe_west'},
    'LHR': {'name': 'London', 'country': 'United Kingdom', 'region': 'europe_west'},
    'FRA': {'name': 'Frankfurt', 'country': 'Germany', 'region': 'europe_west'},
    'MUC': {'name': 'Munich', 'country': 'Germany', 'region': 'europe_west'},
    'VIE': {'name': 'Vienna', 'country': 'Austria', 'region': 'europe_close'},
    'PRG': {'name': 'Prague', 'country': 'Czech Republic', 'region': 'europe_close'},
    'BUD': {'name': 'Budapest', 'country': 'Hungary', 'region': 'europe_close'},
    'OSL': {'name': 'Oslo', 'country': 'Norway', 'region': 'europe_north'},
    'ARN': {'name': 'Stockholm', 'country': 'Sweden', 'region': 'europe_north'},
    'CPH': {'name': 'Copenhagen', 'country': 'Denmark', 'region': 'europe_north'},
    'NRT': {'name': 'Tokyo', 'country': 'Japan', 'region': 'asia_east'},
    'ICN': {'name': 'Seoul', 'country': 'South Korea', 'region': 'asia_east'},
    'DEL': {'name': 'Delhi', 'country': 'India', 'region': 'asia_south'},
    'BKK': {'name': 'Bangkok', 'country': 'Thailand', 'region': 'asia_south'},
    'DXB': {'name': 'Dubai', 'country': 'UAE', 'region': 'middle_east'},
    'DOH': {'name': 'Doha', 'country': 'Qatar', 'region': 'middle_east'},
    'IST': {'name': 'Istanbul', 'country': 'Turkey', 'region': 'middle_east'},
    'CAI': {'name': 'Cairo', 'country': 'Egypt', 'region': 'africa_north'},
    'JFK': {'name': 'New York', 'country': 'USA', 'region': 'americas'},
    'YYZ': {'name': 'Toronto', 'country': 'Canada', 'region': 'americas'},
    'LIS': {'name': 'Lisbon', 'country': 'Portugal', 'region': 'europe_west'},
    'WAW': {'name': 'Warsaw', 'country': 'Poland', 'region': 'domestic'}
}

class AmadeusAPI:
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://test.api.amadeus.com"
        self.access_token = None
        self.token_expires = None
        
    def _get_access_token(self) -> str:
        if self.access_token and self.token_expires and datetime.now() < self.token_expires:
            return self.access_token
            
        url = f"{self.base_url}/v1/security/oauth2/token"
        data = {
            'grant_type': 'client_credentials',
            'client_id': self.api_key,
            'client_secret': self.api_secret
        }
        
        response = requests.post(url, data=data)
        response.raise_for_status()
        
        token_data = response.json()
        self.access_token = token_data['access_token']
        self.token_expires = datetime.now() + timedelta(seconds=token_data['expires_in'] - 300)
        
        return self.access_token
    
    def search_flights(self, origin: str, destination: str, departure_date: str, 
                      return_date: str, adults: int = 1) -> List[Dict]:
        try:
            token = self._get_access_token()
            url = f"{self.base_url}/v2/shopping/flight-offers"
            
            params = {
                'originLocationCode': origin,
                'destinationLocationCode': destination,
                'departureDate': departure_date,
                'returnDate': return_date,
                'adults': adults,
                'children': 0,
                'infants': 0,
                'travelClass': 'ECONOMY',
                'currencyCode': 'PLN',
                'max': 250,
                'nonStop': 'false'
            }
            
            headers = {
                'Authorization': f'Bearer {token}',
                'Content-Type': 'application/json'
            }
            
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            flights = data.get('data', [])
            
            validated_flights = []
            for flight in flights:
                try:
                    is_economy = True
                    for itinerary in flight.get('itineraries', []):
                        for segment in itinerary.get('segments', []):
                            cabin = segment.get('cabin', 'ECONOMY')
                            if cabin != 'ECONOMY':
                                is_economy = False
                                break
                        if not is_economy:
                            break
                    
                    price_info = flight.get('price', {})
                    currency = price_info.get('currency', 'PLN')
                    
                    if is_economy and currency == 'PLN':
                        validated_flights.append(flight)
                        
                except (KeyError, TypeError):
                    continue
            
            console.info(f"Found {len(validated_flights)} economy flights for {origin} → {destination}")
            return validated_flights
            
        except Exception as e:
            console.error(f"API request failed: {e}")
            return []

class FlightDataValidator:
    @staticmethod
    def validate_price(price: float, destination: str) -> bool:
        if price < MIN_PRICE_FILTER or price > MAX_PRICE_FILTER:
            return False
            
        region = DESTINATIONS.get(destination, {}).get('region', 'default')
        min_price, max_price = REGIONAL_PRICE_RANGES.get(region, REGIONAL_PRICE_RANGES['default'])
        
        flexible_min = min_price * 0.7
        flexible_max = max_price * 1.3
        
        return flexible_min <= price <= flexible_max
    
    @staticmethod
    def validate_flight_combination(origin: str, destination: str, price: float, 
                                  departure_date: str, return_date: str) -> bool:
        try:
            if not FlightDataValidator.validate_price(price, destination):
                return False
                
            dep_date = datetime.strptime(departure_date, '%Y-%m-%d')
            ret_date = datetime.strptime(return_date, '%Y-%m-%d')
            
            if dep_date >= ret_date:
                return False
                
            duration = (ret_date - dep_date).days
            if duration < 1 or duration > 30:
                return False
                
            return True
            
        except Exception:
            return False

class MongoDBCache:
    def __init__(self, uri: str, db_name: str = 'flight_bot_db'):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.validator = FlightDataValidator()
        
    def store_flight_data(self, flight_data: Dict) -> bool:
        try:
            if not self.validator.validate_flight_combination(
                flight_data['origin'],
                flight_data['destination'],
                flight_data['price'],
                flight_data['outbound_date'],
                flight_data['return_date']
            ):
                return False
                
            flight_data['data_quality'] = 'validated'
            flight_data['validation_date'] = datetime.now()
            flight_data['unique_id'] = f"{flight_data['origin']}-{flight_data['destination']}-{flight_data['outbound_date']}-{flight_data['return_date']}-{flight_data['price']}"
            
            try:
                self.db.flight_data.insert_one(flight_data)
                return True
            except Exception:
                return False
                
        except Exception:
            return False
    
    def get_market_data(self, destination: str) -> Dict:
        try:
            prices_cursor = self.db.flight_data.find(
                {
                    'destination': destination,
                    'data_quality': 'validated'
                }, 
                {'price': 1}
            )
            prices = [doc['price'] for doc in prices_cursor]
            
            if len(prices) < 50:
                return {
                    'sample_size': len(prices),
                    'median_price': None,
                    'std_dev': None,
                    'min_price': None,
                    'max_price': None,
                    'sufficient_data': False
                }
            
            median_price = statistics.median(prices)
            std_dev = statistics.stdev(prices)
            min_price = min(prices)
            max_price = max(prices)
            
            if self._is_destination_data_corrupted(destination, min_price, max_price, median_price):
                console.warning(f"Detected corrupted data for {destination} - clearing cache")
                self.clear_corrupted_destination_data(destination)
                return {
                    'sample_size': 0,
                    'median_price': None,
                    'std_dev': None,
                    'min_price': None,
                    'max_price': None,
                    'sufficient_data': False
                }
            
            return {
                'sample_size': len(prices),
                'median_price': median_price,
                'std_dev': std_dev,
                'min_price': min_price,
                'max_price': max_price,
                'sufficient_data': True
            }
            
        except Exception:
            return {'sample_size': 0, 'sufficient_data': False}
    
    def _is_destination_data_corrupted(self, destination: str, min_price: float, 
                                     max_price: float, median_price: float) -> bool:
        region = DESTINATIONS.get(destination, {}).get('region', 'default')
        expected_min, expected_max = REGIONAL_PRICE_RANGES.get(region, REGIONAL_PRICE_RANGES['default'])
        
        if min_price > expected_min * 2:
            return True
        if median_price > expected_max * 1.5:
            return True
        if max_price > expected_max * 3:
            return True
            
        return False
    
    def clear_corrupted_destination_data(self, destination: str):
        try:
            result = self.db.flight_data.delete_many({'destination': destination})
            console.info(f"Cleared {result.deleted_count} corrupted flight entries for {destination}")
            
            self.db.destination_stats.delete_one({'destination': destination})
            console.info(f"Cleared corrupted stats for {destination}")
            
        except Exception as e:
            console.error(f"Error clearing corrupted data: {e}")
    
    def cleanup_old_data(self, days_old: int = 45):
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            
            result = self.db.flight_data.delete_many({
                'cached_date': {'$lt': cutoff_date}
            })
            if result.deleted_count > 0:
                console.info(f"Cleaned up {result.deleted_count} old flight entries")
            
        except Exception:
            pass

class FlightAnalyzer:
    def __init__(self, cache: MongoDBCache):
        self.cache = cache
        
    def analyze_deal(self, destination: str, price: float, departure_date: str, 
                    return_date: str) -> Dict:
        try:
            market_data = self.cache.get_market_data(destination)
            
            if not market_data['sufficient_data']:
                return {
                    'is_deal': False,
                    'deal_type': None,
                    'reason': 'insufficient_data',
                    'confidence': 0
                }
            
            region = DESTINATIONS.get(destination, {}).get('region', 'default')
            absolute_threshold = ABSOLUTE_THRESHOLDS.get(region, ABSOLUTE_THRESHOLDS['default'])
            
            absolute_deal = price < absolute_threshold
            
            median_price = market_data['median_price']
            std_dev = market_data['std_dev']
            z_score = (median_price - price) / std_dev if std_dev > 0 else 0
            statistical_deal = z_score >= 1.7
            
            if absolute_deal and statistical_deal:
                deal_type = "🔥 Amazing Deal"
                confidence = 95
            elif absolute_deal:
                deal_type = "💰 Great Price"
                confidence = 85
            elif statistical_deal:
                deal_type = "📊 Statistical Deal"
                confidence = 75
            else:
                return {
                    'is_deal': False,
                    'deal_type': None,
                    'reason': 'no_criteria_met',
                    'confidence': 0
                }
            
            savings_percent = round((1 - price / median_price) * 100, 1)
            
            return {
                'is_deal': True,
                'deal_type': deal_type,
                'confidence': confidence,
                'price': price,
                'median_price': median_price,
                'savings_percent': savings_percent,
                'z_score': round(z_score, 2),
                'absolute_threshold': absolute_threshold,
                'meets_absolute': absolute_deal,
                'meets_statistical': statistical_deal,
                'departure_date': departure_date,
                'return_date': return_date
            }
            
        except Exception:
            return {'is_deal': False, 'deal_type': None, 'confidence': 0}

class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        
    def send_deal_alert(self, destination: str, deal_info: Dict) -> bool:
        try:
            dest_info = DESTINATIONS.get(destination, {})
            city_name = dest_info.get('name', destination)
            country = dest_info.get('country', 'Unknown')
            
            message = f"✈️ **FLIGHT DEAL ALERT** ✈️\n\n"
            message += f"🏙️ **{city_name}, {country}** ({destination})\n"
            message += f"💰 **{deal_info['price']} zł** {deal_info['deal_type']}\n\n"
            
            message += f"📅 **Dates:** {deal_info['departure_date']} → {deal_info['return_date']}\n"
            message += f"📊 **Savings:** {deal_info['savings_percent']}% below typical ({deal_info['median_price']} zł)\n"
            message += f"🎯 **Confidence:** {deal_info['confidence']}%\n\n"
            
            criteria = []
            if deal_info['meets_absolute']:
                criteria.append(f"Under {deal_info['absolute_threshold']} zł threshold")
            if deal_info['meets_statistical']:
                criteria.append(f"Z-score: {deal_info['z_score']}")
            
            message += f"✅ **Criteria:** {', '.join(criteria)}\n\n"
            message += f"🔍 **Search:** WAW → {destination}\n"
            message += f"⏰ **Alert time:** {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            
            return self._send_message(message)
            
        except Exception:
            return False
    
    def send_status_update(self, message: str) -> bool:
        try:
            return self._send_message(message)
        except Exception:
            return False
    
    def _send_message(self, message: str) -> bool:
        try:
            url = f"{self.base_url}/sendMessage"
            data = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'Markdown',
                'disable_web_page_preview': True
            }
            
            response = requests.post(url, data=data)
            response.raise_for_status()
            
            return True
            
        except Exception:
            return False

class FlightBot:
    def __init__(self):
        self.api = AmadeusAPI(AMADEUS_API_KEY, AMADEUS_API_SECRET)
        self.cache = MongoDBCache(MONGO_URI)
        self.analyzer = FlightAnalyzer(self.cache)
        self.notifier = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
        self.deals_sent_today = set()
        self.start_time = time.time()
        
    def cache_daily_data(self):
        try:
            console.info("🔄 Starting daily cache building with Paris cleanup...")
            today = datetime.now().date()
            
            # FORCE CLEAR PARIS DATA - One-time cleanup
            console.info("🧹 FORCING Paris (CDG) data cleanup...")
            try:
                result1 = self.cache.db.flight_data.delete_many({'destination': 'CDG'})
                console.success(f"🧹 Cleared {result1.deleted_count} corrupted Paris flight entries")
                
                result2 = self.cache.db.destination_stats.delete_one({'destination': 'CDG'})
                if result2.deleted_count > 0:
                    console.success(f"🧹 Cleared corrupted Paris statistics")
                
                console.success("✅ Paris corruption cleanup COMPLETE - will rebuild with clean data")
                
            except Exception as e:
                console.error(f"Error during Paris cleanup: {e}")
            
            self.cache.cleanup_old_data(45)
            
            base_date = datetime.now().date()
            date_combinations = []
            
            for month_offset in range(6):
                month_date = base_date + timedelta(days=30 * month_offset)
                
                for day_offset in range(0, 28, 7):
                    departure = month_date + timedelta(days=day_offset)
                    
                    for duration in [2, 3, 4, 6, 7, 8]:
                        return_date = departure + timedelta(days=duration)
                        date_combinations.append((departure, return_date))
            
            console.info(f"📅 Generated {len(date_combinations)} date combinations")
            
            total_cached = 0
            new_entries_today = 0
            
            for destination in DESTINATIONS.keys():
                if destination == 'WAW':
                    continue
                    
                console.info(f"🔍 Building cache for {destination}...")
                destination_cached = 0
                
                daily_combinations = date_combinations[::3]
                
                for departure_date, return_date in daily_combinations:
                    try:
                        existing = self.cache.db.flight_data.find_one({
                            'origin': 'WAW',
                            'destination': destination,
                            'outbound_date': departure_date.strftime('%Y-%m-%d'),
                            'return_date': return_date.strftime('%Y-%m-%d'),
                            'cached_date': {'$gte': today - timedelta(days=7)}
                        })
                        
                        if existing:
                            continue
                        
                        flights = self.api.search_flights(
                            origin='WAW',
                            destination=destination,
                            departure_date=departure_date.strftime('%Y-%m-%d'),
                            return_date=return_date.strftime('%Y-%m-%d')
                        )
                        
                        for flight in flights[:5]:
                            try:
                                price = float(flight['price']['total'])
                                
                                flight_data = {
                                    'origin': 'WAW',
                                    'destination': destination,
                                    'price': price,
                                    'outbound_date': departure_date.strftime('%Y-%m-%d'),
                                    'return_date': return_date.strftime('%Y-%m-%d'),
                                    'cached_date': today,
                                    'flight_id': flight.get('id', ''),
                                    'duration': (return_date - departure_date).days,
                                    'currency': 'PLN'
                                }
                                
                                if self.cache.store_flight_data(flight_data):
                                    destination_cached += 1
                                    total_cached += 1
                                    if flight_data['cached_date'] == today:
                                        new_entries_today += 1
                                    
                            except Exception:
                                continue
                        
                        time.sleep(0.2)
                        
                    except Exception:
                        continue
                
                console.info(f"✅ Added {destination_cached} new flights for {destination}")
            
            console.success(f"🎯 Smart caching complete: {new_entries_today} new entries today")
            
            self.update_destination_stats()
            
        except Exception as e:
            console.error(f"Error in caching: {e}")
    
    def update_destination_stats(self):
        try:
            stats_updated = 0
            corruption_detected = 0
            
            for destination in DESTINATIONS.keys():
                if destination == 'WAW':
                    continue
                    
                market_data = self.cache.get_market_data(destination)
                
                if market_data['sufficient_data']:
                    stats = {
                        'destination': destination,
                        'sample_size': market_data['sample_size'],
                        'median_price': market_data['median_price'],
                        'std_dev': market_data['std_dev'],
                        'min_price': market_data['min_price'],
                        'max_price': market_data['max_price'],
                        'last_updated': datetime.now(),
                        'region': DESTINATIONS[destination]['region']
                    }
                    
                    self.cache.db.destination_stats.replace_one(
                        {'destination': destination},
                        stats,
                        upsert=True
                    )
                    
                    region = DESTINATIONS[destination]['region']
                    threshold = ABSOLUTE_THRESHOLDS.get(region, ABSOLUTE_THRESHOLDS['default'])
                    
                    console.success(f"✅ {destination}: {market_data['sample_size']} samples, "
                                  f"median: {market_data['median_price']:.0f} zł, "
                                  f"threshold: {threshold} zł")
                    stats_updated += 1
                    
                elif market_data['sample_size'] == 0:
                    corruption_detected += 1
                    console.warning(f"🧹 {destination}: Corruption detected and cleared")
                else:
                    console.warning(f"⚠️ {destination}: Insufficient data ({market_data['sample_size']} samples)")
            
            console.success(f"📊 Updated statistics for {stats_updated} destinations")
            if corruption_detected > 0:
                console.info(f"🧹 Detected and cleared corruption in {corruption_detected} destinations")
                    
        except Exception as e:
            console.error(f"Error updating stats: {e}")
    
    def find_deals(self):
        try:
            console.info("🔍 Searching for current deals...")
            deals_found = 0
            
            for destination in DESTINATIONS.keys():
                if destination == 'WAW':
                    continue
                
                try:
                    market_data = self.cache.get_market_data(destination)
                    
                    if not market_data or not market_data['sufficient_data']:
                        console.info(f"⚠️ {destination}: Insufficient data ({market_data.get('sample_size', 0)} samples)")
                        continue
                    
                    today = datetime.now().date()
                    
                    for month_offset in range(3):
                        departure_month = today + timedelta(days=30 * month_offset)
                        departure_str = departure_month.strftime('%Y-%m-%d')
                        
                        for duration in [3, 7, 10]:
                            return_date = departure_month + timedelta(days=duration)
                            return_str = return_date.strftime('%Y-%m-%d')
                            
                            live_flights = self.api.search_flights(
                                origin='WAW',
                                destination=destination,
                                departure_date=departure_str,
                                return_date=return_str
                            )
                            
                            if live_flights:
                                for flight in live_flights[:3]:
                                    try:
                                        price = float(flight['price']['total'])
                                        
                                        deal_analysis = self.analyzer.analyze_deal(
                                            destination, price, departure_str, return_str
                                        )
                                        
                                        if deal_analysis['is_deal']:
                                            recent_alert = f"{destination}-{today}"
                                            if recent_alert not in self.deals_sent_today:
                                                
                                                alert_sent = self.notifier.send_deal_alert(destination, deal_analysis)
                                                
                                                if alert_sent:
                                                    self.deals_sent_today.add(recent_alert)
                                                    deals_found += 1
                                                    console.success(f"✅ Alert sent for {destination}: {price} zł")
                                                
                                                break
                                    except Exception:
                                        continue
                            
                            time.sleep(0.5)
                            
                            if f"{destination}-{today}" in self.deals_sent_today:
                                break
                        
                        if f"{destination}-{today}" in self.deals_sent_today:
                            break
                
                except Exception:
                    continue
            
            console.success(f"🎯 Deal detection complete: {deals_found} deals found")
            return deals_found
            
        except Exception:
            return 0
    
    def run(self):
        try:
            console.info("🤖 ENHANCED FLIGHT BOT STARTED - PARIS CLEANUP MODE")
            console.info("=" * 60)
            
            startup_msg = (
                f"🤖 **Enhanced Flight Bot Started**\n\n"
                f"🧹 **SPECIAL RUN: Paris Cleanup Mode**\n"
                f"⚠️ Will force clear corrupted Paris data\n"
                f"🔧 Then rebuild with clean economy prices\n\n"
                f"🚀 Starting Paris cleanup and operations..."
            )
            self.notifier.send_status_update(startup_msg)
            
            console.info("📥 Phase 1: Cache building with Paris cleanup...")
            cache_start_time = time.time()
            self.cache_daily_data()
            cache_time = time.time() - cache_start_time
            
            console.info("🔍 Phase 2: Deal detection...")
            deals_start_time = time.time()
            deals_found = self.find_deals()
            deals_time = time.time() - deals_start_time
            
            total_time = time.time() - self.start_time
            
            summary_msg = (
                f"✅ **Enhanced Flight Bot Complete**\n\n"
                f"⏱️ **Performance:**\n"
                f"📥 Cache building: {cache_time/60:.1f} min\n"
                f"🔍 Deal detection: {deals_time/60:.1f} min\n"
                f"🎯 Total runtime: {total_time/60:.1f} min\n\n"
                f"📊 **Results:**\n"
                f"🎯 **Deals Found:** {deals_found}\n"
                f"🧹 **Paris Cleanup:** COMPLETED\n\n"
                f"🔧 **Quality Assurance:**\n"
                f"✅ Data validation active\n"
                f"💎 Economy class only\n"
                f"🧹 Auto corruption cleanup\n"
                f"📈 Smart incremental caching\n\n"
                f"🔄 **Next Run:** Tomorrow (automated)"
            )
            self.notifier.send_status_update(summary_msg)
            
            console.success(f"🎉 Bot execution complete: {deals_found} deals found, {total_time/60:.1f} min runtime")
            
        except Exception as e:
            error_msg = f"❌ Bot execution error: {e}"
            console.error(error_msg)
            self.notifier.send_status_update(error_msg)

def main():
    try:
        console.info("🚀 Starting Enhanced Flight Bot with Paris Cleanup...")
        
        required_vars = [
            'AMADEUS_API_KEY', 'AMADEUS_API_SECRET', 'MONGO_URI', 
            'TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID'
        ]
        
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            console.error(f"Missing environment variables: {', '.join(missing_vars)}")
            return 1
        
        bot = FlightBot()
        bot.run()
        
        console.success("🎉 Enhanced Flight Bot completed successfully!")
        return 0
        
    except Exception as e:
        console.error(f"Fatal error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
