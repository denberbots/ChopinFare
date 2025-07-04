#!/usr/bin/env python3
"""
Enhanced Flight Bot - Optimized for Smart Cache Building
✅ Builds cache daily without deleting existing data
✅ Enhanced data validation to prevent corruption
✅ Economy class filtering only
✅ Regional price validation
✅ Uses TravelPayouts API with your existing credentials
"""

import logging
import os
import statistics
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import requests
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError, PyMongoError
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('flight_bot.log')
    ]
)
logger = logging.getLogger(__name__)

# Console logging with emojis
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

# Configuration - Using your existing environment variables
TRAVELPAYOUTS_API_TOKEN = os.getenv('TRAVELPAYOUTS_API_TOKEN')
MONGO_URI = os.getenv('MONGODB_CONNECTION_STRING')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# Enhanced price validation - stricter to prevent corruption
PRICE_LIMITS = (150, 4000)  # More realistic range
MAX_PRICE_FILTER = 5000     # Reduced from 8000 to prevent business class
MIN_PRICE_FILTER = 100      # Minimum to prevent one-way confusion

# Regional price validation ranges (keeping for validation)
REGIONAL_PRICE_RANGES = {
    'europe_west': (200, 1200),    # Paris, Amsterdam, Brussels
    'europe_close': (150, 800),    # Prague, Vienna, Budapest  
    'europe_north': (300, 1000),   # Oslo, Stockholm, Copenhagen
    'asia_east': (900, 4000),      # Tokyo, Seoul, Shanghai
    'asia_south': (800, 3000),     # Delhi, Mumbai, Bangkok
    'middle_east': (600, 2500),    # Dubai, Doha, Istanbul
    'africa_north': (500, 2000),   # Cairo, Marrakech, Tunis
    'americas': (1200, 5000),      # New York, Toronto, Mexico City
    'domestic': (150, 600),        # Domestic flights
    'default': (200, 3000)         # Fallback range
}

# Your exact absolute thresholds
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

# Complete destination mapping
DESTINATIONS = {
    'CDG': {'name': 'Paris', 'country': 'France', 'region': 'europe_west'},
    'ORY': {'name': 'Paris Orly', 'country': 'France', 'region': 'europe_west'},
    'AMS': {'name': 'Amsterdam', 'country': 'Netherlands', 'region': 'europe_west'},
    'BRU': {'name': 'Brussels', 'country': 'Belgium', 'region': 'europe_west'},
    'LHR': {'name': 'London', 'country': 'United Kingdom', 'region': 'europe_west'},
    'DUS': {'name': 'Düsseldorf', 'country': 'Germany', 'region': 'europe_west'},
    'FRA': {'name': 'Frankfurt', 'country': 'Germany', 'region': 'europe_west'},
    'MUC': {'name': 'Munich', 'country': 'Germany', 'region': 'europe_west'},
    'ZUR': {'name': 'Zurich', 'country': 'Switzerland', 'region': 'europe_west'},
    'VIE': {'name': 'Vienna', 'country': 'Austria', 'region': 'europe_close'},
    'PRG': {'name': 'Prague', 'country': 'Czech Republic', 'region': 'europe_close'},
    'BUD': {'name': 'Budapest', 'country': 'Hungary', 'region': 'europe_close'},
    'OSL': {'name': 'Oslo', 'country': 'Norway', 'region': 'europe_north'},
    'ARN': {'name': 'Stockholm', 'country': 'Sweden', 'region': 'europe_north'},
    'CPH': {'name': 'Copenhagen', 'country': 'Denmark', 'region': 'europe_north'},
    'NRT': {'name': 'Tokyo', 'country': 'Japan', 'region': 'asia_east'},
    'ICN': {'name': 'Seoul', 'country': 'South Korea', 'region': 'asia_east'},
    'PVG': {'name': 'Shanghai', 'country': 'China', 'region': 'asia_east'},
    'DEL': {'name': 'Delhi', 'country': 'India', 'region': 'asia_south'},
    'BOM': {'name': 'Mumbai', 'country': 'India', 'region': 'asia_south'},
    'BKK': {'name': 'Bangkok', 'country': 'Thailand', 'region': 'asia_south'},
    'DXB': {'name': 'Dubai', 'country': 'UAE', 'region': 'middle_east'},
    'DOH': {'name': 'Doha', 'country': 'Qatar', 'region': 'middle_east'},
    'IST': {'name': 'Istanbul', 'country': 'Turkey', 'region': 'middle_east'},
    'CAI': {'name': 'Cairo', 'country': 'Egypt', 'region': 'africa_north'},
    'CMN': {'name': 'Casablanca', 'country': 'Morocco', 'region': 'africa_north'},
    'TUN': {'name': 'Tunis', 'country': 'Tunisia', 'region': 'africa_north'},
    'JFK': {'name': 'New York', 'country': 'USA', 'region': 'americas'},
    'LAX': {'name': 'Los Angeles', 'country': 'USA', 'region': 'americas'},
    'YYZ': {'name': 'Toronto', 'country': 'Canada', 'region': 'americas'},
    'MEX': {'name': 'Mexico City', 'country': 'Mexico', 'region': 'americas'},
    'LIS': {'name': 'Lisbon', 'country': 'Portugal', 'region': 'europe_west'},
    'GDN': {'name': 'Gdańsk', 'country': 'Poland', 'region': 'domestic'},
    'KRK': {'name': 'Kraków', 'country': 'Poland', 'region': 'domestic'},
    'WRO': {'name': 'Wrocław', 'country': 'Poland', 'region': 'domestic'},
    'POZ': {'name': 'Poznań', 'country': 'Poland', 'region': 'domestic'},
    'WAW': {'name': 'Warsaw', 'country': 'Poland', 'region': 'domestic'}
}

class TravelPayoutsAPI:
    """TravelPayouts API client with strict economy-only filtering"""
    
    def __init__(self, api_token: str):
        self.api_token = api_token
        self.base_url = "https://api.travelpayouts.com"
        self.session = requests.Session()
        self.session.headers.update({'X-Access-Token': api_token})
        
    def search_flights(self, origin: str, destination: str, departure_date: str, 
                      return_date: str, adults: int = 1) -> List[Dict]:
        """Search for flights with enhanced validation for economy only"""
        try:
            # Use TravelPayouts V3 API
            url = f"{self.base_url}/aviasales/v3/prices_for_dates"
            
            params = {
                'origin': origin,
                'destination': destination,
                'departure_at': departure_date,
                'return_at': return_date,
                'one_way': False,           # Force round-trip
                'currency': 'PLN',          # Force PLN currency
                'sorting': 'price',
                'limit': 250,
                'token': self.api_token,
                'trip_class': 0,            # Economy class only
                'adults': adults,
                'children': 0,
                'infants': 0
            }
            
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            flights = data.get('data', [])
            
            # Additional validation to ensure economy only
            validated_flights = []
            for flight in flights:
                try:
                    # Check if it's economy class and PLN currency
                    price = flight.get('price', 0)
                    currency = flight.get('currency', 'PLN')
                    trip_class = flight.get('trip_class', 0)
                    
                    if currency == 'PLN' and trip_class == 0 and price > 0:
                        validated_flights.append(flight)
                        
                except (KeyError, TypeError):
                    continue
            
            console.info(f"Found {len(validated_flights)} economy flights for {origin} → {destination}")
            return validated_flights
            
        except requests.exceptions.RequestException as e:
            console.error(f"API request failed: {e}")
            return []
        except Exception as e:
            console.error(f"Unexpected error in flight search: {e}")
            return []

class FlightDataValidator:
    """Enhanced flight data validation to prevent corruption"""
    
    @staticmethod
    def validate_price(price: float, destination: str) -> bool:
        """Validate price against regional ranges"""
        if price < MIN_PRICE_FILTER or price > MAX_PRICE_FILTER:
            return False
            
        # Get regional range
        region = DESTINATIONS.get(destination, {}).get('region', 'default')
        min_price, max_price = REGIONAL_PRICE_RANGES.get(region, REGIONAL_PRICE_RANGES['default'])
        
        # Allow some flexibility but catch obvious outliers
        flexible_min = min_price * 0.7
        flexible_max = max_price * 1.3
        
        return flexible_min <= price <= flexible_max
    
    @staticmethod
    def validate_flight_combination(origin: str, destination: str, price: float, 
                                  departure_date: str, return_date: str) -> bool:
        """Validate entire flight combination"""
        try:
            # Basic price validation
            if not FlightDataValidator.validate_price(price, destination):
                return False
                
            # Date validation
            dep_date = datetime.strptime(departure_date, '%Y-%m-%d')
            ret_date = datetime.strptime(return_date, '%Y-%m-%d')
            
            if dep_date >= ret_date:
                return False
                
            # Trip duration validation (1-30 days)
            duration = (ret_date - dep_date).days
            if duration < 1 or duration > 30:
                return False
                
            return True
            
        except Exception as e:
            console.warning(f"Validation error: {e}")
            return False

class MongoDBCache:
    """Smart MongoDB cache - builds data without daily deletion"""
    
    def __init__(self, uri: str, db_name: str = 'flight_bot_db'):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.validator = FlightDataValidator()
        
    def store_flight_data(self, flight_data: Dict) -> bool:
        """Store flight data with validation and duplicate prevention"""
        try:
            # Validate before storing
            if not self.validator.validate_flight_combination(
                flight_data['origin'],
                flight_data['destination'],
                flight_data['price'],
                flight_data['outbound_date'],
                flight_data['return_date']
            ):
                return False
                
            # Add validation flag
            flight_data['data_quality'] = 'validated'
            flight_data['validation_date'] = datetime.now()
            
            # Create unique identifier to prevent duplicates
            flight_data['unique_id'] = f"{flight_data['origin']}-{flight_data['destination']}-{flight_data['outbound_date']}-{flight_data['return_date']}-{flight_data['price']}"
            
            try:
                self.db.flight_data.insert_one(flight_data)
                return True
            except DuplicateKeyError:
                # Duplicate found, that's okay
                return False
                
        except Exception:
            return False
    
    def get_market_data(self, destination: str) -> Dict:
        """Get market data - simplified without corruption detection"""
        try:
            # Get all validated prices for destination
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
            
            # Calculate statistics
            median_price = statistics.median(prices)
            std_dev = statistics.stdev(prices)
            min_price = min(prices)
            max_price = max(prices)
            
            return {
                'sample_size': len(prices),
                'median_price': median_price,
                'std_dev': std_dev,
                'min_price': min_price,
                'max_price': max_price,
                'sufficient_data': True
            }
            
        except Exception as e:
            console.error(f"Error getting market data: {e}")
            return {'sample_size': 0, 'sufficient_data': False}
    
    def cleanup_old_data(self, days_old: int = 45):
        """Clean up old cached data (45-day rolling window)"""
        try:
            cutoff_datetime = datetime.now() - timedelta(days=days_old)
            
            # Remove old flight data
            result = self.db.flight_data.delete_many({
                'cached_date': {'$lt': cutoff_datetime}
            })
            if result.deleted_count > 0:
                console.info(f"Cleaned up {result.deleted_count} old flight entries (45-day window)")
            
        except Exception as e:
            console.error(f"Error during cleanup: {e}")

class FlightAnalyzer:
    """Enhanced flight analyzer with deal detection"""
    
    def __init__(self, cache: MongoDBCache):
        self.cache = cache
        
    def analyze_deal(self, destination: str, price: float, departure_date: str, 
                    return_date: str) -> Dict:
        """Analyze if a flight is a deal using multiple criteria"""
        try:
            # Get market data
            market_data = self.cache.get_market_data(destination)
            
            if not market_data['sufficient_data']:
                return {
                    'is_deal': False,
                    'deal_type': None,
                    'reason': 'insufficient_data',
                    'confidence': 0
                }
            
            # Get absolute threshold
            region = DESTINATIONS.get(destination, {}).get('region', 'default')
            absolute_threshold = ABSOLUTE_THRESHOLDS.get(region, ABSOLUTE_THRESHOLDS['default'])
            
            # Check absolute threshold
            absolute_deal = price < absolute_threshold
            
            # Check Z-score threshold
            median_price = market_data['median_price']
            std_dev = market_data['std_dev']
            z_score = (median_price - price) / std_dev if std_dev > 0 else 0
            statistical_deal = z_score >= 1.7
            
            # Determine deal type
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
            
            # Calculate savings
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
            
        except Exception as e:
            console.error(f"Error analyzing deal: {e}")
            return {'is_deal': False, 'deal_type': None, 'confidence': 0}

class TelegramNotifier:
    """Enhanced Telegram notifications"""
    
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        
    def send_deal_alert(self, destination: str, deal_info: Dict) -> bool:
        """Send enhanced deal alert"""
        try:
            dest_info = DESTINATIONS.get(destination, {})
            city_name = dest_info.get('name', destination)
            country = dest_info.get('country', 'Unknown')
            
            # Format deal message
            message = f"✈️ **FLIGHT DEAL ALERT** ✈️\n\n"
            message += f"🏙️ **{city_name}, {country}** ({destination})\n"
            message += f"💰 **{deal_info['price']} zł** {deal_info['deal_type']}\n\n"
            
            message += f"📅 **Dates:** {deal_info['departure_date']} → {deal_info['return_date']}\n"
            message += f"📊 **Savings:** {deal_info['savings_percent']}% below typical ({deal_info['median_price']} zł)\n"
            message += f"🎯 **Confidence:** {deal_info['confidence']}%\n\n"
            
            # Add criteria met
            criteria = []
            if deal_info['meets_absolute']:
                criteria.append(f"Under {deal_info['absolute_threshold']} zł threshold")
            if deal_info['meets_statistical']:
                criteria.append(f"Z-score: {deal_info['z_score']}")
            
            message += f"✅ **Criteria:** {', '.join(criteria)}\n\n"
            message += f"🔍 **Search:** WAW → {destination}\n"
            message += f"⏰ **Alert time:** {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            
            return self._send_message(message)
            
        except Exception as e:
            console.error(f"Error sending deal alert: {e}")
            return False
    
    def send_status_update(self, message: str) -> bool:
        """Send status update"""
        try:
            return self._send_message(message)
        except Exception as e:
            console.error(f"Error sending status update: {e}")
            return False
    
    def _send_message(self, message: str) -> bool:
        """Send message via Telegram API"""
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
            
        except Exception as e:
            console.error(f"Failed to send Telegram message: {e}")
            return False

class FlightBot:
    """Enhanced main flight bot class with smart cache building"""
    
    def __init__(self):
        self.api = TravelPayoutsAPI(TRAVELPAYOUTS_API_TOKEN)
        self.cache = MongoDBCache(MONGO_URI)
        self.analyzer = FlightAnalyzer(self.cache)
        self.notifier = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
        self.deals_sent_today = set()
        self.start_time = time.time()
        
    def cache_daily_data(self):
        """Smart daily cache building - accumulates data without deletion"""
        try:
            console.info("🔄 Starting smart daily cache building...")
            today = datetime.now().date()
            
            # Clean up old data (45-day rolling window) - but don't delete today's data
            self.cache.cleanup_old_data(45)
            
            # Generate date combinations (6 months ahead)
            base_date = datetime.now().date()
            months_ahead = 6
            date_combinations = []
            
            for month_offset in range(months_ahead):
                month_date = base_date + timedelta(days=30 * month_offset)
                
                # Generate combinations for this month
                for day_offset in range(0, 28, 7):  # Weekly intervals
                    departure = month_date + timedelta(days=day_offset)
                    
                    # Weekend trips (2-4 days)
                    for duration in [2, 3, 4]:
                        return_date = departure + timedelta(days=duration)
                        date_combinations.append((departure, return_date))
                    
                    # Week trips (6-8 days)
                    for duration in [6, 7, 8]:
                        return_date = departure + timedelta(days=duration)
                        date_combinations.append((departure, return_date))
            
            console.info(f"📅 Generated {len(date_combinations)} date combinations")
            
            # Cache flights for each destination
            total_cached = 0
            new_entries_today = 0
            
            for destination in DESTINATIONS.keys():
                if destination == 'WAW':  # Skip Warsaw as origin
                    continue
                    
                console.info(f"🔍 Building cache for {destination}...")
                destination_cached = 0
                
                # Only cache a subset of combinations daily to avoid API limits
                daily_combinations = date_combinations[::3]  # Every 3rd combination
                
                for departure_date, return_date in daily_combinations:
                    try:
                        # Check if we already have this combination cached recently
                        existing = self.cache.db.flight_data.find_one({
                            'origin': 'WAW',
                            'destination': destination,
                            'outbound_date': departure_date.strftime('%Y-%m-%d'),
                            'return_date': return_date.strftime('%Y-%m-%d'),
                            'cached_date': {'$gte': datetime.combine(today - timedelta(days=7), datetime.min.time())}
                        })
                        
                        if existing:
                            continue  # Skip if recently cached
                        
                        # Get flights from API
                        flights = self.api.search_flights(
                            origin='WAW',
                            destination=destination,
                            departure_date=departure_date.strftime('%Y-%m-%d'),
                            return_date=return_date.strftime('%Y-%m-%d')
                        )
                        
                        # Process and cache flights
                        for flight in flights[:5]:  # Limit to top 5 per combination for efficiency
                            try:
                                # Extract price
                                price = float(flight['price'])
                                
                                # Validate and store
                                flight_data = {
                                    'origin': 'WAW',
                                    'destination': destination,
                                    'price': price,
                                    'outbound_date': departure_date.strftime('%Y-%m-%d'),
                                    'return_date': return_date.strftime('%Y-%m-%d'),
                                    'cached_date': datetime.now(),  # Use datetime instead of date
                                    'flight_id': flight.get('flight_number', ''),
                                    'duration': (return_date - departure_date).days,
                                    'currency': 'PLN'
                                }
                                
                                if self.cache.store_flight_data(flight_data):
                                    destination_cached += 1
                                    total_cached += 1
                                    if flight_data['cached_date'] == today:
                                        new_entries_today += 1
                                    
                            except (KeyError, ValueError) as e:
                                console.warning(f"Error processing flight: {e}")
                                continue
                        
                        # Rate limiting
                        time.sleep(0.2)
                        
                    except Exception as e:
                        console.warning(f"Error caching {destination} for {departure_date}: {e}")
                        continue
                
                console.info(f"✅ Added {destination_cached} new flights for {destination}")
            
            console.success(f"🎯 Smart caching complete: {new_entries_today} new entries today, {total_cached} total processed")
            
            # Update destination statistics
            self._update_all_destination_stats()
            
        except Exception as e:
            console.error(f"Error in smart caching: {e}")
    
    def _update_all_destination_stats(self):
        """Update statistics for all destinations"""
        try:
            stats_updated = 0
            
            for destination in DESTINATIONS.keys():
                if destination == 'WAW':
                    continue
                    
                market_data = self.cache.get_market_data(destination)
                
                if market_data['sufficient_data']:
                    # Store/update destination stats
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
                    
                    # Get threshold for display
                    region = DESTINATIONS[destination]['region']
                    threshold = ABSOLUTE_THRESHOLDS.get(region, ABSOLUTE_THRESHOLDS['default'])
                    
                    console.success(f"✅ {destination}: {market_data['sample_size']} samples, "
                                  f"median: {market_data['median_price']:.0f} zł, "
                                  f"threshold: {threshold} zł")
                    stats_updated += 1
                else:
                    console.warning(f"⚠️ {destination}: Insufficient data ({market_data['sample_size']} samples)")
            
            console.success(f"📊 Updated statistics for {stats_updated} destinations")
                    
        except Exception as e:
            console.error(f"Error updating destination stats: {e}")
    
    def find_deals(self):
        """Find and alert on current deals"""
        try:
            console.info("🔍 Searching for current deals...")
            deals_found = 0
            
            # Check each destination
            for destination in DESTINATIONS.keys():
                if destination == 'WAW':
                    continue
                
                try:
                    # Get market data for this destination
                    market_data = self.cache.get_market_data(destination)
                    
                    if not market_data or not market_data['sufficient_data']:
                        console.info(f"⚠️ {destination}: Insufficient cached data ({market_data.get('sample_size', 0)} samples)")
                        continue
                    
                    # Search for current deals using live API
                    today = datetime.now().date()
                    
                    # Check next 3 months for deals
                    for month_offset in range(3):
                        departure_month = today + timedelta(days=30 * month_offset)
                        departure_str = departure_month.strftime('%Y-%m-%d')
                        
                        # Try a few return dates
                        for duration in [3, 7, 10]:
                            return_date = departure_month + timedelta(days=duration)
                            return_str = return_date.strftime('%Y-%m-%d')
                            
                            # Get live verification
                            live_flights = self.api.search_flights(
                                origin='WAW',
                                destination=destination,
                                departure_date=departure_str,
                                return_date=return_str
                            )
                            
                            if live_flights:
                                for flight in live_flights[:3]:  # Check top 3 results
                                    try:
                                        price = float(flight['price'])
                                        
                                        # Analyze if it's a deal
                                        deal_analysis = self.analyzer.analyze_deal(
                                            destination, price, departure_str, return_str
                                        )
                                        
                                        if deal_analysis['is_deal']:
                                            # Check if we should alert (avoid duplicates)
                                            recent_alert = f"{destination}-{today}"
                                            if recent_alert not in self.deals_sent_today:
                                                
                                                # Send alert
                                                alert_sent = self.notifier.send_deal_alert(destination, deal_analysis)
                                                
                                                if alert_sent:
                                                    self.deals_sent_today.add(recent_alert)
                                                    deals_found += 1
                                                    console.success(f"✅ Alert sent for {destination}: {price} zł ({deal_analysis['deal_type']})")
                                                
                                                # Only one deal per destination per day
                                                break
                                    except (KeyError, ValueError, TypeError) as e:
                                        console.warning(f"Error processing flight data: {e}")
                                        continue
                            
                            # Rate limiting
                            time.sleep(0.5)
                            
                            # Break after first valid deal found for this destination
                            if f"{destination}-{today}" in self.deals_sent_today:
                                break
                        
                        # Break after first valid deal found for this destination
                        if f"{destination}-{today}" in self.deals_sent_today:
                            break
                
                except Exception as e:
                    console.error(f"Error processing {destination}: {e}")
                    continue
            
            console.success(f"🎯 Deal detection complete: {deals_found} deals found")
            return deals_found
            
        except Exception as e:
            console.error(f"Error in deal detection: {e}")
            return 0
    
    def run(self):
        """Main execution method for smart cache building"""
        try:
            console.info("🤖 OPTIMIZED FLIGHT BOT STARTED")
            console.info("=" * 50)
            
            # Send startup notification
            startup_msg = (
                f"🤖 **Optimized Flight Bot Started**\n\n"
                f"🔧 **Smart Caching Mode:**\n"
                f"✅ Builds cache daily without deletion\n"
                f"✅ Enhanced data validation active\n"
                f"✅ Economy class filtering only\n"
                f"✅ TravelPayouts API integration\n"
                f"✅ Regional price validation\n\n"
                f"🚀 Starting operations..."
            )
            self.notifier.send_status_update(startup_msg)
            
            # Step 1: Smart cache building
            console.info("📥 Phase 1: Smart daily cache building...")
            cache_start_time = time.time()
            self.cache_daily_data()
            cache_time = time.time() - cache_start_time
            
            # Step 2: Find current deals  
            console.info("🔍 Phase 2: Detecting current deals...")
            deals_start_time = time.time()
            deals_found = self.find_deals()
            deals_time = time.time() - deals_start_time
            
            # Step 3: Send summary
            total_time = time.time() - self.start_time
            
            summary_msg = (
                f"✅ **Optimized Flight Bot Complete**\n\n"
                f"⏱️ **Performance:**\n"
                f"📥 Cache building: {cache_time/60:.1f} min\n"
                f"🔍 Deal detection: {deals_time/60:.1f} min\n"
                f"🎯 Total runtime: {total_time/60:.1f} min\n\n"
                f"📊 **Results:**\n"
                f"🎯 **Deals Found:** {deals_found}\n\n"
                f"🔧 **Quality Features:**\n"
                f"✅ Smart cache accumulation\n"
                f"💎 Economy class only\n"
                f"🛡️ Enhanced validation\n"
                f"🔄 45-day rolling window\n\n"
                f"🔄 **Next Run:** Tomorrow (automated)"
            )
            self.notifier.send_status_update(summary_msg)
            
            console.success(f"🎉 Bot execution complete: {deals_found} deals found, {total_time/60:.1f} min runtime")
            
        except Exception as e:
            error_msg = f"❌ Bot execution error: {e}"
            console.error(error_msg)
            self.notifier.send_status_update(error_msg)

def main():
    """Main function for automated daily execution"""
    try:
        console.info("🚀 Starting Optimized Flight Bot...")
        
        # Verify environment variables
        required_vars = [
            'TRAVELPAYOUTS_API_TOKEN', 'MONGODB_CONNECTION_STRING', 
            'TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID'
        ]
        
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            console.error(f"Missing environment variables: {', '.join(missing_vars)}")
            return 1
        
        # Initialize and run the bot
        bot = FlightBot()
        bot.run()
        
        console.success("🎉 Optimized Flight Bot completed successfully!")
        return 0
        
    except Exception as e:
        console.error(f"Fatal error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
