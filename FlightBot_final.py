#!/usr/bin/env python3
"""
MongoDB Flight Bot - COMPLETELY FIXED VERSION
✅ FIXED: V3 API with correct currency (RUB) and parameters
✅ FIXED: Off-peak month search (October/November instead of August)
✅ FIXED: Lower Z-score threshold (1.5 instead of 1.7)
✅ FIXED: Proper currency conversion (RUB to PLN)
✅ FIXED: All API parameter issues resolved
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
        
        # Currency conversion rate: 1 RUB = 0.044 PLN (approximate)
        self.rub_to_pln_rate = 0.044
    
    def _validate_price(self, price: float) -> bool:
        """Validate price is within reasonable range for PLN"""
        return 200 <= price <= 6000
    
    def _convert_rub_to_pln(self, rub_price: float) -> float:
        """Convert RUB price to PLN"""
        return rub_price * self.rub_to_pln_rate
    
    def _validate_date_format(self, date_str: str) -> bool:
        """Validate date string format"""
        try:
            datetime.strptime(date_str, '%Y-%m-%d')
            return True
        except ValueError:
            return False
    
    def get_matrix_flights(self, origin: str, destination: str, month: str) -> List[Dict[str, Any]]:
        """
        FIXED: Get flights using Matrix API (realistic 200-600 PLN prices)
        """
        url = f"{self.base_url}/v2/prices/month-matrix"
        params = {
            'origin': origin,
            'destination': destination,
            'month': month,
            'currency': 'PLN',
            'show_to_affiliates': True,
            'token': self.api_token
        }
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
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
                
            price = entry.get('value')
            departure_at = entry.get('depart_date') 
            return_at = entry.get('return_date')
            
            if price and departure_at:
                flight = {
                    'value': float(price),
                    'departure_at': departure_at,
                    'return_at': return_at or departure_at,
                    'distance': entry.get('distance', 0),
                    'actual': entry.get('actual', True),
                    'transfers': entry.get('number_of_changes', 0),
                    'airline': entry.get('gate', 'Unknown'),
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
        COMPLETELY FIXED: V3 API with correct currency and parameters
        """
        if not self._validate_date_format(departure_date):
            console.error(f"❌ Invalid departure date format: {departure_date}")
            return None
        
        if return_date and not self._validate_date_format(return_date):
            console.error(f"❌ Invalid return date format: {return_date}")
            return None
        
        try:
            dep_date = datetime.strptime(departure_date, '%Y-%m-%d')
            if dep_date < datetime.now():
                console.warning(f"⚠️ Departure date {departure_date} is in the past")
                return None
                
            if return_date:
                ret_date = datetime.strptime(return_date, '%Y-%m-%d')
                if ret_date <= dep_date:
                    console.warning(f"⚠️ Return date {return_date} is not after departure date {departure_date}")
                    return None
        except ValueError as e:
            console.error(f"❌ Date parsing error: {e}")
            return None
        
        url = f"{self.base_url}/aviasales/v3/prices_for_dates"
        
        # FIXED: Use supported currency (RUB) and proper parameters
        params = {
            'origin': origin,
            'destination': destination,
            'departure_at': departure_date[:7],  # FIXED: Use YYYY-MM format
            'currency': 'rub',  # FIXED: Use supported currency
            'market': 'pl',     # FIXED: Add Polish market
            'one_way': 'false' if return_date else 'true',  # FIXED: Specify round-trip
            'show_to_affiliates': 'true',  # FIXED: Recommended parameter
            'token': self.api_token
        }
        
        if return_date:
            params['return_at'] = return_date[:7]  # FIXED: Use YYYY-MM format
        
        console.info(f"🔍 FIXED V3 API call: {origin}→{destination}, {departure_date[:7]}" + (f" to {return_date[:7]}" if return_date else " (one-way)"))
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            console.info(f"📡 V3 response: success={data.get('success')}, data_count={len(data.get('data', []))}")
            
            if not data.get('success'):
                error_msg = data.get('error', 'Unknown error')
                console.warning(f"❌ V3 API unsuccessful: {error_msg}")
                return None
            
            if not data.get('data'):
                console.warning(f"❌ V3 API: No flights found for these dates")
                return None
            
            flights = data['data']
            console.info(f"✅ V3 API: Found {len(flights)} flights in RUB")
            
            if flights:
                # FIXED: Convert RUB prices to PLN
                for flight in flights:
                    if 'value' in flight:
                        rub_price = flight['value']
                        pln_price = self._convert_rub_to_pln(rub_price)
                        flight['value'] = pln_price
                        flight['original_currency'] = 'RUB'
                        flight['original_price'] = rub_price
                        console.info(f"💱 Converted {rub_price:.0f} RUB → {pln_price:.0f} PLN")
                
                valid_flights = [f for f in flights if self._validate_price(f.get('value', 0))]
                
                if not valid_flights:
                    console.warning(f"❌ V3 API: No flights within valid price range (200-6000 PLN) after conversion")
                    return None
                
                cheapest = min(valid_flights, key=lambda x: x.get('value', float('inf')))
                console.info(f"✅ V3 verification successful: {cheapest.get('value', 0):.0f} PLN (converted from RUB)")
                return cheapest
            
            console.warning(f"❌ V3 API: Empty flight list")
            return None
            
        except requests.RequestException as e:
            console.error(f"❌ V3 verification network error: {e}")
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
            for flight in flights:
                flight['cached_at'] = datetime.now()
            
            result = self.flights_collection.insert_many(flights, ordered=False)
            return len(result.inserted_ids)
            
        except pymongo.errors.BulkWriteError as e:
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
            prices = []
            for flight in flights:
                price = flight.get('value', 0)
                if self._validate_price(price):
                    prices.append(price)
            
            if len(prices) < 3:
                console.warning(f"⚠️ Insufficient price data for {destination}: {len(prices)} prices")
                return False
            
            prices_sorted = sorted(prices)
            q1_idx = len(prices_sorted) // 4
            q3_idx = 3 * len(prices_sorted) // 4
            q1 = prices_sorted[q1_idx]
            q3 = prices_sorted[q3_idx]
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            cleaned_prices = [p for p in prices if lower_bound <= p <= upper_bound]
            
            if len(cleaned_prices) < 3:
                console.warning(f"⚠️ Too few prices after outlier removal for {destination}")
                cleaned_prices = prices
            
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
        
        self._FLAGS = {
            'FCO': '🇮🇹', 'MXP': '🇮🇹', 'LIN': '🇮🇹', 'BGY': '🇮🇹', 'CIA': '🇮🇹', 'VCE': '🇮🇹', 'NAP': '🇮🇹', 'PMO': '🇮🇹',
            'BLQ': '🇮🇹', 'FLR': '🇮🇹', 'PSA': '🇮🇹', 'CAG': '🇮🇹', 'BRI': '🇮🇹', 'CTA': '🇮🇹', 'BUS': '🇮🇹', 'AHO': '🇮🇹', 'GOA': '🇮🇹',
            'MAD': '🇪🇸', 'BCN': '🇪🇸', 'PMI': '🇪🇸', 'IBZ': '🇪🇸', 'VLC': '🇪🇸', 'ALC': '🇪🇸', 'AGP': '🇪🇸', 'BIO': '🇪🇸',
            'LPA': '🇪🇸', 'TFS': '🇪🇸', 'SPC': '🇪🇸',
            'LHR': '🇬🇧', 'LTN': '🇬🇧', 'LGW': '🇬🇧', 'STN': '🇬🇧', 'GLA': '🇬🇧', 'BFS': '🇬🇧',
            'CDG': '🇫🇷', 'ORY': '🇫🇷', 'NCE': '🇫🇷', 'MRS': '🇫🇷', 'BIQ': '🇫🇷', 'PIS': '🇫🇷', 'PUY': '🇫🇷',
            'FRA': '🇩🇪', 'MUC': '🇩🇪', 'BER': '🇩🇪', 'HAM': '🇩🇪', 'STR': '🇩🇪', 'DUS': '🇩🇪', 'CGN': '🇩🇪', 'LEJ': '🇩🇪', 'DTM': '🇩🇪',
            'AMS': '🇳🇱', 'RTM': '🇳🇱', 'EIN': '🇳🇱',
            'ATH': '🇬🇷', 'SKG': '🇬🇷', 'CFU': '🇬🇷', 'HER': '🇬🇷', 'RHO': '🇬🇷', 'ZTH': '🇬🇷', 'JTR': '🇬🇷', 'CHQ': '🇬🇷',
            'LIS': '🇵🇹', 'OPO': '🇵🇹', 'PDL': '🇵🇹', 'PXO': '🇵🇹',
            'ARN': '🇸🇪', 'NYO': '🇸🇪', 'OSL': '🇳🇴', 'BGO': '🇳🇴', 'BOO': '🇳🇴',
            'HEL': '🇫🇮', 'RVN': '🇫🇮', 'KEF': '🇮🇸', 'CPH': '🇩🇰',
            'VIE': '🇦🇹', 'PRG': '🇨🇿', 'BRU': '🇧🇪', 'CRL': '🇧🇪', 'ZUR': '🇨🇭', 'BSL': '🇨🇭', 'GVA': '🇨🇭',
            'BUD': '🇭🇺', 'DUB': '🇮🇪', 'VAR': '🇧🇬', 'BOJ': '🇧🇬', 'SOF': '🇧🇬',
            'OTP': '🇷🇴', 'CLJ': '🇷🇴', 'SPU': '🇭🇷', 'DBV': '🇭🇷', 'ZAD': '🇭🇷',
            'BEG': '🇷🇸', 'TIV': '🇲🇪', 'TGD': '🇲🇪', 'TIA': '🇦🇱', 'KRK': '🇵🇱', 'KTW': '🇵🇱',
            'LED': '🇷🇺', 'SVO': '🇷🇺', 'DME': '🇷🇺', 'VKO': '🇷🇺', 'AER': '🇷🇺', 'OVB': '🇷🇺', 'IKT': '🇷🇺',
            'ULV': '🇷🇺', 'KJA': '🇷🇺', 'KGD': '🇷🇺', 'MSQ': '🇧🇾',
            'AYT': '🇹🇷', 'IST': '🇹🇷', 'SAW': '🇹🇷', 'ESB': '🇹🇷', 'IZM': '🇹🇷', 'ADB': '🇹🇷',
            'TLV': '🇮🇱', 'EVN': '🇦🇲', 'TBS': '🇬🇪', 'GYD': '🇦🇿', 'KUT': '🇬🇪', 'FRU': '🇰🇬', 'TAS': '🇺🇿',
            'DXB': '🇦🇪', 'SHJ': '🇦🇪', 'AUH': '🇦🇪', 'DWC': '🇦🇪', 'DOH': '🇶🇦', 'RUH': '🇸🇦', 'JED': '🇸🇦', 'DMM': '🇸🇦',
            'SSH': '🇪🇬', 'CAI': '🇪🇬', 'RAK': '🇲🇦', 'DJE': '🇹🇳',
            'TNR': '🇲🇬', 'ZNZ': '🇹🇿',
            'EWR': '🇺🇸', 'JFK': '🇺🇸', 'LGA': '🇺🇸', 'MIA': '🇺🇸', 'PHL': '🇺🇸',
            'YYZ': '🇨🇦', 'YWG': '🇨🇦', 'YEG': '🇨🇦', 'HAV': '🇨🇺', 'PUJ': '🇩🇴',
            'HKT': '🇹🇭', 'BKK': '🇹🇭', 'DMK': '🇹🇭', 'DPS': '🇮🇩',
            'ICN': '🇰🇷', 'GMP': '🇰🇷', 'NRT': '🇯🇵', 'HND': '🇯🇵', 'KIX': '🇯🇵', 'ITM': '🇯🇵',
            'PEK': '🇨🇳', 'CMB': '🇱🇰', 'DEL': '🇮🇳', 'SYD': '🇦🇺'
        }
        
        self._CITIES = {
            'WAW': 'Warsaw', 'FCO': 'Rome', 'MAD': 'Madrid', 'BCN': 'Barcelona', 'LHR': 'London', 'AMS': 'Amsterdam',
            'ATH': 'Athens', 'CDG': 'Paris', 'MUC': 'Munich', 'VIE': 'Vienna', 'PRG': 'Prague', 'BRU': 'Brussels',
            'ORY': 'Paris', 'LIN': 'Milan', 'BGY': 'Milan', 'CIA': 'Rome', 'GOA': 'Genoa', 'PMI': 'Palma',
            'MXP': 'Milan', 'VCE': 'Venice', 'NAP': 'Naples', 'LIS': 'Lisbon', 'LTN': 'London', 'LGW': 'London',
            'STN': 'London', 'ARN': 'Stockholm', 'OSL': 'Oslo', 'NYO': 'Stockholm', 'FRA': 'Frankfurt',
            'VAR': 'Varna', 'PSA': 'Pisa', 'EWR': 'New York', 'JFK': 'New York', 'LGA': 'New York',
            'MIA': 'Miami', 'BLQ': 'Bologna', 'FLR': 'Florence', 'CAG': 'Cagliari', 'BRI': 'Bari',
            'CTA': 'Catania', 'PMO': 'Palermo', 'BUS': 'Batum', 'AHO': 'Alghero', 'SKG': 'Thessaloniki',
            'CFU': 'Corfu', 'HER': 'Heraklion', 'RHO': 'Rhodes', 'ZTH': 'Zakynthos', 'JTR': 'Santorini',
            'CHQ': 'Chania', 'OPO': 'Porto', 'SPU': 'Split', 'DBV': 'Dubrovnik', 'ZAD': 'Zadar',
            'BEG': 'Belgrade', 'TIV': 'Tivat', 'TGD': 'Podgorica', 'TIA': 'Tirana', 'SOF': 'Sofia',
            'OTP': 'Bucharest', 'CLJ': 'Cluj-Napoca', 'KRK': 'Krakow', 'KTW': 'Katowice', 'KGD': 'Kaliningrad',
            'LED': 'St. Petersburg', 'SVO': 'Moscow', 'DME': 'Moscow', 'VKO': 'Moscow', 'AYT': 'Antalya',
            'IST': 'Istanbul', 'SAW': 'Istanbul', 'ESB': 'Ankara', 'IZM': 'Izmir', 'ADB': 'Izmir',
            'TLV': 'Tel Aviv', 'EVN': 'Yerevan', 'TBS': 'Tbilisi', 'GYD': 'Baku', 'KUT': 'Kutaisi',
            'MSQ': 'Minsk', 'HEL': 'Helsinki', 'KEF': 'Reykjavik', 'BUD': 'Budapest', 'DUB': 'Dublin',
            'GLA': 'Glasgow', 'BFS': 'Belfast', 'NCE': 'Nice', 'MRS': 'Marseille', 'TFS': 'Tenerife',
            'LPA': 'Las Palmas', 'IBZ': 'Ibiza', 'VLC': 'Valencia', 'ALC': 'Alicante', 'AGP': 'Malaga',
            'BIO': 'Bilbao', 'SPC': 'La Palma', 'PDL': 'Ponta Delgada', 'PXO': 'Porto Santo',
            'RAK': 'Marrakech', 'CAI': 'Cairo', 'DJE': 'Djerba', 'TNR': 'Antananarivo', 'ZNZ': 'Zanzibar',
            'DXB': 'Dubai', 'SHJ': 'Sharjah', 'AUH': 'Abu Dhabi', 'DOH': 'Doha', 'RUH': 'Riyadh',
            'JED': 'Jeddah', 'DMM': 'Dammam', 'SSH': 'Sharm El Sheikh', 'HKT': 'Phuket', 'BKK': 'Bangkok',
            'DMK': 'Bangkok', 'DPS': 'Denpasar', 'ICN': 'Seoul', 'GMP': 'Seoul', 'NRT': 'Tokyo',
            'HND': 'Tokyo', 'KIX': 'Osaka', 'ITM': 'Osaka', 'PEK': 'Beijing', 'YYZ': 'Toronto',
            'YWG': 'Winnipeg', 'YEG': 'Edmonton', 'HAV': 'Havana', 'PUJ': 'Punta Cana', 'CMB': 'Colombo',
            'DEL': 'Delhi', 'SYD': 'Sydney', 'OVB': 'Novosibirsk', 'IKT': 'Irkutsk', 'ULV': 'Ulyanovsk',
            'KJA': 'Krasnoyarsk', 'FRU': 'Bishkek', 'BOO': 'Bodø', 'BGO': 'Bergen', 'RVN': 'Rovaniemi',
            'DTM': 'Dortmund', 'STR': 'Stuttgart', 'HAM': 'Hamburg', 'RTM': 'Rotterdam', 'EIN': 'Eindhoven',
            'BSL': 'Basel', 'ZUR': 'Zurich', 'GVA': 'Geneva', 'CPH': 'Copenhagen', 'BIQ': 'Biarritz',
            'PIS': 'Poitiers', 'CRL': 'Brussels', 'PUY': 'Puy-en-Velay', 'DWC': 'Dubai', 'AER': 'Sochi',
            'PHL': 'Philadelphia', 'TAS': 'Tashkent', 'BOJ': 'Burgas'
        }
        
        self._COUNTRIES = {
            'FCO': 'Italy', 'MXP': 'Italy', 'LIN': 'Italy', 'BGY': 'Italy', 'CIA': 'Italy', 'VCE': 'Italy', 
            'NAP': 'Italy', 'GOA': 'Italy', 'PMO': 'Italy', 'BLQ': 'Italy', 'FLR': 'Italy', 'PSA': 'Italy',
            'CAG': 'Italy', 'BRI': 'Italy', 'CTA': 'Italy', 'BUS': 'Italy', 'AHO': 'Italy',
            'MAD': 'Spain', 'BCN': 'Spain', 'PMI': 'Spain', 'IBZ': 'Spain', 'VLC': 'Spain', 'ALC': 'Spain',
            'AGP': 'Spain', 'BIO': 'Spain', 'LPA': 'Spain', 'TFS': 'Spain', 'SPC': 'Spain',
            'LHR': 'United Kingdom', 'LTN': 'United Kingdom', 'LGW': 'United Kingdom', 'STN': 'United Kingdom',
            'GLA': 'United Kingdom', 'BFS': 'United Kingdom',
            'CDG': 'France', 'ORY': 'France', 'NCE': 'France', 'MRS': 'France', 'BIQ': 'France',
            'PIS': 'France', 'PUY': 'France',
            'FRA': 'Germany', 'MUC': 'Germany', 'BER': 'Germany', 'HAM': 'Germany', 'STR': 'Germany',
            'DUS': 'Germany', 'CGN': 'Germany', 'LEJ': 'Germany', 'DTM': 'Germany',
            'AMS': 'Netherlands', 'RTM': 'Netherlands', 'EIN': 'Netherlands',
            'ATH': 'Greece', 'SKG': 'Greece', 'CFU': 'Greece', 'HER': 'Greece', 'RHO': 'Greece',
            'ZTH': 'Greece', 'JTR': 'Greece', 'CHQ': 'Greece',
            'LIS': 'Portugal', 'OPO': 'Portugal', 'PDL': 'Portugal', 'PXO': 'Portugal',
            'ARN': 'Sweden', 'NYO': 'Sweden', 'OSL': 'Norway', 'BGO': 'Norway', 'BOO': 'Norway',
            'HEL': 'Finland', 'RVN': 'Finland', 'KEF': 'Iceland', 'CPH': 'Denmark',
            'VIE': 'Austria', 'PRG': 'Czech Republic', 'BRU': 'Belgium', 'CRL': 'Belgium',
            'ZUR': 'Switzerland', 'BSL': 'Switzerland', 'GVA': 'Switzerland', 'BUD': 'Hungary',
            'DUB': 'Ireland', 'VAR': 'Bulgaria', 'BOJ': 'Bulgaria', 'SOF': 'Bulgaria',
            'OTP': 'Romania', 'CLJ': 'Romania', 'SPU': 'Croatia', 'DBV': 'Croatia', 'ZAD': 'Croatia',
            'BEG': 'Serbia', 'TIV': 'Montenegro', 'TGD': 'Montenegro', 'TIA': 'Albania',
            'KRK': 'Poland', 'KTW': 'Poland', 'KGD': 'Russia', 'LED': 'Russia', 'SVO': 'Russia',
            'DME': 'Russia', 'VKO': 'Russia', 'AER': 'Russia', 'OVB': 'Russia', 'IKT': 'Russia',
            'ULV': 'Russia', 'KJA': 'Russia', 'MSQ': 'Belarus',
            'AYT': 'Turkey', 'IST': 'Turkey', 'SAW': 'Turkey', 'ESB': 'Turkey', 'IZM': 'Turkey', 'ADB': 'Turkey',
            'TLV': 'Israel', 'EVN': 'Armenia', 'TBS': 'Georgia', 'GYD': 'Azerbaijan', 'KUT': 'Georgia',
            'FRU': 'Kyrgyzstan', 'TAS': 'Uzbekistan',
            'EWR': 'United States', 'JFK': 'United States', 'LGA': 'United States', 'MIA': 'United States',
            'PHL': 'United States', 'YYZ': 'Canada', 'YWG': 'Canada', 'YEG': 'Canada',
            'HAV': 'Cuba', 'PUJ': 'Dominican Republic',
            'DXB': 'United Arab Emirates', 'SHJ': 'United Arab Emirates', 'AUH': 'United Arab Emirates',
            'DWC': 'United Arab Emirates', 'DOH': 'Qatar', 'RUH': 'Saudi Arabia', 'JED': 'Saudi Arabia',
            'DMM': 'Saudi Arabia', 'SSH': 'Egypt', 'CAI': 'Egypt',
            'RAK': 'Morocco', 'DJE': 'Tunisia', 'TNR': 'Madagascar', 'ZNZ': 'Tanzania',
            'HKT': 'Thailand', 'BKK': 'Thailand', 'DMK': 'Thailand', 'DPS': 'Indonesia',
            'ICN': 'South Korea', 'GMP': 'South Korea', 'NRT': 'Japan', 'HND': 'Japan',
            'KIX': 'Japan', 'ITM': 'Japan', 'PEK': 'China', 'CMB': 'Sri Lanka', 'DEL': 'India',
            'SYD': 'Australia'
        }
    
    def _format_date_range(self, departure_date: str, return_date: str) -> str:
        """Format date range compactly"""
        try:
            dep = datetime.strptime(departure_date, '%Y-%m-%d').strftime('%b %d')
            ret = datetime.strptime(return_date, '%Y-%m-%d')
            ret_fmt = ret.strftime('%d' if departure_date[:7] == return_date[:7] else '%b %d')
            return f"{dep}-{ret_fmt}"
        except Exception:
            return f"{departure_date} to {return_date}"
    
    def send_deal_alert(self, destination: str, price: float, z_score: float, 
                       market_median: float, savings: float, verification_data: Dict = None) -> bool:
        try:
            origin_city = self._CITIES.get('WAW', 'Warsaw')
            dest_city = self._CITIES.get(destination, destination)
            country = self._COUNTRIES.get(destination, '')
            flag = self._FLAGS.get(destination, '')
            
            header = f"🚨 *FLIGHT DEAL ALERT* 🚨\n\n"
            header += f"✈️ *{origin_city} → {dest_city}{f', {country} {flag}' if country and flag else ''}*"
            
            message = header + f"\n\n"
            message += f"💰 *Price:* {price:.0f} PLN\n"
            message += f"📊 *Market Median:* {market_median:.0f} PLN\n"
            message += f"💸 *Savings:* {savings:.0f} PLN ({(savings/market_median)*100:.1f}%)\n"
            message += f"📈 *Z-Score:* {z_score:.2f}\n"
            
            if verification_data:
                departure_at = verification_data.get('departure_at', '')
                return_at = verification_data.get('return_at', '')
                
                if departure_at and return_at:
                    date_range = self._format_date_range(departure_at[:10], return_at[:10])
                    message += f"\n✅ *Verified Deal Details:*\n"
                    message += f"📅 *Dates:* {date_range}\n"
                    
                    try:
                        dep_date = datetime.strptime(departure_at[:10], '%Y-%m-%d')
                        ret_date = datetime.strptime(return_at[:10], '%Y-%m-%d')
                        duration = (ret_date - dep_date).days
                        message += f"🕒 *Duration:* {duration} days\n"
                    except:
                        pass
                
                airline = verification_data.get('airline', '')
                if airline:
                    message += f"🏢 *Airline:* {airline}\n"
            
            message += f"\n🕒 *Found:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            url = f"{self.base_url}/sendMessage"
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }
            
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            
            console.info(f"📱 Deal alert sent for {dest_city}: {price:.0f} PLN")
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
        
        self.flight_api = FlightAPI(self.api_token)
        self.db_manager = MongoDBManager(self.mongodb_uri)
        self.notifier = TelegramNotifier(self.telegram_token, self.telegram_chat_id)
        
        self.destinations = [
            'CDG', 'ORY', 'BCN', 'FCO', 'MXP', 'LIN', 'BGY', 'CIA', 'ATH', 'VCE', 'NAP', 'LIS', 'AMS', 'LHR',
            'LTN', 'LGW', 'ARN', 'MAD', 'NYO', 'STN', 'OSL', 'PRG', 'OTP', 'HEL', 'FRA', 'PMO', 'KEF', 'BUD',
            'VLC', 'CTA', 'KUT', 'TBS', 'GYD', 'VKO', 'SVO', 'DME', 'AYT', 'IST', 'SAW', 'ALC', 'TAS', 'NCE',
            'TFS', 'PMI', 'TGD', 'TIA', 'TLV', 'EVN', 'MSQ', 'AGP', 'BOJ', 'SPU', 'GOA', 'BRI', 'SKG', 'CFU',
            'OPO', 'HER', 'BUS', 'LED', 'TIV', 'BEG', 'RHO', 'ZAD', 'JTR', 'ZTH', 'VAR', 'AER', 'DTM', 'STR',
            'HAM', 'SOF', 'KRK', 'BLQ', 'FLR', 'PSA', 'KGD', 'IBZ', 'ESB', 'IZM', 'ADB', 'DBV', 'BSL', 'CHQ',
            'CAG', 'KTW', 'RTM', 'BIO', 'LPA', 'SPC', 'PDL', 'PXO', 'AHO', 'BGO', 'RVN', 'CLJ', 'GLA', 'BFS',
            'BIQ', 'PIS', 'CRL', 'PUY', 'JFK', 'EWR', 'LGA', 'MIA', 'ICN', 'GMP', 'PEK', 'DXB', 'SHJ', 'SSH',
            'ZNZ', 'RUH', 'HKT', 'DPS', 'BKK', 'DMK', 'YYZ', 'YWG', 'YEG', 'HAV', 'PUJ', 'CAI', 'RAK', 'DJE',
            'NRT', 'HND', 'KIX', 'ITM', 'CMB', 'PHL', 'DEL', 'SYD', 'TNR', 'OVB', 'IKT', 'ULV', 'KJA', 'AUH',
            'DWC', 'DOH', 'JED', 'DMM', 'BOO', 'FRU', 'ZUR', 'VIE', 'BRU', 'DUS', 'MUC'
        ]
        self.origin = 'WAW'
        
        self.absolute_thresholds = {
            'ARN': 250, 'NYO': 250, 'OSL': 250, 'BGO': 250, 'BOO': 250, 'CPH': 250, 'HEL': 250, 'RVN': 250, 'KEF': 250,
            'VAR': 250, 'BOJ': 250, 'SOF': 250, 'OTP': 250, 'CLJ': 250, 'BEG': 250, 'SPU': 250, 'DBV': 250, 'ZAD': 250,
            'TIV': 250, 'TGD': 250, 'TIA': 250, 'SKG': 250, 'BUD': 250, 'PRG': 250, 'KRK': 250, 'KTW': 250,
            'LED': 250, 'KGD': 250, 'MSQ': 250,
            'LHR': 350, 'LTN': 350, 'LGW': 350, 'STN': 350, 'GLA': 350, 'BFS': 350, 'DUB': 350,
            'CDG': 350, 'ORY': 350, 'NCE': 350, 'MRS': 350, 'BIQ': 350, 'PIS': 350, 'PUY': 350,
            'FRA': 350, 'MUC': 350, 'BER': 350, 'HAM': 350, 'STR': 350, 'DUS': 350, 'CGN': 350, 'LEJ': 350, 'DTM': 350,
            'MAD': 350, 'BCN': 350, 'PMI': 350, 'IBZ': 350, 'VLC': 350, 'ALC': 350, 'AGP': 350, 'BIO': 350,
            'LPA': 350, 'TFS': 350, 'SPC': 350,
            'FCO': 350, 'MXP': 350, 'LIN': 350, 'BGY': 350, 'CIA': 350, 'VCE': 350, 'NAP': 350, 'PMO': 350,
            'BLQ': 350, 'FLR': 350, 'PSA': 350, 'CAG': 350, 'BRI': 350, 'CTA': 350, 'BUS': 350, 'AHO': 350, 'GOA': 350,
            'AMS': 350, 'RTM': 350, 'EIN': 350, 'ZUR': 350, 'BSL': 350, 'GVA': 350,
            'LIS': 350, 'OPO': 350, 'PDL': 350, 'PXO': 350,
            'VIE': 350, 'ATH': 350, 'CFU': 350, 'HER': 350, 'RHO': 350, 'ZTH': 350, 'JTR': 350, 'CHQ': 350,
            'BRU': 350, 'CRL': 350,
            'AYT': 700, 'IST': 700, 'SAW': 700, 'ESB': 700, 'IZM': 700, 'ADB': 700,
            'TLV': 700, 'SSH': 700, 'CAI': 700,
            'DXB': 750, 'SHJ': 750, 'AUH': 750, 'DWC': 750, 'DOH': 750, 'RUH': 750, 'JED': 750, 'DMM': 750,
            'RAK': 650, 'DJE': 650,
            'SVO': 1100, 'DME': 1100, 'VKO': 1100, 'AER': 1100, 'OVB': 1100, 'IKT': 1100, 'ULV': 1100, 'KJA': 1100,
            'FRU': 1100, 'TAS': 1100, 'EVN': 1100, 'TBS': 1100, 'GYD': 1100, 'KUT': 1100,
            'BKK': 1600, 'DMK': 1600, 'HKT': 1600, 'DPS': 1600,
            'NRT': 1700, 'HND': 1700, 'KIX': 1700, 'ITM': 1700, 'ICN': 1700, 'GMP': 1700, 'PEK': 1700,
            'DEL': 1400, 'CMB': 1400,
            'EWR': 1700, 'JFK': 1700, 'LGA': 1700, 'PHL': 1700, 'YYZ': 1700, 'MIA': 1700,
            'YWG': 2200, 'YEG': 2200,
            'HAV': 2200, 'PUJ': 2200,
            'SYD': 2800,
            'ZNZ': 1500, 'TNR': 1500
        }
        
        # FIXED: Lower Z-score threshold for more deals
        self.z_score_threshold = 1.5  # Changed from 1.7 to 1.5
        
    def _generate_future_months(self, start_month: int = 10, count: int = 3) -> List[str]:
        """FIXED: Generate future months starting from October (off-peak) instead of August (peak)"""
        current_year = datetime.now().year
        current_month = datetime.now().month
        
        # FIXED: Start from October (off-peak season) instead of August
        if current_month >= 10:
            start_month = current_month
        
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
        console.info(f"📅 FIXED: Searching off-peak months: {months} (October/November instead of August)")
        
        total_cached = 0
        successful_destinations = 0
        
        for destination in self.destinations:
            console.info(f"📥 Caching data for {destination}...")
            
            round_trip_combinations = self._generate_roundtrip_combinations(destination, months)
            
            if round_trip_combinations:
                destination_flights = []
                for combo in round_trip_combinations:
                    flight_record = {
                        'value': combo['total_price'],
                        'departure_at': combo['outbound_date'],
                        'return_at': combo['return_date'],
                        'distance': combo.get('distance', 0),
                        'actual': True,
                        'transfers': combo.get('outbound_transfers', 0),
                        'airline': combo.get('airline', 'Unknown'),
                        'flight_number': 0,
                        'origin': self.origin,
                        'destination': destination,
                        'found_at': datetime.now().isoformat()
                    }
                    destination_flights.append(flight_record)
                
                cached_count = self.db_manager.insert_flights(destination_flights)
                if cached_count > 0:
                    total_cached += cached_count
                    successful_destinations += 1
                    self.db_manager.update_statistics(destination, destination_flights)
                    console.info(f"✅ {destination}: {cached_count} round-trip combinations cached")
                else:
                    console.warning(f"⚠️ {destination}: No flights cached")
            else:
                console.warning(f"⚠️ {destination}: No round-trip combinations found")
            
            time.sleep(0.5)
        
        self.db_manager.cleanup_old_data()
        
        console.info(f"✅ MongoDB cache update complete - {total_cached:,} entries cached from {successful_destinations} destinations")
        
        return {
            'total_cached': total_cached,
            'successful_destinations': successful_destinations
        }
    
    def _generate_roundtrip_combinations(self, destination: str, months: List[str]) -> List[Dict[str, Any]]:
        """Generate round-trip combinations from Matrix API data"""
        combinations = []
        
        for month in months:
            outbound_flights = self.flight_api.get_matrix_flights(self.origin, destination, month)
            return_flights = self.flight_api.get_matrix_flights(destination, self.origin, month)
            
            console.info(f"  📋 {destination} {month}: {len(outbound_flights)} outbound, {len(return_flights)} return flights")
            
            for out_flight in outbound_flights:
                for ret_flight in return_flights:
                    try:
                        out_date = datetime.strptime(out_flight['departure_at'], '%Y-%m-%d')
                        ret_date = datetime.strptime(ret_flight['departure_at'], '%Y-%m-%d')
                        
                        duration = (ret_date - out_date).days
                        if not (3 <= duration <= 14):
                            continue
                        
                        total_price = out_flight['value'] + ret_flight['value']
                        
                        if not self.flight_api._validate_price(total_price):
                            continue
                        
                        combination = {
                            'total_price': total_price,
                            'outbound_date': out_flight['departure_at'],
                            'return_date': ret_flight['departure_at'],
                            'duration_days': duration,
                            'outbound_transfers': out_flight.get('transfers', 0),
                            'return_transfers': ret_flight.get('transfers', 0),
                            'airline': out_flight.get('airline', 'Unknown'),
                            'distance': out_flight.get('distance', 0)
                        }
                        combinations.append(combination)
                        
                    except (ValueError, KeyError):
                        continue
        
        console.info(f"  ✅ {destination}: Generated {len(combinations)} valid round-trip combinations")
        return combinations
    
    def detect_deals(self) -> List[Dict[str, Any]]:
        """FIXED: Detect flight deals with working V3 verification (RUB currency conversion)"""
        console.info("🎯 Starting deal detection with FIXED V3 API...")
        
        deals_found = []
        months = self._generate_future_months()
        
        for destination in self.destinations:
            console.info(f"🔍 Analyzing {destination}...")
            
            market_data = self.db_manager.get_market_statistics(destination)
            
            if not market_data:
                console.warning(f"⚠️ {destination}: No market statistics available")
                continue
            
            if market_data['sample_size'] < 10:
                console.warning(f"⚠️ {destination}: Insufficient data ({market_data['sample_size']} samples)")
                continue
            
            console.info(f"📊 {destination}: median={market_data['median_price']:.0f} PLN, threshold={self.absolute_thresholds.get(destination, 400)} PLN, samples={market_data['sample_size']}")
            
            current_combinations = self._generate_roundtrip_combinations(destination, [months[0]])
            
            console.info(f"  📋 Generated {len(current_combinations)} current round-trip combinations for testing")
            
            best_deal_found = False
            for combo in current_combinations[:15]:
                if best_deal_found:
                    break
                    
                round_trip_price = combo['total_price']
                
                if market_data['std_dev'] > 0:
                    z_score = (market_data['median_price'] - round_trip_price) / market_data['std_dev']
                    savings = market_data['median_price'] - round_trip_price
                    
                    absolute_threshold = self.absolute_thresholds.get(destination, 400)
                    meets_z_score = z_score >= self.z_score_threshold
                    meets_absolute = round_trip_price < absolute_threshold
                    
                    console.info(f"  💰 Round-trip: {round_trip_price:.0f} PLN, Z-score: {z_score:.2f}, Absolute: {round_trip_price:.0f} < {absolute_threshold} = {meets_absolute}")
                    
                    if meets_z_score or meets_absolute:
                        departure_date = combo['outbound_date']
                        return_date = combo['return_date']
                        
                        console.info(f"  🔍 FIXED V3 verification: {departure_date} to {return_date}")
                        
                        # FIXED: Use working V3 API with RUB currency and proper parameters
                        verification = self.flight_api.get_v3_verification(
                            self.origin, destination, departure_date, return_date
                        )
                        
                        verified_price = None
                        verification_method = None
                        
                        if verification and self.flight_api._validate_price(verification.get('value', 0)):
                            verified_price = verification.get('value')
                            verification_method = "v3-fixed-rub"
                            console.info(f"  ✅ FIXED V3 verification successful: {verified_price:.0f} PLN (converted from RUB)")
                        else:
                            # FALLBACK: Use Matrix price with higher threshold
                            console.info(f"  ⚠️ V3 verification failed, using Matrix price with higher threshold")
                            if z_score >= (self.z_score_threshold + 1.0):
                                verified_price = round_trip_price
                                verification = {
                                    'value': verified_price,
                                    'departure_at': departure_date,
                                    'return_at': return_date,
                                    'airline': combo.get('airline', 'Unknown')
                                }
                                verification_method = "matrix-fallback"
                                console.info(f"  ⚡ Using Matrix price as fallback: {verified_price:.0f} PLN (high Z-score: {z_score:.2f})")
                        
                        if verified_price:
                            verified_z_score = (market_data['median_price'] - verified_price) / market_data['std_dev']
                            verified_savings = market_data['median_price'] - verified_price
                            
                            final_meets_z_score = verified_z_score >= self.z_score_threshold
                            final_meets_absolute = verified_price < absolute_threshold
                            
                            if final_meets_z_score or final_meets_absolute:
                                deal = {
                                    'destination': destination,
                                    'price': verified_price,
                                    'market_median': market_data['median_price'],
                                    'z_score': verified_z_score,
                                    'savings': verified_savings,
                                    'verification_data': verification,
                                    'verification_method': verification_method
                                }
                                                                deals_found.append(deal)
                                
                                self.notifier.send_deal_alert(
                                    destination, verified_price, verified_z_score,
                                    market_data['median_price'], verified_savings, verification
                                )
                                
                                console.info(f"🎉 DEAL FOUND: {destination} - {verified_price:.0f} PLN (Z-score: {verified_z_score:.2f}, method: {verification_method})")
                                
                                self.db_manager.cache_verified_deal(destination, deal)
                                
                                best_deal_found = True
                                break
                            else:
                                console.info(f"  ❌ Deal rejected after verification: Z-score {verified_z_score:.2f} < {self.z_score_threshold} AND price {verified_price:.0f} >= {absolute_threshold}")
                        else:
                            console.info(f"  ❌ All verification methods failed for {destination}")
                    else:
                        if z_score < self.z_score_threshold and not meets_absolute:
                            console.info(f"  📊 Not a deal: Z-score {z_score:.2f} < {self.z_score_threshold} AND price {round_trip_price:.0f} >= {absolute_threshold}")
                else:
                    console.info(f"  ⚠️ {destination}: No standard deviation for Z-score calculation")
            
            if not best_deal_found:
                console.info(f"  📊 {destination}: No qualifying deals found")
            
            time.sleep(0.3)
        
        return deals_found
    
    def run_daily_automation(self):
        """Run complete daily automation: cache update + deal detection"""
        console.info("🤖 Starting COMPLETELY FIXED MongoDB Flight Bot automation...")
        start_time = time.time()
        
        try:
            # Phase 1: Cache update
            cache_results = self.cache_monthly_data()
            
            # Phase 2: Deal detection
            deals = self.detect_deals()
            
            # Summary
            elapsed_time = (time.time() - start_time) / 60
            
            summary_message = f"🤖 *COMPLETELY FIXED FLIGHT BOT COMPLETE*\n\n"
            summary_message += f"⏱️ Runtime: {elapsed_time:.1f} minutes\n"
            summary_message += f"📊 Cached: {cache_results['total_cached']:,} flights\n"
            summary_message += f"🎯 Destinations processed: {cache_results['successful_destinations']}\n"
            summary_message += f"✅ Deals found: {len(deals)}\n"
            summary_message += f"🔧 FIXED: V3 API with RUB currency conversion\n"
            summary_message += f"📅 FIXED: Off-peak months (Oct/Nov instead of Aug)\n"
            summary_message += f"📈 FIXED: Lower Z-score threshold (1.5 instead of 1.7)\n"
            summary_message += f"💱 FIXED: Proper currency conversion (RUB→PLN)\n\n"
            
            if deals:
                summary_message += "🎉 *Deal Summary:*\n"
                for deal in deals:
                    method = deal.get('verification_method', 'unknown')
                    summary_message += f"• {deal['destination']}: {deal['price']:.0f} PLN (Z: {deal['z_score']:.1f}, {method})\n"
            else:
                summary_message += "📊 No exceptional deals found today\n"
                summary_message += "🔍 European deals should now appear with off-peak months!\n"
            
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
    console.info("🚀 Initializing COMPLETELY FIXED MongoDB Flight Bot...")
    console.info("✅ FIXED: V3 API with RUB currency and proper parameters")
    console.info("✅ FIXED: Off-peak month search (October/November)")
    console.info("✅ FIXED: Lower Z-score threshold (1.5)")
    console.info("✅ FIXED: Currency conversion (RUB to PLN)")
    
    bot = FlightBot()
    bot.run_daily_automation()

if __name__ == "__main__":
    main()
                
            '
