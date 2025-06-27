#!/usr/bin/env python3
"""
Fully Automated Flight Bot - Single Command Version
- Automatically updates 4-month cache AND detects deals in one run
- Z-score 1.7 threshold for ~50 deals/week
- Smart deduplication: price drops allowed, weekly reset
- Perfect for automation with cron jobs
"""

import requests
import time
import logging
import sqlite3
import hashlib
import statistics
import math
import os
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from pathlib import Path

# Logging configuration
logging.basicConfig(
    level=logging.WARNING,
    format='%(levelname)s - %(message)s',
    handlers=[logging.FileHandler('flight_bot.log', encoding='utf-8'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

console = logging.getLogger('console')
console.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(message)s'))
console.addHandler(console_handler)
console.propagate = False

@dataclass
class MatrixEntry:
    """Flight data entry from API"""
    date: str
    price: float
    transfers: int
    airline: str

@dataclass
class RoundTripCandidate:
    """Round-trip combination before verification"""
    destination: str
    outbound_date: str
    return_date: str
    total_price: float
    duration_days: int
    outbound_transfers: int
    return_transfers: int
    outbound_airline: str
    return_airline: str
    estimated_savings_percent: float = 0.0

@dataclass
class VerifiedDeal:
    """Verified flight deal"""
    destination: str
    departure_month: str
    return_month: str
    price: float
    departure_at: str
    return_at: str
    duration_total: int
    outbound_stops: int
    return_stops: int
    airline: str
    booking_link: str
    deal_type: str
    median_price: float
    savings_percent: float
    trip_duration_days: int
    z_score: float = 0.0
    percentile: float = 0.0
    outbound_flight_number: str = ""
    return_flight_number: str = ""
    outbound_duration: int = 0
    return_duration: int = 0
    
    # Consolidated mappings
    _FLAGS = {
        'FCO': '🇮🇹', 'MXP': '🇮🇹', 'LIN': '🇮🇹', 'BGY': '🇮🇹', 'CIA': '🇮🇹', 'VCE': '🇮🇹', 'NAP': '🇮🇹', 'PMO': '🇮🇹',
        'BLQ': '🇮🇹', 'FLR': '🇮🇹', 'PSA': '🇮🇹', 'CAG': '🇮🇹', 'BRI': '🇮🇹', 'CTA': '🇮🇹', 'BUS': '🇮🇹', 'AHO': '🇮🇹', 'GOA': '🇮🇹',
        'MAD': '🇪🇸', 'BCN': '🇪🇸', 'PMI': '🇪🇸', 'IBZ': '🇪🇸', 'VLC': '🇪🇸', 'ALC': '🇪🇸', 'AGP': '🇪🇸', 'BIO': '🇪🇸',
        'LPA': '🇪🇸', 'TFS': '🇪🇸', 'SPC': '🇪🇸', 'MAH': '🇪🇸',
        'LHR': '🇬🇧', 'LTN': '🇬🇧', 'LGW': '🇬🇧', 'STN': '🇬🇧', 'GLA': '🇬🇧', 'BFS': '🇬🇧',
        'CDG': '🇫🇷', 'ORY': '🇫🇷', 'NCE': '🇫🇷', 'MRS': '🇫🇷', 'BIQ': '🇫🇷',
        'FRA': '🇩🇪', 'MUC': '🇩🇪', 'BER': '🇩🇪', 'HAM': '🇩🇪', 'STR': '🇩🇪', 'DUS': '🇩🇪', 'CGN': '🇩🇪', 'LEJ': '🇩🇪', 'DTM': '🇩🇪',
        'AMS': '🇳🇱', 'RTM': '🇳🇱', 'EIN': '🇳🇱',
        'ATH': '🇬🇷', 'SKG': '🇬🇷', 'CFU': '🇬🇷', 'HER': '🇬🇷', 'RHO': '🇬🇷', 'ZTH': '🇬🇷', 'JTR': '🇬🇷', 'CHQ': '🇬🇷'
    }
    
    _CITIES = {
        'WAW': 'Warsaw', 'FCO': 'Rome', 'MAD': 'Madrid', 'BCN': 'Barcelona', 'LHR': 'London', 'AMS': 'Amsterdam',
        'ATH': 'Athens', 'CDG': 'Paris', 'MUC': 'Munich', 'VIE': 'Vienna', 'PRG': 'Prague', 'BRU': 'Brussels',
        'ORY': 'Paris', 'LIN': 'Milan', 'BGY': 'Milan', 'CIA': 'Rome', 'GOA': 'Genoa', 'PMI': 'Palma',
        'MXP': 'Milan', 'VCE': 'Venice', 'NAP': 'Naples', 'LIS': 'Lisbon', 'LTN': 'London', 'LGW': 'London',
        'STN': 'London', 'ARN': 'Stockholm', 'OSL': 'Oslo', 'NYO': 'Stockholm', 'FRA': 'Frankfurt'
    }
    
    _COUNTRIES = {
        'FCO': 'Italy', 'MXP': 'Italy', 'LIN': 'Italy', 'BGY': 'Italy', 'CIA': 'Italy', 'VCE': 'Italy', 
        'NAP': 'Italy', 'GOA': 'Italy', 'PMO': 'Italy', 'BLQ': 'Italy', 'FLR': 'Italy', 'PSA': 'Italy',
        'MAD': 'Spain', 'BCN': 'Spain', 'PMI': 'Spain', 'IBZ': 'Spain', 'VLC': 'Spain', 'ALC': 'Spain',
        'LHR': 'United Kingdom', 'LTN': 'United Kingdom', 'LGW': 'United Kingdom', 'STN': 'United Kingdom',
        'CDG': 'France', 'ORY': 'France', 'NCE': 'France', 'MRS': 'France', 'BIQ': 'France',
        'FRA': 'Germany', 'MUC': 'Germany', 'BER': 'Germany', 'HAM': 'Germany', 'STR': 'Germany',
        'AMS': 'Netherlands', 'RTM': 'Netherlands', 'EIN': 'Netherlands',
        'ATH': 'Greece', 'SKG': 'Greece', 'CFU': 'Greece', 'HER': 'Greece', 'RHO': 'Greece'
    }
    
    def _format_date_range(self, departure_date: str, return_date: str) -> str:
        """Format date range compactly"""
        try:
            dep = datetime.strptime(departure_date, '%Y-%m-%d').strftime('%b %d')
            ret = datetime.strptime(return_date, '%Y-%m-%d')
            ret_fmt = ret.strftime('%d' if departure_date[:7] == return_date[:7] else '%b %d')
            return f"{dep}-{ret_fmt}"
        except Exception:
            return f"{departure_date}-{return_date}"
    
    def _format_flight_type(self) -> str:
        """Format flight connections"""
        out = "Direct" if self.outbound_stops == 0 else f"{self.outbound_stops} stop{'s' if self.outbound_stops > 1 else ''}"
        ret = "Direct" if self.return_stops == 0 else f"{self.return_stops} stop{'s' if self.return_stops > 1 else ''}"
        return "Direct flights" if out == ret == "Direct" else f"Out: {out}, Return: {ret}"
    
    def __str__(self):
        origin = self._CITIES.get('WAW', 'Warsaw')
        dest = self._CITIES.get(self.destination, self.destination)
        country = self._COUNTRIES.get(self.destination, '')
        flag = self._FLAGS.get(self.destination, '')
        
        header = f"*{origin} → {dest}{f', {country} {flag}' if country and flag else ''}: {self.price:.0f} zł*"
        date_range = self._format_date_range(self.departure_at[:10], self.return_at[:10])
        
        return (f"{header}\n\n"
                f"📅 {date_range} ({self.trip_duration_days} days) • {self._format_flight_type()}\n"
                f"📊 {self.savings_percent:.0f}% below typical ({self.median_price:.0f} zł)\n\n"
                f"🔗 [Book Deal]({self.booking_link})")

class FlightCache:
    """4-month rolling cache with optimized operations"""
    
    def __init__(self, cache_dir: str = "flight_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.db_path = self.cache_dir / "flight_cache.db"
        self.CACHE_DAYS = 120
        self._conn = None
        self._init_database()
    
    def _get_connection(self):
        """Reuse database connection for efficiency"""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, timeout=30.0)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA cache_size=10000")
        return self._conn
    
    def _init_database(self):
        """Initialize optimized database schema"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Single execution with all tables and indexes
        cursor.executescript('''
            CREATE TABLE IF NOT EXISTS flight_data (
                id INTEGER PRIMARY KEY,
                destination TEXT NOT NULL,
                outbound_date TEXT NOT NULL,
                return_date TEXT NOT NULL,
                price REAL NOT NULL,
                transfers_out INTEGER,
                transfers_return INTEGER,
                airline TEXT,
                cached_date TEXT NOT NULL,
                trip_duration INTEGER
            );
            
            CREATE TABLE IF NOT EXISTS destination_stats (
                destination TEXT PRIMARY KEY,
                median_price REAL,
                std_dev REAL,
                min_price REAL,
                max_price REAL,
                sample_size INTEGER,
                last_updated TEXT
            );
            
            CREATE TABLE IF NOT EXISTS deal_alerts (
                id INTEGER PRIMARY KEY,
                destination TEXT NOT NULL,
                price REAL NOT NULL,
                z_score REAL NOT NULL,
                alert_date TEXT NOT NULL
            );
            
            CREATE INDEX IF NOT EXISTS idx_destination_date ON flight_data(destination, cached_date);
            CREATE INDEX IF NOT EXISTS idx_cached_date ON flight_data(cached_date);
            CREATE INDEX IF NOT EXISTS idx_alerts_dest_date ON deal_alerts(destination, alert_date);
        ''')
        conn.commit()
    
    def is_cache_current(self) -> bool:
        """Check if cache has been updated today"""
        today = datetime.now().strftime('%Y-%m-%d')
        cursor = self._get_connection().cursor()
        cursor.execute('SELECT COUNT(*) FROM flight_data WHERE cached_date = ?', (today,))
        count = cursor.fetchone()[0]
        return count > 1000  # At least some meaningful data from today
    
    def cache_daily_data(self, api, destinations: List[str], months: List[str]):
        """Optimized daily caching with batch operations"""
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Check if we already cached today
        if self.is_cache_current():
            console.info(f"✅ Cache already current for {today} - skipping cache update")
            return
        
        console.info(f"🗃️ Starting cache update for {len(destinations)} destinations")
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Remove today's data if any exists (partial update)
        cursor.execute('DELETE FROM flight_data WHERE cached_date = ?', (today,))
        
        # Batch insert for better performance
        all_entries = []
        total_cached = 0
        successful_destinations = 0
        
        for i, destination in enumerate(destinations, 1):
            console.info(f"📥 [{i}/{len(destinations)}] Caching {destination}")
            
            try:
                combinations = api.generate_comprehensive_roundtrip_combinations('WAW', destination, months)
                
                if combinations:
                    for combo in combinations:
                        all_entries.append((
                            destination, combo.outbound_date, combo.return_date, combo.total_price,
                            combo.outbound_transfers, combo.return_transfers, combo.outbound_airline,
                            today, combo.duration_days
                        ))
                    
                    successful_destinations += 1
                    console.info(f"  ✅ {destination}: Cached {len(combinations)} combinations")
                else:
                    console.info(f"  ⚠️ {destination}: No valid combinations found")
                
                # Batch insert every 1000 entries for memory efficiency
                if len(all_entries) >= 1000:
                    cursor.executemany('''
                        INSERT INTO flight_data 
                        (destination, outbound_date, return_date, price, transfers_out, 
                         transfers_return, airline, cached_date, trip_duration)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', all_entries)
                    total_cached += len(all_entries)
                    all_entries.clear()
                    conn.commit()  # Commit in batches
                
                # Small delay to be nice to the API
                if i % 10 == 0:
                    time.sleep(1)
                
            except Exception as e:
                console.info(f"  ❌ {destination}: Error - {e}")
                logger.error(f"Cache error for {destination}: {e}")
        
        # Insert remaining entries
        if all_entries:
            cursor.executemany('''
                INSERT INTO flight_data 
                (destination, outbound_date, return_date, price, transfers_out, 
                 transfers_return, airline, cached_date, trip_duration)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', all_entries)
            total_cached += len(all_entries)
        
        # Update statistics for all destinations that have data
        self._update_all_destination_stats(cursor)
        
        # Clean up old data
        self._manage_rolling_window(cursor, today)
        
        conn.commit()
        console.info(f"✅ Cache update complete - {total_cached:,} entries cached from {successful_destinations} destinations")
    
    def _update_all_destination_stats(self, cursor):
        """Update statistics for all destinations with sufficient data"""
        console.info("📊 Updating destination statistics...")
        
        # Get all destinations with data
        cursor.execute('SELECT DISTINCT destination FROM flight_data')
        destinations = [row[0] for row in cursor.fetchall()]
        
        stats_updated = 0
        for destination in destinations:
            cursor.execute('SELECT price FROM flight_data WHERE destination = ?', (destination,))
            prices = [row[0] for row in cursor.fetchall()]
            
            if len(prices) >= 50:  # Need minimum data for reliable stats
                stats = (
                    destination,
                    statistics.median(prices),
                    statistics.stdev(prices) if len(prices) > 1 else 0,
                    min(prices),
                    max(prices),
                    len(prices),
                    datetime.now().strftime('%Y-%m-%d')
                )
                cursor.execute('''
                    INSERT OR REPLACE INTO destination_stats 
                    (destination, median_price, std_dev, min_price, max_price, sample_size, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', stats)
                stats_updated += 1
        
        console.info(f"📊 Updated statistics for {stats_updated} destinations")
    
    def _manage_rolling_window(self, cursor, current_date: str):
        """Efficiently remove old data"""
        cutoff_date = (datetime.strptime(current_date, '%Y-%m-%d') - timedelta(days=self.CACHE_DAYS)).strftime('%Y-%m-%d')
        cursor.execute('DELETE FROM flight_data WHERE cached_date < ?', (cutoff_date,))
        if cursor.rowcount > 0:
            console.info(f"🧹 Removed {cursor.rowcount:,} old entries (keeping 4-month window)")
    
    def get_market_data(self, destination: str) -> Optional[Dict]:
        """Get cached market statistics"""
        cursor = self._get_connection().cursor()
        cursor.execute('SELECT * FROM destination_stats WHERE destination = ?', (destination,))
        stats = cursor.fetchone()
        
        if stats:
            return {
                'destination': stats[0], 'median_price': stats[1], 'std_dev': stats[2],
                'min_price': stats[3], 'max_price': stats[4], 'sample_size': stats[5]
            }
        return None
    
    def get_cache_summary(self) -> Dict:
        """Get cache statistics"""
        cursor = self._get_connection().cursor()
        
        # Total entries
        cursor.execute('SELECT COUNT(*) FROM flight_data')
        total_entries = cursor.fetchone()[0]
        
        # Destinations with stats
        cursor.execute('SELECT COUNT(*) FROM destination_stats WHERE sample_size >= 50')
        ready_destinations = cursor.fetchone()[0]
        
        # Date range
        cursor.execute('SELECT MIN(cached_date), MAX(cached_date) FROM flight_data')
        date_range = cursor.fetchone()
        
        return {
            'total_entries': total_entries,
            'ready_destinations': ready_destinations,
            'date_range': date_range
        }
    
    def __del__(self):
        """Clean up database connection"""
        if self._conn:
            self._conn.close()

class SmartAPI:
    """Optimized API handler with efficient caching"""
    
    PRICE_LIMITS = (200, 6000)
    MAX_PRICE_FILTER = 8000
    
    # Consolidated region mappings
    REGIONS = {
        **{dest: 'europe' for dest in [
            'FCO', 'MAD', 'BCN', 'LHR', 'AMS', 'ATH', 'CDG', 'MUC', 'VIE', 'PRG', 'BRU', 'GVA', 'ARN', 
            'CPH', 'OSL', 'DUB', 'LIS', 'OPO', 'MXP', 'NAP', 'PMI', 'IBZ', 'VLC', 'BUD', 'ZUR', 'FRA',
            'LTN', 'LGW', 'STN', 'NYO', 'ORY', 'LIN', 'BGY', 'CIA', 'VCE', 'OTP', 'HEL', 'PMO', 'KEF'
        ]},
        **{dest: 'middle_east' for dest in ['DXB', 'SHJ', 'AUH', 'RUH', 'SSH', 'JED', 'DMM', 'CAI', 'DOH']},
        **{dest: 'asia' for dest in ['NRT', 'HND', 'KIX', 'ITM', 'ICN', 'GMP', 'PEK', 'BKK', 'DPS', 'HKT']},
        **{dest: 'americas' for dest in ['JFK', 'EWR', 'LGA', 'MIA', 'YYZ', 'YWG', 'YEG', 'HAV', 'PUJ']},
        **{dest: 'africa' for dest in ['TNR', 'RAK', 'DJE', 'ZNZ', 'CMB']}
    }
    
    DURATION_CONSTRAINTS = {
        'europe': (3, 5), 'middle_east': (5, 7), 'asia': (10, 16), 
        'americas': (9, 12), 'africa': (7, 10)
    }
    
    def __init__(self, api_token: str, affiliate_marker: str = "default_marker"):
        self.api_token = api_token
        self.affiliate_marker = affiliate_marker
        self.session = requests.Session()
        self.session.headers.update({'X-Access-Token': api_token})
        self.cache = {}
    
    def get_duration_constraints(self, destination: str) -> Tuple[int, int]:
        """Get trip duration constraints for destination"""
        region = self.REGIONS.get(destination, 'europe')
        return self.DURATION_CONSTRAINTS[region]
    
    def _validate_flight_data(self, price: float, departure_date: str, month: str) -> bool:
        """Validate flight data"""
        if not (0 < price < self.MAX_PRICE_FILTER and departure_date):
            return False
        try:
            flight_date = datetime.strptime(departure_date, '%Y-%m-%d')
            request_month = datetime.strptime(month, '%Y-%m')
            month_diff = abs((flight_date.year - request_month.year) * 12 + (flight_date.month - request_month.month))
            return month_diff <= 3
        except ValueError:
            return False
    
    def _extract_flights(self, origin: str, destination: str, month: str, api_type: str = 'v3') -> List[MatrixEntry]:
        """Extract flights using specified API"""
        if api_type == 'v3':
            url = "https://api.travelpayouts.com/aviasales/v3/prices_for_dates"
            params = {
                'origin': origin, 'destination': destination, 'departure_at': month, 'return_at': month,
                'one_way': False, 'currency': 'PLN', 'sorting': 'price', 'limit': 1000, 'token': self.api_token
            }
        else:  # matrix
            url = "https://api.travelpayouts.com/v2/prices/month-matrix"
            params = {
                'origin': origin, 'destination': destination, 'month': month,
                'currency': 'PLN', 'show_to_affiliates': True, 'token': self.api_token
            }
        
        entries = []
        for page in range(1, 6 if api_type == 'v3' else 2):
            try:
                if api_type == 'v3':
                    params['page'] = page
                
                response = self.session.get(url, params=params, timeout=15)
                if response.status_code != 200:
                    if response.status_code == 429:  # Rate limit
                        time.sleep(2)
                        continue
                    break
                
                data = response.json()
                flights = data.get('data', [])
                if not flights:
                    break
                
                for entry in flights:
                    if api_type == 'v3':
                        price = entry.get('price', 0)
                        date = entry.get('departure_at', '').split('T')[0] if entry.get('departure_at') else ''
                        transfers = entry.get('transfers', 0)
                        airline = entry.get('airline', 'Unknown')
                    else:
                        price = entry.get('value', 0) or entry.get('price', 0)
                        date = entry.get('depart_date', '') or entry.get('departure_at', '')
                        transfers = entry.get('number_of_changes', 0) or entry.get('transfers', 0)
                        airline = entry.get('gate', 'Unknown') or entry.get('airline', 'Unknown')
                    
                    if self._validate_flight_data(price, date, month):
                        entries.append(MatrixEntry(date, price, transfers, airline))
                
                time.sleep(0.2)
                if api_type != 'v3' or len(flights) < 1000:
                    break
                    
            except Exception as e:
                logger.warning(f"API error for {origin}-{destination}: {e}")
                break
        
        return entries
    
    def extract_all_available_flights(self, origin: str, destination: str, month: str) -> List[MatrixEntry]:
        """Extract flights with caching and fallback strategies"""
        cache_key = f"{origin}-{destination}-{month}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Primary: V3 API
        entries = self._extract_flights(origin, destination, month, 'v3')
        
        # Fallback: Matrix API if insufficient data
        if len(entries) < 50:
            matrix_entries = self._extract_flights(origin, destination, month, 'matrix')
            existing = {(e.date, e.price) for e in entries}
            entries.extend([e for e in matrix_entries if (e.date, e.price) not in existing])
        
        self.cache[cache_key] = entries
        return entries
    
    def generate_comprehensive_roundtrip_combinations(self, origin: str, destination: str, months: List[str]) -> List[RoundTripCandidate]:
        """Generate optimized round-trip combinations"""
        min_days, max_days = self.get_duration_constraints(destination)
        
        # Collect all flights efficiently
        outbound = []
        return_flights = []
        
        for month in months:
            try:
                outbound.extend(self.extract_all_available_flights(origin, destination, month))
                return_flights.extend(self.extract_all_available_flights(destination, origin, month))
            except Exception as e:
                logger.warning(f"Error getting flights for {destination} in {month}: {e}")
                continue
        
        if not outbound or not return_flights:
            return []
        
        # Parse dates once for efficiency
        valid_outbound = []
        valid_return = []
        
        for f in outbound:
            if self._validate_date(f.date):
                try:
                    parsed_date = datetime.strptime(f.date, '%Y-%m-%d')
                    valid_outbound.append((f, parsed_date))
                except ValueError:
                    continue
        
        for f in return_flights:
            if self._validate_date(f.date):
                try:
                    parsed_date = datetime.strptime(f.date, '%Y-%m-%d')
                    valid_return.append((f, parsed_date))
                except ValueError:
                    continue
        
        # Generate combinations efficiently
        candidates = []
        for out_flight, out_date in valid_outbound[:500]:  # Limit for performance
            for ret_flight, ret_date in valid_return[:500]:
                duration = (ret_date - out_date).days
                if min_days <= duration <= max_days:
                    total_price = out_flight.price + ret_flight.price
                    if self.PRICE_LIMITS[0] <= total_price <= self.PRICE_LIMITS[1]:
                        candidates.append(RoundTripCandidate(
                            destination, out_flight.date, ret_flight.date, total_price, duration,
                            out_flight.transfers, ret_flight.transfers, out_flight.airline, ret_flight.airline
                        ))
                        if len(candidates) >= 10000:  # Performance limit
                            return candidates
        
        return candidates
    
    def _validate_date(self, date_str: str) -> bool:
        """Quick date validation"""
        try:
            datetime.strptime(date_str, '%Y-%m-%d')
            return True
        except ValueError:
            return False
    
    def get_v3_verification(self, origin: str, destination: str, departure_date: str, return_date: str) -> Optional[Dict]:
        """Verify deal with V3 API"""
        url = "https://api.travelpayouts.com/aviasales/v3/prices_for_dates"
        params = {
            'origin': origin, 'destination': destination, 'departure_at': departure_date,
            'return_at': return_date, 'one_way': False, 'currency': 'PLN', 'sorting': 'price',
            'limit': 1, 'token': self.api_token
        }
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                flights = data.get('data', [])
                return flights[0] if flights else None
            elif response.status_code == 429:
                time.sleep(1)
                return None
        except Exception as e:
            logger.warning(f"V3 verification error: {e}")
        return None

class FastTelegram:
    """Optimized Telegram sender"""
    
    def __init__(self, bot_token: str, chat_id: str):
        self.url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        self.chat_id = chat_id
    
    def send(self, message: str) -> bool:
        """Send message with error handling"""
        try:
            response = requests.post(self.url, json={
                'chat_id': self.chat_id, 'text': message, 'parse_mode': 'Markdown'
            }, timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Telegram error: {e}")
            return False

class AutomatedFlightBot:
    """Fully automated bot that updates cache AND detects deals in one run"""
    
    # Class constants for better memory usage
    Z_THRESHOLDS = {'exceptional': 2.5, 'excellent': 2.0, 'great': 1.7, 'minimum': 1.7}
    WEEKLY_RESET_DAYS = 7
    PRICE_IMPROVEMENT_THRESHOLD = 0.05
    
    # Consolidated destinations list
    DESTINATIONS = [
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
        'DWC', 'DOH', 'JED', 'DMM', 'BOO', 'FRU'
    ]
    
    def __init__(self, api_token: str, affiliate_marker: str, telegram_token: str, telegram_chat_id: str):
        self.api = SmartAPI(api_token, affiliate_marker)
        self.telegram = FastTelegram(telegram_token, telegram_chat_id)
        self.cache = FlightCache()
        self.start_time = None
        self.total_start_time = None
    
    @staticmethod
    def _generate_future_months(start_month: int = 8, start_year: int = 2025, count: int = 6) -> List[str]:
        """Generate future months efficiently"""
        months = []
        for i in range(count):
            month = start_month + i
            year = start_year + (month - 1) // 12
            month = ((month - 1) % 12) + 1
            months.append(f"{year:04d}-{month:02d}")
        return months
    
    def should_alert_destination(self, destination: str, current_price: float, z_score: float) -> bool:
        """Smart alerting logic with optimized database access"""
        if z_score < self.Z_THRESHOLDS['minimum']:
            return False
        
        cursor = self.cache._get_connection().cursor()
        cursor.execute('''
            SELECT price, alert_date FROM deal_alerts 
            WHERE destination = ? ORDER BY alert_date DESC LIMIT 1
        ''', (destination,))
        
        recent_alert = cursor.fetchone()
        if not recent_alert:
            return True
        
        last_price, last_date = recent_alert
        try:
            days_since = (datetime.now() - datetime.strptime(last_date, '%Y-%m-%d')).days
        except ValueError:
            return True  # Invalid date format, allow alert
        
        return (days_since >= self.WEEKLY_RESET_DAYS or 
                (last_price - current_price) / last_price > self.PRICE_IMPROVEMENT_THRESHOLD)
    
    def log_deal_alert(self, deal: VerifiedDeal):
        """Log deal efficiently"""
        cursor = self.cache._get_connection().cursor()
        cursor.execute('''
            INSERT INTO deal_alerts (destination, price, z_score, alert_date)
            VALUES (?, ?, ?, ?)
        ''', (deal.destination, deal.price, deal.z_score, datetime.now().strftime('%Y-%m-%d')))
        cursor.connection.commit()
    
    def classify_deal_with_zscore(self, price: float, market_data: Dict) -> Tuple[str, float, float, float]:
        """Optimized deal classification"""
        if market_data['std_dev'] <= 0:
            return "Unknown Deal", 0.0, 0.0, 0.0
        
        median_price = market_data['median_price']
        z_score = (median_price - price) / market_data['std_dev']
        savings_percent = ((median_price - price) / median_price) * 100
        
        try:
            percentile = 50 + 50 * math.erf(z_score / math.sqrt(2)) if z_score >= 0 else 50 - 50 * math.erf(abs(z_score) / math.sqrt(2))
        except (OverflowError, ValueError):
            percentile = 99.9 if z_score > 0 else 0.1
        
        # Simplified classification
        if z_score >= self.Z_THRESHOLDS['exceptional']:
            return "🔥 Exceptional Deal", z_score, savings_percent, percentile
        elif z_score >= self.Z_THRESHOLDS['excellent']:
            return "💎 Excellent Deal", z_score, savings_percent, percentile
        elif z_score >= self.Z_THRESHOLDS['great']:
            return "💰 Great Deal", z_score, savings_percent, percentile
        else:
            return "📊 Fair Price", z_score, savings_percent, percentile
    
    def _create_booking_link(self, candidate: RoundTripCandidate, v3_result: Dict) -> str:
        """Create optimized booking link"""
        link = v3_result.get('link', '')
        if link:
            return f"https://www.aviasales.com{link}"
        else:
            return (f"https://www.aviasales.com/search/WAW{candidate.outbound_date}"
                   f"{candidate.destination}{candidate.return_date}?marker={self.api.affiliate_marker}")
    
    def find_and_verify_deals_for_destination(self, destination: str, market_data: Dict, months: List[str]) -> List[VerifiedDeal]:
        """Optimized deal finding and verification"""
        console.info(f"  🔍 Searching for deals in {destination}")
        
        try:
            candidates = self.api.generate_comprehensive_roundtrip_combinations('WAW', destination, months)
        except Exception as e:
            console.info(f"  ❌ Error generating combinations for {destination}: {e}")
            return []
        
        if not candidates:
            console.info(f"  📊 {destination}: No valid combinations found")
            return []
        
        # Efficient sorting and filtering
        for candidate in candidates:
            if market_data['std_dev'] > 0:
                candidate.estimated_savings_percent = ((market_data['median_price'] - candidate.total_price) / 
                                                      market_data['std_dev'])
            else:
                candidate.estimated_savings_percent = 0
        
        # Take top candidates efficiently
        top_candidates = sorted(candidates, key=lambda x: x.estimated_savings_percent, reverse=True)[:10]
        console.info(f"  📋 Verifying top {len(top_candidates)} candidates from {len(candidates):,} combinations")
        
        best_deal = None
        best_z_score = 0
        
        for candidate in top_candidates:
            if candidate.estimated_savings_percent < 1.0:
                continue
            
            try:
                v3_result = self.api.get_v3_verification('WAW', destination, candidate.outbound_date, candidate.return_date)
            except Exception as e:
                logger.warning(f"V3 verification error for {destination}: {e}")
                continue
            
            if v3_result:
                actual_price = v3_result.get('price', 0)
                if actual_price <= 0:
                    continue
                
                deal_type, z_score, savings_percent, percentile = self.classify_deal_with_zscore(actual_price, market_data)
                
                if (z_score >= self.Z_THRESHOLDS['minimum'] and z_score > best_z_score and
                    self.should_alert_destination(destination, actual_price, z_score)):
                    
                    best_z_score = z_score
                    
                    # Extract date information safely
                    departure_at = v3_result.get('departure_at', candidate.outbound_date)
                    return_at = v3_result.get('return_at', candidate.return_date)
                    
                    # Handle datetime strings
                    if 'T' in departure_at:
                        departure_at = departure_at.split('T')[0]
                    if 'T' in return_at:
                        return_at = return_at.split('T')[0]
                    
                    best_deal = VerifiedDeal(
                        destination=destination,
                        departure_month=months[0],
                        return_month=months[0],
                        price=actual_price,
                        departure_at=departure_at,
                        return_at=return_at,
                        duration_total=v3_result.get('duration', 0),
                        outbound_stops=v3_result.get('transfers', candidate.outbound_transfers),
                        return_stops=v3_result.get('return_transfers', candidate.return_transfers),
                        airline=v3_result.get('airline', candidate.outbound_airline),
                        booking_link=self._create_booking_link(candidate, v3_result),
                        deal_type=deal_type,
                        median_price=market_data['median_price'],
                        savings_percent=savings_percent,
                        trip_duration_days=candidate.duration_days,
                        z_score=z_score,
                        percentile=percentile,
                        outbound_flight_number=v3_result.get('flight_number', ''),
                        return_flight_number=v3_result.get('return_flight_number', ''),
                        outbound_duration=v3_result.get('outbound_duration', 0),
                        return_duration=v3_result.get('return_duration', 0)
                    )
                    
                    console.info(f"  🏆 SMART DEAL: {actual_price:.0f} zł (Z-score: {z_score:.1f})")
            
            time.sleep(0.3)
        
        return [best_deal] if best_deal else []
    
    def send_immediate_deal_alert(self, deal: VerifiedDeal, deal_number: int, elapsed_minutes: float):
        """Send optimized alert"""
        success = self.telegram.send(str(deal))
        if success:
            self.log_deal_alert(deal)
            console.info(f"📱 Alert #{deal_number} for {deal.destination} - {deal.price:.0f} zł")
        else:
            console.info(f"⚠️ Failed to send alert for {deal.destination}")
    
    def update_cache_and_detect_deals(self):
        """Main automated method: updates cache AND detects deals"""
        self.total_start_time = time.time()
        
        console.info("🤖 FULLY AUTOMATED FLIGHT BOT STARTED")
        console.info("=" * 50)
        
        months = self._generate_future_months()
        
        # Send startup notification
        startup_msg = (f"🤖 *AUTOMATED FLIGHT BOT STARTED*\n\n"
                      f"🔄 Phase 1: Cache Update\n"
                      f"🎯 Phase 2: Deal Detection\n"
                      f"📅 Months: {', '.join(months)}\n\n"
                      f"⚡ Z-score ≥1.7 | Smart deduplication active")
        
        if not self.telegram.send(startup_msg):
            console.info("⚠️ Failed to send startup notification")
        
        # PHASE 1: UPDATE CACHE
        console.info("\n🔄 PHASE 1: CACHE UPDATE")
        console.info("=" * 30)
        
        cache_start = time.time()
        try:
            self.cache.cache_daily_data(self.api, self.DESTINATIONS, months)
            cache_time = (time.time() - cache_start) / 60
            
            # Get cache summary
            cache_summary = self.cache.get_cache_summary()
            
            console.info(f"✅ Cache update completed in {cache_time:.1f} minutes")
            console.info(f"📊 Cache summary: {cache_summary['total_entries']:,} entries, {cache_summary['ready_destinations']} destinations ready")
            
            # Send cache update notification
            cache_msg = (f"✅ *CACHE UPDATE COMPLETE*\n\n"
                        f"⏱️ Time: {cache_time:.1f} minutes\n"
                        f"📊 Entries: {cache_summary['total_entries']:,}\n"
                        f"🎯 Ready destinations: {cache_summary['ready_destinations']}\n\n"
                        f"🚀 Starting deal detection...")
            
            self.telegram.send(cache_msg)
            
        except Exception as e:
            error_msg = f"❌ Cache update failed: {e}"
            console.info(error_msg)
            self.telegram.send(error_msg)
            return []
        
        # PHASE 2: DEAL DETECTION
        console.info("\n🎯 PHASE 2: DEAL DETECTION")
        console.info("=" * 30)
        
        self.start_time = time.time()
        all_deals = []
        deals_found = 0
        
        for i, destination in enumerate(self.DESTINATIONS, 1):
            elapsed_time = time.time() - self.start_time
            console.info(f"🎯 [{i}/{len(self.DESTINATIONS)}] Processing {destination} ({elapsed_time/60:.1f}min elapsed)")
            
            try:
                market_data = self.cache.get_market_data(destination)
                
                if market_data and market_data['sample_size'] >= 50:
                    console.info(f"  ✅ {destination}: {market_data['sample_size']} samples, median: {market_data['median_price']:.0f} zł")
                    
                    verified_deals = self.find_and_verify_deals_for_destination(destination, market_data, months)
                    
                    if verified_deals:
                        deals_found += len(verified_deals)
                        for deal in verified_deals:
                            all_deals.append(deal)
                            self.send_immediate_deal_alert(deal, deals_found, elapsed_time/60)
                    else:
                        console.info(f"  📊 {destination}: No deals passed smart filter")
                else:
                    sample_size = market_data['sample_size'] if market_data else 0
                    console.info(f"  ⚠️ {destination}: Insufficient cached data ({sample_size} samples)")
                
                # Progress update every 25 destinations
                if i % 25 == 0:
                    progress_time = time.time() - self.start_time
                    console.info(f"🔄 Progress: {i}/{len(self.DESTINATIONS)} ({(i/len(self.DESTINATIONS))*100:.1f}%) - {deals_found} deals found - {progress_time/60:.1f}min elapsed")
            
            except Exception as e:
                console.info(f"  ❌ Error processing {destination}: {e}")
                logger.error(f"Error processing {destination}: {e}")
        
        return all_deals
    
    def send_final_summary(self, deals: List[VerifiedDeal]):
        """Send comprehensive final summary"""
        total_time = (time.time() - self.total_start_time) / 60
        detection_time = (time.time() - self.start_time) / 60
        cache_time = total_time - detection_time
        
        if not deals:
            summary = (f"🤖 *AUTOMATED BOT COMPLETE*\n\n"
                      f"⏱️ Total runtime: {total_time:.1f} minutes\n"
                      f"🔄 Cache update: {cache_time:.1f} min\n"
                      f"🎯 Deal detection: {detection_time:.1f} min\n\n"
                      f"🔍 Processed {len(self.DESTINATIONS)} destinations\n"
                      f"❌ No deals found (Z-score ≥ {self.Z_THRESHOLDS['minimum']} required)\n\n"
                      f"🔄 Next run: Tomorrow (automated)")
            
            self.telegram.send(summary)
            return
        
        # Efficient categorization
        exceptional = sum(1 for d in deals if d.z_score >= self.Z_THRESHOLDS['exceptional'])
        excellent = sum(1 for d in deals if self.Z_THRESHOLDS['excellent'] <= d.z_score < self.Z_THRESHOLDS['exceptional'])
        great = sum(1 for d in deals if self.Z_THRESHOLDS['great'] <= d.z_score < self.Z_THRESHOLDS['excellent'])
        
        # Calculate savings
        total_savings = sum(d.savings_percent for d in deals)
        avg_savings = total_savings / len(deals) if deals else 0
        
        summary = (f"🤖 *AUTOMATED BOT COMPLETE*\n\n"
                  f"⏱️ Total runtime: {total_time:.1f} minutes\n"
                  f"🔄 Cache update: {cache_time:.1f} min\n"
                  f"🎯 Deal detection: {detection_time:.1f} min\n\n"
                  f"✅ **{len(deals)} DEALS FOUND**\n"
                  f"🔥 {exceptional} exceptional (Z≥{self.Z_THRESHOLDS['exceptional']})\n"
                  f"💎 {excellent} excellent (Z≥{self.Z_THRESHOLDS['excellent']})\n"
                  f"💰 {great} great (Z≥{self.Z_THRESHOLDS['great']})\n\n"
                  f"📊 Average savings: {avg_savings:.0f}%\n"
                  f"🎯 Smart deduplication active\n"
                  f"🗃️ 4-month cache analysis\n\n"
                  f"🔄 Next run: Tomorrow (automated)")
        
        self.telegram.send(summary)
        console.info(f"📱 Sent final summary - {len(deals)} deals in {total_time:.1f} minutes")
    
    def run(self):
        """Single command that does EVERYTHING"""
        try:
            # Clean up old alerts first
            cursor = self.cache._get_connection().cursor()
            cutoff_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            cursor.execute('DELETE FROM deal_alerts WHERE alert_date < ?', (cutoff_date,))
            if cursor.rowcount > 0:
                console.info(f"🧹 Cleaned up {cursor.rowcount} old alerts")
            cursor.connection.commit()
            
            # Main automation: cache update + deal detection
            deals = self.update_cache_and_detect_deals()
            
            # Summary
            total_time = (time.time() - self.total_start_time) / 60
            console.info(f"\n🤖 AUTOMATED BOT COMPLETE")
            console.info(f"⏱️ Total time: {total_time:.1f} minutes")
            console.info(f"🎉 Found {len(deals)} deals")
            
            self.send_final_summary(deals)
            
        except Exception as e:
            error_msg = f"\n❌ Bot error: {str(e)}"
            console.info(error_msg)
            logger.error(f"Bot error: {e}")
            self.telegram.send(f"❌ Automated bot error: {str(e)}")

def main():
    """Simplified main function for automation"""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    # Get environment variables efficiently
    env_vars = {
        'API_TOKEN': os.getenv('TRAVELPAYOUTS_API_TOKEN'),
        'AFFILIATE_MARKER': os.getenv('TRAVELPAYOUTS_AFFILIATE_MARKER', 'default_marker'),
        'TELEGRAM_TOKEN': os.getenv('TELEGRAM_BOT_TOKEN'),
        'TELEGRAM_CHAT_ID': os.getenv('TELEGRAM_CHAT_ID')
    }
    
    missing = [k for k, v in env_vars.items() if not v and k != 'AFFILIATE_MARKER']
    if missing:
        console.info(f"❌ Missing environment variables: {', '.join(missing)}")
        return
    
    bot = AutomatedFlightBot(
        env_vars['API_TOKEN'], env_vars['AFFILIATE_MARKER'],
        env_vars['TELEGRAM_TOKEN'], env_vars['TELEGRAM_CHAT_ID']
    )
    
    # Handle command line arguments for flexibility
    import sys
    command = sys.argv[1] if len(sys.argv) > 1 else None
    
    if command == '--cache-only':
        # Just update cache (for testing)
        months = bot._generate_future_months()
        bot.cache.cache_daily_data(bot.api, bot.DESTINATIONS, months)
    elif command == '--detect-only':
        # Just detect deals (for testing)
        months = bot._generate_future_months()
        bot.start_time = time.time()
        deals = []
        for destination in bot.DESTINATIONS[:10]:  # Test with first 10
            market_data = bot.cache.get_market_data(destination)
            if market_data:
                deals.extend(bot.find_and_verify_deals_for_destination(destination, market_data, months))
        console.info(f"Found {len(deals)} deals in test")
    else:
        # DEFAULT: Full automation (cache + detection)
        bot.run()

if __name__ == "__main__":
    main()