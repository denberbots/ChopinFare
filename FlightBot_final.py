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

class MongoFlightBot:
    """CORRECTED MongoDB-powered flight bot with data quality validation"""
    
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
    
    def __init__(self, api_token: str, affiliate_marker: str, telegram_token: str, telegram_chat_id: str, mongodb_connection: str):
        self.api = SmartAPI(api_token, affiliate_marker)
        self.telegram = FastTelegram(telegram_token, telegram_chat_id)
        self.cache = MongoFlightCache(mongodb_connection)
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
        """Smart alerting logic with MongoDB access - works for both Z-score and absolute deals"""
        recent_alert = self.cache.get_recent_alert(destination)
        if not recent_alert:
            return True
        
        last_price = recent_alert['price']
        last_date = recent_alert['alert_date']
        
        try:
            days_since = (datetime.now() - datetime.strptime(last_date, '%Y-%m-%d')).days
        except ValueError:
            return True  # Invalid date format, allow alert
        
        return (days_since >= self.WEEKLY_RESET_DAYS or 
                (last_price - current_price) / last_price > self.PRICE_IMPROVEMENT_THRESHOLD)
    
    def classify_deal_with_zscore(self, price: float, destination: str, market_data: Dict) -> Tuple[str, float, float, float, bool]:
        """Deal classification with Z-score AND absolute thresholds - COMBINED LOGIC"""
        z_score = 0.0
        savings_percent = 0.0
        percentile = 50.0
        
        # Calculate Z-score if we have market data
        if market_data and market_data['std_dev'] > 0:
            median_price = market_data['median_price']
            z_score = (median_price - price) / market_data['std_dev']
            savings_percent = ((median_price - price) / median_price) * 100
            
            try:
                percentile = 50 + 50 * math.erf(z_score / math.sqrt(2)) if z_score >= 0 else 50 - 50 * math.erf(abs(z_score) / math.sqrt(2))
            except (OverflowError, ValueError):
                percentile = 99.9 if z_score > 0 else 0.1
        
        # Check absolute threshold
        absolute_threshold = self.api.get_absolute_threshold(destination)
        is_absolute_deal = price < absolute_threshold
        
        # COMBINED LOGIC: Z-score OR absolute threshold qualifies as deal
        if z_score >= self.Z_THRESHOLDS['exceptional'] or (is_absolute_deal and price < absolute_threshold * 0.8):
            return "🔥 Exceptional Deal", z_score, savings_percent, percentile, True
        elif z_score >= self.Z_THRESHOLDS['excellent'] or (is_absolute_deal and price < absolute_threshold * 0.9):
            return "💎 Excellent Deal", z_score, savings_percent, percentile, True
        elif z_score >= self.Z_THRESHOLDS['great'] or is_absolute_deal:
            return "💰 Great Deal", z_score, savings_percent, percentile, True
        else:
            return "📊 Fair Price", z_score, savings_percent, percentile, False
    
    def _create_booking_link(self, candidate: RoundTripCandidate, v3_result: Dict) -> str:
        """Create optimized booking link"""
        link = v3_result.get('link', '')
        if link:
            return f"https://www.aviasales.com{link}"
        else:
            return (f"https://www.aviasales.com/search/WAW{candidate.outbound_date}"
                   f"{candidate.destination}{candidate.return_date}?marker={self.api.affiliate_marker}")
    
    def find_and_verify_deals_for_destination(self, destination: str, market_data: Dict, months: List[str]) -> List[VerifiedDeal]:
        """Find and verify deals - MAXIMUM 1 DEAL PER DESTINATION"""
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
                
                deal_type, z_score, savings_percent, percentile, is_deal = self.classify_deal_with_zscore(actual_price, destination, market_data)
                
                if (is_deal and z_score > best_z_score and
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
                    
                    console.info(f"  🏆 CORRECTED DEAL: {actual_price:.0f} zł (Z-score: {z_score:.1f}, Threshold: {self.api.get_absolute_threshold(destination)})")
            
            time.sleep(0.3)
        
        return [best_deal] if best_deal else []
    
    def send_immediate_deal_alert(self, deal: VerifiedDeal, deal_number: int, elapsed_minutes: float):
        """Send optimized alert"""
        success = self.telegram.send(str(deal))
        if success:
            self.cache.log_deal_alert(deal)
            console.info(f"📱 Alert #{deal_number} for {deal.destination} - {deal.price:.0f} zł")
        else:
            console.info(f"⚠️ Failed to send alert for {deal.destination}")
    
    def update_cache_and_detect_deals(self):
        """Main automated method: CORRECTED MongoDB cache + deal detection"""
        self.total_start_time = time.time()
        
        console.info("🤖 CORRECTED MONGODB FLIGHT BOT STARTED")
        console.info("=" * 60)
        
        months = self._generate_future_months()
        
        # Send startup notification
        startup_msg = (f"🤖 *CORRECTED FLIGHT BOT STARTED*\n\n"
                      f"🔧 FIXES: Enhanced data validation, economy-only filtering\n"
                      f"🗃️ Phase 1: MongoDB Cache Update (45-day window)\n"
                      f"⚡ ALWAYS performs full daily update\n"
                      f"🎯 Phase 2: Deal Detection\n"
                      f"📅 Months: {', '.join(months)}\n\n"
                      f"⚡ Z-score ≥1.7 OR Absolute thresholds | Data quality validation\n"
                      f"☁️ Persistent MongoDB Atlas cache (1.5 months)")
        
        if not self.telegram.send(startup_msg):
            console.info("⚠️ Failed to send startup notification")
        
        # PHASE 1: UPDATE MONGODB CACHE (CORRECTED)
        console.info("\n🗃️ PHASE 1: CORRECTED MONGODB CACHE UPDATE")
        console.info("=" * 50)
        
        cache_start = time.time()
        try:
            # CORRECTED cache update with validation
            self.cache.cache_daily_data(self.api, self.DESTINATIONS, months)
            cache_time = (time.time() - cache_start) / 60
            
            # Get cache summary
            cache_summary = self.cache.get_cache_summary()
            
            console.info(f"✅ CORRECTED MongoDB cache update completed in {cache_time:.1f} minutes")
            console.info(f"📊 Cache summary: {cache_summary['total_entries']:,} entries, {cache_summary['ready_destinations']} destinations ready")
            
            # Send cache update notification
            cache_msg = (f"✅ *CORRECTED CACHE UPDATE COMPLETE*\n\n"
                        f"⏱️ Time: {cache_time:.1f} minutes\n"
                        f"📊 Total entries: {cache_summary['total_entries']:,}\n"
                        f"🎯 Ready destinations: {cache_summary['ready_destinations']}\n"
                        f"🔧 Enhanced data validation applied\n"
                        f"💎 Economy class filtering active\n"
                        f"🗃️ 45-day rolling window (optimized for 512 MB)\n"
                        f"⚡ FULL daily update performed\n"
                        f"☁️ Persistent cloud storage\n\n"
                        f"🚀 Starting deal detection...")
            
            self.telegram.send(cache_msg)
            
        except Exception as e:
            error_msg = f"❌ CORRECTED MongoDB cache update failed: {e}"
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
                    console.info(f"  ✅ {destination}: {market_data['sample_size']} samples, median: {market_data['median_price']:.0f} zł, threshold: {self.api.get_absolute_threshold(destination)} zł")
                    
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
        
        # Get cache stats
        cache_summary = self.cache.get_cache_summary()
        
        if not deals:
            summary = (f"🤖 *CORRECTED FLIGHT BOT COMPLETE*\n\n"
                      f"⏱️ Total runtime: {total_time:.1f} minutes\n"
                      f"🗃️ MongoDB cache: {cache_time:.1f} min (CORRECTED UPDATE)\n"
                      f"🎯 Deal detection: {detection_time:.1f} min\n\n"
                      f"📊 Database: {cache_summary['total_entries']:,} entries\n"
                      f"🔍 Processed {len(self.DESTINATIONS)} destinations\n"
                      f"❌ No deals found (Z-score ≥ {self.Z_THRESHOLDS['minimum']} OR absolute thresholds required)\n\n"
                      f"🔧 Data quality improvements applied\n"
                      f"💎 Economy class filtering active\n"
                      f"🗃️ 45-day rolling cache (optimized)\n"
                      f"⚡ ALWAYS updates cache - no skipping\n"
                      f"☁️ Persistent MongoDB Atlas storage\n"
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
        
        summary = (f"🤖 *CORRECTED FLIGHT BOT COMPLETE*\n\n"
                  f"⏱️ Total runtime: {total_time:.1f} minutes\n"
                  f"🗃️ MongoDB cache: {cache_time:.1f} min (CORRECTED UPDATE)\n"
                  f"🎯 Deal detection: {detection_time:.1f} min\n\n"
                  f"✅ **{len(deals)} QUALITY DEALS FOUND**\n"
                  f"🔥 {exceptional} exceptional (Z≥{self.Z_THRESHOLDS['exceptional']})\n"
                  f"💎 {excellent} excellent (Z≥{self.Z_THRESHOLDS['excellent']})\n"
                  f"💰 {great} great (Z≥{self.Z_THRESHOLDS['great']})\n\n"
                  f"📊 Average savings: {avg_savings:.0f}%\n"
                  f"🗃️ Database: {cache_summary['total_entries']:,} entries (45-day window)\n"
                  f"🎯 Smart deduplication active (max 1 deal per destination)\n"
                  f"🔧 Enhanced data validation applied\n"
                  f"💎 Economy class filtering active\n"
                  f"⚡ ALWAYS updates cache - no skipping\n"
                  f"☁️ Persistent MongoDB Atlas cache\n\n"
                  f"🔄 Next run: Tomorrow (automated)")
        
        self.telegram.send(summary)
        console.info(f"📱 Sent final summary - {len(deals)} quality deals in {total_time:.1f} minutes")
    
    def run(self):
        """Single command that does EVERYTHING with CORRECTED MongoDB"""
        try:
            # Clean up old alerts first
            self.cache.cleanup_old_alerts()
            
            # Main automation: CORRECTED MongoDB cache update + deal detection
            deals = self.update_cache_and_detect_deals()
            
            # Summary
            total_time = (time.time() - self.total_start_time) / 60
            console.info(f"\n🤖 CORRECTED MONGODB FLIGHT BOT COMPLETE")
            console.info(f"⏱️ Total time: {total_time:.1f} minutes")
            console.info(f"🎉 Found {len(deals)} quality deals")
            console.info(f"🔧 Data quality improvements applied")
            console.info(f"💎 Economy class filtering active")
            console.info(f"🗃️ 45-day cache with CORRECTED daily updates")
            console.info(f"⚡ No cache skipping - always updates")
            console.info(f"☁️ Persistent storage maintained")
            
            self.send_final_summary(deals)
            
        except Exception as e:
            error_msg = f"\n❌ Bot error: {str(e)}"
            console.info(error_msg)
            logger.error(f"Bot error: {e}")
            self.telegram.send(f"❌ CORRECTED MongoDB bot error: {str(e)}")

def main():
    """Main function for CORRECTED MongoDB-powered automation"""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    # Get environment variables
    env_vars = {
        'API_TOKEN': os.getenv('TRAVELPAYOUTS_API_TOKEN'),
        'AFFILIATE_MARKER': os.getenv('TRAVELPAYOUTS_AFFILIATE_MARKER', 'default_marker'),
        'TELEGRAM_TOKEN': os.getenv('TELEGRAM_BOT_TOKEN'),
        'TELEGRAM_CHAT_ID': os.getenv('TELEGRAM_CHAT_ID'),
        'MONGODB_CONNECTION': os.getenv('MONGODB_CONNECTION_STRING')
    }
    
    missing = [k for k, v in env_vars.items() if not v and k != 'AFFILIATE_MARKER']
    if missing:
        console.info(f"❌ Missing environment variables: {', '.join(missing)}")
        return
    
    bot = MongoFlightBot(
        env_vars['API_TOKEN'], env_vars['AFFILIATE_MARKER'],
        env_vars['TELEGRAM_TOKEN'], env_vars['TELEGRAM_CHAT_ID'],
        env_vars['MONGODB_CONNECTION']
    )
    
    # Handle command line arguments for flexibility
    import sys
    command = sys.argv[1] if len(sys.argv) > 1 else None
    
    if command == '--cache-only':
        # Just update MongoDB cache (for testing)
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
        # DEFAULT: Full automation (CORRECTED MongoDB cache + detection)
        bot.run()

if __name__ == "__main__":
    main()#!/usr/bin/env python3
"""
MongoDB Flight Bot - CORRECTED Data Collection Version
✅ Fixed price validation - stricter economy class filtering
✅ Enhanced API parameters - economy only, proper round-trip validation
✅ Robust currency and data type validation
✅ Cache corruption prevention and cleanup
✅ Debug logging for data quality monitoring

FIXES: Prevents 750+ PLN minimums, business class mixing, currency errors
"""

import requests
import time
import logging
import statistics
import math
import os
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure

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
    
    # COMPLETE mappings - FIXED Lisbon and all formatting issues
    _FLAGS = {
        'FCO': '🇮🇹', 'MXP': '🇮🇹', 'LIN': '🇮🇹', 'BGY': '🇮🇹', 'CIA': '🇮🇹', 'VCE': '🇮🇹', 'NAP': '🇮🇹', 'PMO': '🇮🇹',
        'BLQ': '🇮🇹', 'FLR': '🇮🇹', 'PSA': '🇮🇹', 'CAG': '🇮🇹', 'BRI': '🇮🇹', 'CTA': '🇮🇹', 'BUS': '🇮🇹', 'AHO': '🇮🇹', 'GOA': '🇮🇹',
        'MAD': '🇪🇸', 'BCN': '🇪🇸', 'PMI': '🇪🇸', 'IBZ': '🇪🇸', 'VLC': '🇪🇸', 'ALC': '🇪🇸', 'AGP': '🇪🇸', 'BIO': '🇪🇸',
        'LPA': '🇪🇸', 'TFS': '🇪🇸', 'SPC': '🇪🇸', 'MAH': '🇪🇸',
        'LHR': '🇬🇧', 'LTN': '🇬🇧', 'LGW': '🇬🇧', 'STN': '🇬🇧', 'GLA': '🇬🇧', 'BFS': '🇬🇧',
        'CDG': '🇫🇷', 'ORY': '🇫🇷', 'NCE': '🇫🇷', 'MRS': '🇫🇷', 'BIQ': '🇫🇷', 'PIS': '🇫🇷', 'PUY': '🇫🇷',
        'FRA': '🇩🇪', 'MUC': '🇩🇪', 'BER': '🇩🇪', 'HAM': '🇩🇪', 'STR': '🇩🇪', 'DUS': '🇩🇪', 'CGN': '🇩🇪', 'LEJ': '🇩🇪', 'DTM': '🇩🇪',
        'AMS': '🇳🇱', 'RTM': '🇳🇱', 'EIN': '🇳🇱',
        'ATH': '🇬🇷', 'SKG': '🇬🇷', 'CFU': '🇬🇷', 'HER': '🇬🇷', 'RHO': '🇬🇷', 'ZTH': '🇬🇷', 'JTR': '🇬🇷', 'CHQ': '🇬🇷',
        'LIS': '🇵🇹', 'OPO': '🇵🇹', 'PDL': '🇵🇹', 'PXO': '🇵🇹',  # Portugal - FIXED: LIS included
        'ARN': '🇸🇪', 'NYO': '🇸🇪', 'OSL': '🇳🇴', 'BGO': '🇳🇴', 'BOO': '🇳🇴',  # Scandinavia
        'HEL': '🇫🇮', 'RVN': '🇫🇮', 'KEF': '🇮🇸', 'CPH': '🇩🇰',  # Nordic
        'VIE': '🇦🇹', 'PRG': '🇨🇿', 'BRU': '🇧🇪', 'CRL': '🇧🇪', 'ZUR': '🇨🇭', 'BSL': '🇨🇭', 'GVA': '🇨🇭',  # Central Europe
        'BUD': '🇭🇺', 'DUB': '🇮🇪', 'VAR': '🇧🇬', 'BOJ': '🇧🇬', 'SOF': '🇧🇬',  # Eastern Europe
        'OTP': '🇷🇴', 'CLJ': '🇷🇴', 'SPU': '🇭🇷', 'DBV': '🇭🇷', 'ZAD': '🇭🇷',  # Balkans
        'BEG': '🇷🇸', 'TIV': '🇲🇪', 'TGD': '🇲🇪', 'TIA': '🇦🇱', 'KRK': '🇵🇱', 'KTW': '🇵🇱',  # Balkans/Poland
        'LED': '🇷🇺', 'SVO': '🇷🇺', 'DME': '🇷🇺', 'VKO': '🇷🇺', 'AER': '🇷🇺', 'OVB': '🇷🇺', 'IKT': '🇷🇺',  # Russia
        'ULV': '🇷🇺', 'KJA': '🇷🇺', 'KGD': '🇷🇺', 'MSQ': '🇧🇾',  # Russia/Belarus
        'AYT': '🇹🇷', 'IST': '🇹🇷', 'SAW': '🇹🇷', 'ESB': '🇹🇷', 'IZM': '🇹🇷', 'ADB': '🇹🇷',  # Turkey
        'TLV': '🇮🇱', 'EVN': '🇦🇲', 'TBS': '🇬🇪', 'GYD': '🇦🇿', 'KUT': '🇬🇪', 'FRU': '🇰🇬', 'TAS': '🇺🇿',  # Middle East/Central Asia
        'DXB': '🇦🇪', 'SHJ': '🇦🇪', 'AUH': '🇦🇪', 'DWC': '🇦🇪', 'DOH': '🇶🇦', 'RUH': '🇸🇦', 'JED': '🇸🇦', 'DMM': '🇸🇦',  # Gulf
        'SSH': '🇪🇬', 'CAI': '🇪🇬', 'RAK': '🇲🇦', 'DJE': '🇹🇳',  # North Africa
        'TNR': '🇲🇬', 'ZNZ': '🇹🇿',  # East Africa
        'EWR': '🇺🇸', 'JFK': '🇺🇸', 'LGA': '🇺🇸', 'MIA': '🇺🇸', 'PHL': '🇺🇸',  # USA
        'YYZ': '🇨🇦', 'YWG': '🇨🇦', 'YEG': '🇨🇦', 'HAV': '🇨🇺', 'PUJ': '🇩🇴',  # Americas
        'HKT': '🇹🇭', 'BKK': '🇹🇭', 'DMK': '🇹🇭', 'DPS': '🇮🇩',  # Southeast Asia
        'ICN': '🇰🇷', 'GMP': '🇰🇷', 'NRT': '🇯🇵', 'HND': '🇯🇵', 'KIX': '🇯🇵', 'ITM': '🇯🇵',  # East Asia
        'PEK': '🇨🇳', 'CMB': '🇱🇰', 'DEL': '🇮🇳', 'SYD': '🇦🇺'  # Asia/Oceania
    }
    
    _CITIES = {
        'WAW': 'Warsaw', 'FCO': 'Rome', 'MAD': 'Madrid', 'BCN': 'Barcelona', 'LHR': 'London', 'AMS': 'Amsterdam',
        'ATH': 'Athens', 'CDG': 'Paris', 'MUC': 'Munich', 'VIE': 'Vienna', 'PRG': 'Prague', 'BRU': 'Brussels',
        'ORY': 'Paris', 'LIN': 'Milan', 'BGY': 'Milan', 'CIA': 'Rome', 'GOA': 'Genoa', 'PMI': 'Palma',
        'MXP': 'Milan', 'VCE': 'Venice', 'NAP': 'Naples', 'LIS': 'Lisbon', 'LTN': 'London', 'LGW': 'London',  # FIXED: LIS added
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
    
    _COUNTRIES = {
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
        'LIS': 'Portugal', 'OPO': 'Portugal', 'PDL': 'Portugal', 'PXO': 'Portugal',  # FIXED: LIS included
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

class MongoFlightCache:
    """MongoDB-based flight cache with persistent 45-day rolling window - CORRECTED"""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.client = None
        self.db = None
        self.CACHE_DAYS = 45
        self._connect()
    
    def _connect(self):
        """Connect to MongoDB Atlas"""
        try:
            console.info("🔗 Connecting to MongoDB Atlas...")
            self.client = MongoClient(self.connection_string, serverSelectionTimeoutMS=10000)
            self.client.admin.command('ping')
            self.db = self.client['flight_bot_db']
            console.info("✅ Connected to MongoDB Atlas successfully")
        except Exception as e:
            console.info(f"❌ MongoDB connection error: {e}")
            raise
    
    def clear_corrupted_destination_data(self, destination: str):
        """Clear corrupted data for specific destination"""
        try:
            # Update statistics for all destinations that have data
        self._update_all_destination_stats()
        
        # Clean up old data (keep 45-day window)
        self._manage_rolling_window(today)
        
        console.info(f"✅ CORRECTED MongoDB cache update complete - {total_cached:,} entries cached from {successful_destinations} destinations")
        console.info(f"🔍 Data quality: Rejected {data_quality_rejections} corrupted entries")
    
    def _update_all_destination_stats(self):
        """Update statistics with enhanced validation"""
        console.info("📊 Updating destination statistics with quality validation...")
        
        try:
            destinations = self.db.flight_data.distinct('destination')
            stats_updated = 0
            corrupted_destinations = 0
            
            for destination in destinations:
                prices_cursor = self.db.flight_data.find(
                    {'destination': destination, 'data_quality': 'validated'}, 
                    {'price': 1, '_id': 0}
                )
                prices = [doc['price'] for doc in prices_cursor]
                
                if len(prices) >= 50:
                    # Additional validation - check for reasonable price ranges
                    min_price = min(prices)
                    max_price = max(prices)
                    median_price = statistics.median(prices)
                    
                    # Detect corrupted data patterns
                    if self._is_destination_data_corrupted(destination, min_price, max_price, median_price):
                        console.info(f"⚠️ Detected corrupted data for {destination} - clearing cache")
                        self.clear_corrupted_destination_data(destination)
                        corrupted_destinations += 1
                        continue
                    
                    stats_doc = {
                        'destination': destination,
                        'median_price': median_price,
                        'std_dev': statistics.stdev(prices) if len(prices) > 1 else 0,
                        'min_price': min_price,
                        'max_price': max_price,
                        'sample_size': len(prices),
                        'last_updated': datetime.now().strftime('%Y-%m-%d'),
                        'data_quality': 'validated'
                    }
                    
                    self.db.destination_stats.replace_one(
                        {'destination': destination},
                        stats_doc,
                        upsert=True
                    )
                    stats_updated += 1
            
            console.info(f"📊 Updated statistics for {stats_updated} destinations")
            if corrupted_destinations > 0:
                console.info(f"🧹 Cleared {corrupted_destinations} destinations with corrupted data")
            
        except Exception as e:
            console.info(f"⚠️ Error updating destination stats: {e}")
    
    def _is_destination_data_corrupted(self, destination: str, min_price: float, max_price: float, median_price: float) -> bool:
        """Detect corrupted destination data"""
        # Get expected price ranges by region
        region_ranges = {
            'europe_close': (150, 800),      # Oslo, Stockholm, etc.
            'europe_west': (200, 1200),     # Paris, London, etc.
            'middle_east_close': (400, 1500),
            'middle_east_gulf': (500, 2000),
            'asia_close': (600, 2500),
            'asia_southeast': (800, 3500),
            'asia_east': (900, 4000),
            'north_america_east': (800, 4000),
            'north_america_west': (1200, 5000)
        }
        
        # Find destination region
        from SmartAPI import SmartAPI
        temp_api = SmartAPI("dummy", "dummy")
        region = temp_api.ABSOLUTE_REGIONS.get(destination, 'europe_west')
        expected_min, expected_max = region_ranges.get(region, (200, 2000))
        
        # Check for corruption indicators
        if min_price < expected_min * 0.5:  # Too cheap (likely one-way)
            return True
        if median_price > expected_max * 1.5:  # Too expensive (likely business class)
            return True
        if min_price > expected_min * 2:  # No cheap flights at all
            return True
        if max_price > expected_max * 3:  # Extreme outliers
            return True
        
        return False
    
    def _manage_rolling_window(self, current_date: str):
        """Remove data older than 45 days"""
        try:
            cutoff_date = (datetime.strptime(current_date, '%Y-%m-%d') - timedelta(days=self.CACHE_DAYS)).strftime('%Y-%m-%d')
            result = self.db.flight_data.delete_many({'cached_date': {'$lt': cutoff_date}})
            if result.deleted_count > 0:
                console.info(f"🧹 Removed {result.deleted_count:,} old entries (keeping 45-day window)")
        except Exception as e:
            console.info(f"⚠️ Error managing rolling window: {e}")
    
    def get_market_data(self, destination: str) -> Optional[Dict]:
        """Get cached market statistics"""
        try:
            stats = self.db.destination_stats.find_one({'destination': destination})
            if stats and stats.get('data_quality') == 'validated':
                return {
                    'destination': stats['destination'],
                    'median_price': stats['median_price'],
                    'std_dev': stats['std_dev'],
                    'min_price': stats['min_price'],
                    'max_price': stats['max_price'],
                    'sample_size': stats['sample_size']
                }
            return None
        except Exception as e:
            console.info(f"⚠️ Error getting market data for {destination}: {e}")
            return None
    
    def get_cache_summary(self) -> Dict:
        """Get cache statistics"""
        try:
            total_entries = self.db.flight_data.count_documents({})
            ready_destinations = self.db.destination_stats.count_documents({'sample_size': {'$gte': 50}, 'data_quality': 'validated'})
            
            date_range = None
            try:
                oldest = self.db.flight_data.find_one(sort=[('cached_date', 1)])
                newest = self.db.flight_data.find_one(sort=[('cached_date', -1)])
                if oldest and newest:
                    date_range = (oldest['cached_date'], newest['cached_date'])
            except:
                pass
            
            return {
                'total_entries': total_entries,
                'ready_destinations': ready_destinations,
                'date_range': date_range
            }
        except Exception as e:
            console.info(f"⚠️ Error getting cache summary: {e}")
            return {'total_entries': 0, 'ready_destinations': 0, 'date_range': None}
    
    def log_deal_alert(self, deal: VerifiedDeal):
        """Log deal alert to prevent duplicates"""
        try:
            alert_doc = {
                'destination': deal.destination,
                'price': deal.price,
                'z_score': deal.z_score,
                'alert_date': datetime.now().strftime('%Y-%m-%d')
            }
            self.db.deal_alerts.insert_one(alert_doc)
        except Exception as e:
            console.info(f"⚠️ Error logging deal alert: {e}")
    
    def get_recent_alert(self, destination: str) -> Optional[Dict]:
        """Get most recent alert for destination"""
        try:
            alert = self.db.deal_alerts.find_one(
                {'destination': destination},
                sort=[('alert_date', -1)]
            )
            return alert
        except Exception as e:
            console.info(f"⚠️ Error getting recent alert for {destination}: {e}")
            return None
    
    def cleanup_old_alerts(self):
        """Remove alerts older than 30 days"""
        try:
            cutoff_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            result = self.db.deal_alerts.delete_many({'alert_date': {'$lt': cutoff_date}})
            if result.deleted_count > 0:
                console.info(f"🧹 Cleaned up {result.deleted_count} old alerts")
        except Exception as e:
            console.info(f"⚠️ Error cleaning up alerts: {e}")

class SmartAPI:
    """CORRECTED API handler with enhanced data validation"""
    
    # STRICTER PRICE LIMITS - Prevent corruption
    PRICE_LIMITS = (150, 4000)  # More realistic round-trip ranges
    MAX_PRICE_FILTER = 5000     # Reduced from 8000 to prevent business class
    MIN_PRICE_FILTER = 100      # Minimum to prevent one-way confusion
    
    # Regional price expectations for validation
    REGIONAL_PRICE_RANGES = {
        'europe_close': (150, 800),
        'europe_west': (200, 1200), 
        'middle_east_close': (400, 1500),
        'middle_east_gulf': (500, 2000),
        'asia_close': (600, 2500),
        'asia_southeast': (800, 3500),
        'asia_east': (900, 4000),
        'north_america_east': (800, 4000),
        'north_america_west': (1200, 5000)
    }
    
    # YOUR EXACT ABSOLUTE DEAL THRESHOLDS
    ABSOLUTE_DEAL_THRESHOLDS = {
        'europe_close': 250,        # Scandinavia, Balkans, Eastern Europe
        'europe_west': 350,         # UK, France, Germany, Spain, Italy
        'middle_east_close': 700,   # Turkey, Israel, Egypt
        'middle_east_gulf': 750,    # UAE, Qatar, Saudi Arabia
        'north_africa': 650,        # Morocco, Tunisia
        'asia_close': 1100,         # Central Asia, Russia
        'asia_southeast': 1600,     # Thailand, Indonesia, Malaysia  
        'asia_east': 1700,          # Japan, South Korea, China
        'asia_south': 1400,         # India, Sri Lanka
        'north_america_east': 1700, # New York, Toronto, Montreal
        'north_america_west': 2200, # LA, Vancouver, Seattle
        'central_america': 2200,    # Mexico, Cuba
        'south_america': 2800,      # Brazil, Argentina
        'east_africa': 1500,        # Tanzania (Zanzibar), Madagascar
        'west_africa': 2200,        # Ghana, Nigeria, Senegal
        'south_africa': 2500        # South Africa
    }
    
    # Detailed region mappings for absolute thresholds
    ABSOLUTE_REGIONS = {
        # Europe Close (Scandinavia, Balkans, Eastern Europe)
        **{dest: 'europe_close' for dest in [
            'ARN', 'NYO', 'OSL', 'BGO', 'BOO', 'CPH', 'HEL', 'RVN', 'KEF',  # Scandinavia/Nordic
            'VAR', 'BOJ', 'SOF', 'OTP', 'CLJ', 'BEG', 'SPU', 'DBV', 'ZAD',  # Balkans
            'TIV', 'TGD', 'TIA', 'SKG', 'BUD', 'PRG', 'BRU', 'CRL', 'KRK', 'KTW',  # Eastern Europe
            'LED', 'KGD', 'MSQ', 'EVN', 'TBS', 'GYD', 'KUT'  # Eastern Europe/Caucasus
        ]},
        
        # Europe West (UK, France, Germany, Spain, Italy, etc.)
        **{dest: 'europe_west' for dest in [
            'LHR', 'LTN', 'LGW', 'STN', 'GLA', 'BFS', 'DUB',  # UK/Ireland
            'CDG', 'ORY', 'NCE', 'MRS', 'BIQ', 'PIS', 'PUY',  # France
            'FRA', 'MUC', 'BER', 'HAM', 'STR', 'DUS', 'CGN', 'LEJ', 'DTM',  # Germany
            'MAD', 'BCN', 'PMI', 'IBZ', 'VLC', 'ALC', 'AGP', 'BIO', 'LPA', 'TFS', 'SPC',  # Spain
            'FCO', 'MXP', 'LIN', 'BGY', 'CIA', 'VCE', 'NAP', 'PMO', 'BLQ', 'FLR', 'PSA',  # Italy
            'CAG', 'BRI', 'CTA', 'BUS', 'AHO', 'GOA',  # Italy continued
            'AMS', 'RTM', 'EIN', 'ZUR', 'BSL', 'GVA', 'LIS', 'OPO', 'PDL', 'PXO',  # Netherlands/Switzerland/Portugal
            'VIE', 'ATH', 'CFU', 'HER', 'RHO', 'ZTH', 'JTR', 'CHQ'  # Austria/Greece
        ]},
        
        # Middle East Close (Turkey, Israel, Egypt)
        **{dest: 'middle_east_close' for dest in [
            'AYT', 'IST', 'SAW', 'ESB', 'IZM', 'ADB', 'TLV', 'SSH', 'CAI'
        ]},
        
        # Middle East Gulf (UAE, Qatar, Saudi Arabia)
        **{dest: 'middle_east_gulf' for dest in [
            'DXB', 'SHJ', 'AUH', 'DWC', 'DOH', 'RUH', 'JED', 'DMM'
        ]},
        
        # North Africa
        **{dest: 'north_africa' for dest in [
            'RAK', 'DJE'
        ]},
        
        # Asia Close (Central Asia, Russia)
        **{dest: 'asia_close' for dest in [
            'SVO', 'DME', 'VKO', 'AER', 'OVB', 'IKT', 'ULV', 'KJA', 'FRU', 'TAS'
        ]},
        
        # Asia Southeast
        **{dest: 'asia_southeast' for dest in [
            'BKK', 'DMK', 'HKT', 'DPS'
        ]},
        
        # Asia East
        **{dest: 'asia_east' for dest in [
            'NRT', 'HND', 'KIX', 'ITM', 'ICN', 'GMP', 'PEK'
        ]},
        
        # Asia South
        **{dest: 'asia_south' for dest in [
            'DEL', 'CMB'
        ]},
        
        # North America East
        **{dest: 'north_america_east' for dest in [
            'EWR', 'JFK', 'LGA', 'PHL', 'YYZ', 'MIA'
        ]},
        
        # North America West
        **{dest: 'north_america_west' for dest in [
            'YWG', 'YEG'
        ]},
        
        # Central America
        **{dest: 'central_america' for dest in [
            'HAV', 'PUJ'
        ]},
        
        # South America
        **{dest: 'south_america' for dest in [
            'SYD'  # Note: SYD is actually Australia, might need adjustment
        ]},
        
        # East Africa
        **{dest: 'east_africa' for dest in [
            'ZNZ', 'TNR'
        ]},
        
        # West Africa (empty for now)
        **{dest: 'west_africa' for dest in []},
        
        # South Africa (empty for now)
        **{dest: 'south_africa' for dest in []}
    }
    
    # Legacy regions for duration constraints (keep existing)
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
    
    def get_absolute_threshold(self, destination: str) -> float:
        """Get absolute price threshold for destination"""
        region = self.ABSOLUTE_REGIONS.get(destination, 'europe_west')  # Default to europe_west
        return self.ABSOLUTE_DEAL_THRESHOLDS[region]
    
    def _validate_flight_data(self, price: float, departure_date: str, month: str) -> bool:
        """ENHANCED flight data validation - prevents corruption"""
        # Basic price validation - much stricter
        if not (self.MIN_PRICE_FILTER <= price <= self.MAX_PRICE_FILTER):
            return False
        
        # Date validation
        if not departure_date:
            return False
            
        try:
            flight_date = datetime.strptime(departure_date, '%Y-%m-%d')
            request_month = datetime.strptime(month, '%Y-%m')
            month_diff = abs((flight_date.year - request_month.year) * 12 + (flight_date.month - request_month.month))
            if month_diff > 3:  # Flight too far from requested month
                return False
        except ValueError:
            return False
        
        return True
    
    def _validate_combination_quality(self, combo: RoundTripCandidate, destination: str) -> bool:
        """Validate round-trip combination quality"""
        # Get expected price range for destination
        region = self.ABSOLUTE_REGIONS.get(destination, 'europe_west')
        expected_min, expected_max = self.REGIONAL_PRICE_RANGES.get(region, (200, 2000))
        
        # Check if price is within reasonable range
        if combo.total_price < expected_min * 0.7:  # Too cheap - likely one-way
            return False
        if combo.total_price > expected_max * 2:    # Too expensive - likely business class
            return False
        
        # Duration validation
        min_days, max_days = self.get_duration_constraints(destination)
        if not (min_days <= combo.duration_days <= max_days):
            return False
        
        # Basic data integrity
        if combo.total_price <= 0 or combo.duration_days <= 0:
            return False
        
        return True
    
    def _extract_flights(self, origin: str, destination: str, month: str, api_type: str = 'v3') -> List[MatrixEntry]:
        """Extract flights with ENHANCED API parameters"""
        if api_type == 'v3':
            url = "https://api.travelpayouts.com/aviasales/v3/prices_for_dates"
            params = {
                'origin': origin, 
                'destination': destination, 
                'departure_at': month, 
                'return_at': month,
                'one_way': False,           # Explicitly round-trip
                'currency': 'PLN',          # Force PLN currency
                'sorting': 'price', 
                'limit': 1000, 
                'token': self.api_token,
                'trip_class': 0,            # Economy class only (0=economy, 1=business, 2=first)
                'adults': 1,                # Single adult
                'children': 0,              # No children
                'infants': 0                # No infants
            }
        else:  # matrix
            url = "https://api.travelpayouts.com/v2/prices/month-matrix"
            params = {
                'origin': origin, 
                'destination': destination, 
                'month': month,
                'currency': 'PLN', 
                'show_to_affiliates': True, 
                'token': self.api_token,
                'trip_class': 0             # Economy only
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
                
                # Enhanced data extraction with validation
                for entry in flights:
                    if api_type == 'v3':
                        price = entry.get('price', 0)
                        date = entry.get('departure_at', '').split('T')[0] if entry.get('departure_at') else ''
                        transfers = entry.get('transfers', 0)
                        airline = entry.get('airline', 'Unknown')
                        
                        # Additional V3 validation
                        currency = entry.get('currency', 'PLN')
                        trip_class = entry.get('trip_class', 0)
                        
                        # Skip non-PLN or non-economy flights
                        if currency != 'PLN' or trip_class != 0:
                            continue
                            
                    else:
                        price = entry.get('value', 0) or entry.get('price', 0)
                        date = entry.get('depart_date', '') or entry.get('departure_at', '')
                        transfers = entry.get('number_of_changes', 0) or entry.get('transfers', 0)
                        airline = entry.get('gate', 'Unknown') or entry.get('airline', 'Unknown')
                    
                    # Enhanced validation
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
        
        # Primary: V3 API with enhanced parameters
        entries = self._extract_flights(origin, destination, month, 'v3')
        
        # Fallback: Matrix API if insufficient data
        if len(entries) < 50:
            matrix_entries = self._extract_flights(origin, destination, month, 'matrix')
            existing = {(e.date, e.price) for e in entries}
            entries.extend([e for e in matrix_entries if (e.date, e.price) not in existing])
        
        self.cache[cache_key] = entries
        return entries
    
    def generate_comprehensive_roundtrip_combinations(self, origin: str, destination: str, months: List[str]) -> List[RoundTripCandidate]:
        """Generate optimized round-trip combinations with enhanced validation"""
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
        """Verify deal with V3 API - Enhanced parameters"""
        url = "https://api.travelpayouts.com/aviasales/v3/prices_for_dates"
        params = {
            'origin': origin, 
            'destination': destination, 
            'departure_at': departure_date,
            'return_at': return_date, 
            'one_way': False, 
            'currency': 'PLN', 
            'sorting': 'price',
            'limit': 1, 
            'token': self.api_token,
            'trip_class': 0,            # Economy only
            'adults': 1
        }
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                flights = data.get('data', [])
                if flights:
                    flight = flights[0]
                    # Additional validation for verification
                    if flight.get('currency') == 'PLN' and flight.get('trip_class', 0) == 0:
                        return flight
                return None
            elif response.status_code == 429:
                time.sleep(1)
                return None
        except Exception as e:
            logger.warning(f"V3 verification error: {e}")
        return None Remove bad flight data
            result1 = self.db.flight_data.delete_many({'destination': destination})
            # Remove bad stats
            result2 = self.db.destination_stats.delete_one({'destination': destination})
            
            console.info(f"🧹 Cleared {result1.deleted_count} flight entries and stats for {destination}")
            return True
        except Exception as e:
            console.info(f"⚠️ Error clearing {destination} data: {e}")
            return False
    
    def cache_daily_data(self, api, destinations: List[str], months: List[str]):
        """Cache daily flight data with ENHANCED VALIDATION"""
        today = datetime.now().strftime('%Y-%m-%d')
        
        console.info(f"🗃️ Starting CORRECTED MongoDB cache update for {len(destinations)} destinations")
        console.info(f"📅 Cache date: {today} (ALWAYS updates + data quality validation)")
        
        # Remove today's data if any exists
        try:
            deleted = self.db.flight_data.delete_many({'cached_date': today})
            if deleted.deleted_count > 0:
                console.info(f"🧹 Removed {deleted.deleted_count} existing entries for {today}")
        except Exception as e:
            console.info(f"⚠️ Error cleaning today's data: {e}")
        
        all_entries = []
        total_cached = 0
        successful_destinations = 0
        data_quality_rejections = 0
        
        for i, destination in enumerate(destinations, 1):
            console.info(f"📥 [{i}/{len(destinations)}] Caching {destination}")
            
            try:
                combinations = api.generate_comprehensive_roundtrip_combinations('WAW', destination, months)
                
                if combinations:
                    valid_combinations = 0
                    for combo in combinations:
                        # ENHANCED VALIDATION - Prevent data corruption
                        if api._validate_combination_quality(combo, destination):
                            all_entries.append({
                                'destination': destination,
                                'outbound_date': combo.outbound_date,
                                'return_date': combo.return_date,
                                'price': combo.total_price,
                                'transfers_out': combo.outbound_transfers,
                                'transfers_return': combo.return_transfers,
                                'airline': combo.outbound_airline,
                                'cached_date': today,
                                'trip_duration': combo.duration_days,
                                'data_quality': 'validated'
                            })
                            valid_combinations += 1
                        else:
                            data_quality_rejections += 1
                    
                    successful_destinations += 1
                    console.info(f"  ✅ {destination}: Cached {valid_combinations} valid combinations (rejected {len(combinations) - valid_combinations} bad entries)")
                else:
                    console.info(f"  ⚠️ {destination}: No valid combinations found")
                
                # Batch insert every 1000 entries
                if len(all_entries) >= 1000:
                    try:
                        self.db.flight_data.insert_many(all_entries, ordered=False)
                        total_cached += len(all_entries)
                        all_entries.clear()
                    except Exception as e:
                        console.info(f"⚠️ Batch insert error: {e}")
                        all_entries.clear()
                
                if i % 10 == 0:
                    time.sleep(1)
                
            except Exception as e:
                console.info(f"  ❌ {destination}: Error - {e}")
                logger.error(f"Cache error for {destination}: {e}")
        
        # Insert remaining entries
        if all_entries:
            try:
                self.db.flight_data.insert_many(all_entries, ordered=False)
                total_cached += len(all_entries)
            except Exception as e:
                console.info(f"⚠️ Final batch insert error: {e}")
        
        #
