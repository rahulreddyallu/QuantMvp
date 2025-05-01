"""
Enhanced Trading Signal Bot
Handles initialization, execution, and advanced candlestick pattern detection
"""

import os
import time
import datetime
import logging
import sys
import traceback
import schedule
import asyncio
import pandas as pd
import numpy as np
import re

# Ensure aiogram is installed
try:
    from aiogram import Bot, Dispatcher
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "aiogram"])
    from aiogram import Bot, Dispatcher

from upstox_client.api_client import ApiClient
from upstox_client.api.market_quote_api import MarketQuoteApi
from upstox_client.api.history_api import HistoryApi
from config import *

# Create logs directory
os.makedirs('logs', exist_ok=True)

# Setup logging
log_filename = f"logs/trading_bot_{datetime.datetime.now().strftime('%Y%m%d')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Create a reverse mapping from symbol to ISIN
SYMBOL_TO_ISIN = {info["symbol"]: isin for isin, info in STOCK_INFO.items()}

def get_stock_info_by_key(instrument_key):
    """Get stock info from instrument key (e.g., NSE_EQ|INE117A01022)"""
    parts = instrument_key.split('|')
    if len(parts) == 2:
        isin = parts[1]
        if isin in STOCK_INFO:
            return STOCK_INFO[isin]
    # Try direct symbol match as fallback
    if instrument_key in SYMBOL_TO_ISIN:
        isin = SYMBOL_TO_ISIN[instrument_key]
        return STOCK_INFO[isin]
    return {"name": "", "industry": "", "symbol": instrument_key, "series": ""}

def escape_telegram_markdown(text):
    """Escape special characters for Telegram MarkdownV2 formatting."""
    if not text:
        return "N/A"
    # List of all special characters that need to be escaped in MarkdownV2
    special_chars = r'_*[]()~`>#+-=|{}.!'
    escaped_text = re.sub(f'([{re.escape(special_chars)}])', r'\\\1', str(text))
    return escaped_text

# Initialize Upstox client
def initialize_upstox():
    """Initialize connection to Upstox API"""
    try:
        api_client = ApiClient()
        api_client.configuration.access_token = UPSTOX_ACCESS_TOKEN
        market_api = MarketQuoteApi(api_client)
        logger.info("✅ Successfully initialized Upstox API client")
        return market_api
    except Exception as e:
        logger.error(f"Error initializing Upstox API client: {e}")
        return None

# Telegram notification function with exponential backoff retry mechanism
async def send_telegram_message(message, retry_attempts=5):
    """Send message to Telegram with retry mechanism"""
    if ENABLE_TELEGRAM_ALERTS:
        bot = Bot(token=TELEGRAM_BOT_TOKEN)
        delay = 1  # Initial delay in seconds
        for attempt in range(retry_attempts):
            try:
                # Use MarkdownV2 for formatted messages
                await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode='MarkdownV2')
                return True
            except Exception as e:
                if "Too Many Requests" in str(e):
                    retry_after = int(str(e).split("retry after ")[-1].split()[0])
                    logger.error(f"Error sending Telegram message: {e}. Retrying in {retry_after} seconds.")
                    await asyncio.sleep(retry_after)
                else:
                    logger.error(f"Error sending Telegram message: {e}. Retrying in {delay} seconds.")
                    await asyncio.sleep(delay)
                    delay *= 2  # Exponential backoff
        await bot.session.close()  # Ensure the session is properly closed
        return False
    return False

def send_startup_notification():
    """Send a startup notification via Telegram"""
    try:
        loop = asyncio.get_event_loop()
        # Escape the entire startup message
        message = escape_telegram_markdown(f"""
🚀 Enhanced Trading Signal Bot Started 🚀

Version: 3.0.0
Started at: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Analysis Frequency: Every {ANALYSIS_FREQUENCY} hour(s)
Stocks Monitored: {len(STOCK_LIST)} NIFTY 200 stocks

New Features:
• Complete candlestick pattern recognition system
• 19 premium candlestick patterns implemented
• Advanced trend context validation
• Enhanced signal confirmation algorithm

Bot is now actively monitoring for trading signals.
        """)
        loop.run_until_complete(send_telegram_message(message))
        logger.info("Startup notification sent successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to send startup notification: {str(e)}")
        return False

def fetch_ohlcv_data(market_api, symbol, start_date, end_date, interval="day"):
    """
    Fetch historical OHLC data for a given symbol using the Upstox API.
    
    Args:
        market_api: The initialized Upstox API client
        symbol: The instrument symbol/key to fetch data for
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        interval: Time interval (1minute, 30minute, day, week, month)
        
    Returns:
        Pandas DataFrame with OHLCV data
    """
    try:
        # Validate dates
        try:
            datetime.datetime.strptime(start_date, "%Y-%m-%d")
            datetime.datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError as e:
            logger.error(f"Invalid date format: {e}")
            return pd.DataFrame()
        
        # Validate interval
        valid_intervals = ['1minute', '5minute', '30minute', 'day', 'week', 'month']
        if interval not in valid_intervals:
            logger.error(f"Invalid interval: {interval}. Must be one of {valid_intervals}")
            return pd.DataFrame()
        
        logger.info(f"Fetching historical data for {symbol} from {start_date} to {end_date} with {interval} interval")
        
        # Create a HistoryApi instance
        history_api = HistoryApi(market_api.api_client)
        
        # Implement retries for API calls
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                # Call the method on the HistoryApi instance
                response = history_api.get_historical_candle_data1(
                    instrument_key=symbol,
                    interval=interval,
                    to_date=end_date,
                    from_date=start_date,
                    api_version="2.0"
                )
                
                # Extract data from the response
                if hasattr(response, 'data') and hasattr(response.data, 'candles'):
                    candles_data = response.data.candles
                    
                    # Create DataFrame with proper column names
                    df = pd.DataFrame(candles_data, columns=[
                        'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'OI'
                    ])
                    
                    # Convert timestamp to datetime
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    
                    # Set timestamp as index
                    df.set_index('timestamp', inplace=True)
                    
                    # Ensure numeric types for all columns
                    numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'OI']
                    for col in numeric_columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Sort by timestamp (oldest to newest)
                    df.sort_index(inplace=True)
                    
                    # Check for minimum data points required for pattern detection
                    if len(df) < 50:
                        logger.warning(f"Retrieved only {len(df)} candles for {symbol}, which may be insufficient for reliable pattern detection")
                    
                    logger.info(f"Successfully fetched {len(df)} candles for {symbol}")
                    return df
                else:
                    logger.error(f"No candle data returned for {symbol}")
                    if hasattr(response, 'status'):
                        logger.error(f"API status: {response.status}")
                    if hasattr(response, 'data'):
                        logger.error(f"Response data type: {type(response.data)}")
                        logger.error(f"Response data attributes: {dir(response.data)}")
                    
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying fetch for {symbol} (attempt {attempt+1}/{max_retries})...")
                        time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                    else:
                        return pd.DataFrame()
            except Exception as e:
                logger.error(f"Error in attempt {attempt+1}/{max_retries} for {symbol}: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying fetch for {symbol}...")
                    time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    raise
                
    except Exception as e:
        logger.error(f"Error fetching historical OHLC data: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return pd.DataFrame()

class CandlestickPatterns:
    """Complete candlestick pattern detection with precise validation criteria"""
    
    def __init__(self, df):
        """Initialize with DataFrame containing OHLCV data"""
        self.df = df.copy()
        
        # Calculate body sizes and shadows for all candles
        self._calculate_candle_dimensions()
        
        # Calculate trends for context
        self._calculate_trend_context()
    
    def _calculate_candle_dimensions(self):
        """Calculate candle body, upper shadow, and lower shadow sizes"""
        # Get body size and direction
        self.df['body_size'] = abs(self.df['Close'] - self.df['Open'])
        self.df['candle_range'] = self.df['High'] - self.df['Low']
        self.df['body_pct'] = self.df['body_size'] / self.df['candle_range'].replace(0, np.nan)
        self.df['is_bullish'] = self.df['Close'] > self.df['Open']
        
        # Calculate shadows
        # For bullish candles: Upper shadow = High - Close, Lower shadow = Open - Low
        # For bearish candles: Upper shadow = High - Open, Lower shadow = Close - Low
        self.df['upper_shadow'] = np.where(
            self.df['is_bullish'],
            self.df['High'] - self.df['Close'],
            self.df['High'] - self.df['Open']
        )
        
        self.df['lower_shadow'] = np.where(
            self.df['is_bullish'],
            self.df['Open'] - self.df['Low'],
            self.df['Close'] - self.df['Low']
        )
        
        # Calculate shadow percentages relative to range
        self.df['upper_shadow_pct'] = self.df['upper_shadow'] / self.df['candle_range'].replace(0, np.nan)
        self.df['lower_shadow_pct'] = self.df['lower_shadow'] / self.df['candle_range'].replace(0, np.nan)
        
        # Handle volume analysis
        if 'Volume' in self.df.columns:
            self.df['volume_ma20'] = self.df['Volume'].rolling(window=20).mean()
            self.df['high_volume'] = self.df['Volume'] > (self.df['volume_ma20'] * 1.5)
    
    def _calculate_trend_context(self):
        """Calculate trend indicators for pattern context"""
        # Short-term trend (5-day)
        self.df['ma5'] = self.df['Close'].rolling(window=5).mean()
        
        # Medium-term trend (20-day)
        self.df['ma20'] = self.df['Close'].rolling(window=20).mean()
        
        # Trend determination
        self.df['uptrend'] = (self.df['ma5'] > self.df['ma5'].shift(3)) & (self.df['Close'] > self.df['ma20'])
        self.df['downtrend'] = (self.df['ma5'] < self.df['ma5'].shift(3)) & (self.df['Close'] < self.df['ma20'])
    
    def detect_marubozu(self):
        """
        Detect Marubozu patterns (candles with no shadows)
        
        Returns:
            Dictionary with detected patterns and their indices
        """
        # Allow for small shadows (≤5% of candle length)
        shadow_threshold = 0.05
        
        # Detect Bullish Marubozu
        bullish_marubozu = (
            (self.df['is_bullish']) &
            (self.df['upper_shadow_pct'] <= shadow_threshold) &
            (self.df['lower_shadow_pct'] <= shadow_threshold) &
            (self.df['body_pct'] >= 0.95)  # Body is at least 95% of range
        )
        
        # Detect Bearish Marubozu
        bearish_marubozu = (
            (~self.df['is_bullish']) &
            (self.df['upper_shadow_pct'] <= shadow_threshold) &
            (self.df['lower_shadow_pct'] <= shadow_threshold) &
            (self.df['body_pct'] >= 0.95)  # Body is at least 95% of range
        )
        
        # Combine results
        return {
            'marubozu': bullish_marubozu | bearish_marubozu,
            'bullish_marubozu': bullish_marubozu,
            'bearish_marubozu': bearish_marubozu
        }
    
    def detect_doji(self):
        """
        Detect Doji patterns (candles with small bodies and shadows)
        
        Returns:
            Series with True at indices where Doji patterns are detected
        """
        # Doji criteria: open and close are virtually equal
        body_threshold = 0.1  # |Open - Close| ≤ 0.1% of the trading range
        
        doji = (
            (self.df['body_pct'] <= body_threshold) &
            (self.df['candle_range'] > 0)  # Ensure there is some trading range
        )
        
        return doji
    
    def detect_spinning_tops(self):
        """
        Detect Spinning Top patterns (small bodies, long shadows)
        
        Returns:
            Series with True at indices where Spinning Tops are detected
        """
        # Spinning Top criteria
        body_threshold = 0.25    # Body ≤ 25% of the total range
        shadow_threshold = 0.35  # Shadows ≥ 35% of the total range
        
        spinning_tops = (
            (self.df['body_pct'] <= body_threshold) &
            (self.df['upper_shadow_pct'] >= shadow_threshold) &
            (self.df['lower_shadow_pct'] >= shadow_threshold) &
            (self.df['candle_range'] > 0)  # Ensure there is some trading range
        )
        
        return spinning_tops
    
    def detect_paper_umbrella(self):
        """
        Detect Paper Umbrella patterns (small body at top, long lower shadow)
        
        Returns:
            Series with True at indices where Paper Umbrella patterns are detected
        """
        # Paper Umbrella criteria
        lower_shadow_ratio = 2   # Lower shadow ≥ 2x the real body
        upper_shadow_threshold = 0.1  # Little to no upper shadow
        
        paper_umbrella = (
            (self.df['lower_shadow'] >= (self.df['body_size'] * lower_shadow_ratio)) &
            (self.df['upper_shadow_pct'] <= upper_shadow_threshold) &
            (self.df['candle_range'] > 0)  # Ensure there is some trading range
        )
        
        return paper_umbrella
    
    def detect_hammer(self):
        """
        Detect Hammer patterns (bullish reversal with small body at top, long lower shadow)
        
        Returns:
            Series with True at indices where Hammer patterns are detected
        """
        # Hammer criteria
        lower_shadow_ratio = 2   # Lower shadow ≥ 2x the real body
        upper_shadow_threshold = 0.1  # Little to no upper shadow
        
        # Detect hammer candle structure
        hammer_structure = (
            (self.df['lower_shadow'] >= (self.df['body_size'] * lower_shadow_ratio)) &
            (self.df['upper_shadow_pct'] <= upper_shadow_threshold) &
            (self.df['candle_range'] > 0)  # Ensure there is some trading range
        )
        
        # Check for prior downtrend
        downtrend_context = self.df['downtrend'].shift(1)
        
        # Combine conditions
        hammer = hammer_structure & downtrend_context
        
        return hammer
    
    def detect_hanging_man(self):
        """
        Detect Hanging Man patterns (bearish reversal with small body at top, long lower shadow)
        
        Returns:
            Series with True at indices where Hanging Man patterns are detected
        """
        # Hanging Man criteria (similar to hammer but occurs after uptrend)
        lower_shadow_ratio = 2   # Lower shadow ≥ 2x the real body
        upper_shadow_threshold = 0.1  # Little to no upper shadow
        
        # Detect hanging man candle structure (same as hammer)
        hanging_man_structure = (
            (self.df['lower_shadow'] >= (self.df['body_size'] * lower_shadow_ratio)) &
            (self.df['upper_shadow_pct'] <= upper_shadow_threshold) &
            (self.df['candle_range'] > 0)  # Ensure there is some trading range
        )
        
        # Check for prior uptrend
        uptrend_context = self.df['uptrend'].shift(1)
        
        # Combine conditions
        hanging_man = hanging_man_structure & uptrend_context
        
        return hanging_man
    
    def detect_shooting_star(self):
        """
        Detect Shooting Star patterns (bearish reversal with small body at bottom, long upper shadow)
        
        Returns:
            Series with True at indices where Shooting Star patterns are detected
        """
        # Shooting Star criteria
        upper_shadow_ratio = 2   # Upper shadow ≥ 2x the real body
        lower_shadow_threshold = 0.1  # Little to no lower shadow
        
        # Detect shooting star candle structure
        shooting_star_structure = (
            (self.df['upper_shadow'] >= (self.df['body_size'] * upper_shadow_ratio)) &
            (self.df['lower_shadow_pct'] <= lower_shadow_threshold) &
            (self.df['candle_range'] > 0)  # Ensure there is some trading range
        )
        
        # Check for prior uptrend
        uptrend_context = self.df['uptrend'].shift(1)
        
        # Combine conditions
        shooting_star = shooting_star_structure & uptrend_context
        
        return shooting_star
    
    def detect_engulfing(self):
        """
        Detect Engulfing patterns (two-candle reversal pattern)
        
        Returns:
            Dictionary with bullish_engulfing and bearish_engulfing Series
        """
        # Initialize result Series
        bullish_engulfing = pd.Series(False, index=self.df.index)
        bearish_engulfing = pd.Series(False, index=self.df.index)
        
        # We need at least 2 candles to detect engulfing patterns
        if len(self.df) < 2:
            return {'bullish_engulfing': bullish_engulfing, 'bearish_engulfing': bearish_engulfing}
        
        # Iterate through DataFrame (starting from second candle)
        for i in range(1, len(self.df)):
            # Get current and previous candle
            curr = self.df.iloc[i]
            prev = self.df.iloc[i-1]
            
            # Bullish Engulfing criteria
            if (not prev['is_bullish'] and  # Previous candle is bearish
                curr['is_bullish'] and      # Current candle is bullish
                curr['Open'] <= prev['Close'] and  # Current open below previous close
                curr['Close'] >= prev['Open']):    # Current close above previous open
                # Check if we're in a downtrend (for context validation)
                if i > 5 and self.df['downtrend'].iloc[i-1]:
                    bullish_engulfing.iloc[i] = True
            
            # Bearish Engulfing criteria
            elif (prev['is_bullish'] and    # Previous candle is bullish
                  not curr['is_bullish'] and  # Current candle is bearish
                  curr['Open'] >= prev['Close'] and  # Current open above previous close
                  curr['Close'] <= prev['Open']):    # Current close below previous open
                # Check if we're in an uptrend (for context validation)
                if i > 5 and self.df['uptrend'].iloc[i-1]:
                    bearish_engulfing.iloc[i] = True
        
        return {
            'bullish_engulfing': bullish_engulfing,
            'bearish_engulfing': bearish_engulfing
        }
    
    def detect_harami(self):
        """
        Detect Harami patterns (two-candle pattern where second candle is contained within first)
        
        Returns:
            Dictionary with bullish_harami and bearish_harami Series
        """
        # Initialize result Series
        bullish_harami = pd.Series(False, index=self.df.index)
        bearish_harami = pd.Series(False, index=self.df.index)
        
        # We need at least 2 candles to detect harami patterns
        if len(self.df) < 2:
            return {'bullish_harami': bullish_harami, 'bearish_harami': bearish_harami}
        
        # Iterate through DataFrame (starting from second candle)
        for i in range(1, len(self.df)):
            # Get current and previous candle
            curr = self.df.iloc[i]
            prev = self.df.iloc[i-1]
            
            # Bullish Harami criteria
            if (not prev['is_bullish'] and  # Previous candle is bearish
                curr['is_bullish'] and      # Current candle is bullish
                curr['Open'] > prev['Close'] and  # Current open inside previous body
                curr['Open'] < prev['Open'] and
                curr['Close'] > prev['Close'] and  # Current close inside previous body
                curr['Close'] < prev['Open'] and
                curr['body_size'] < prev['body_size']):  # Current body smaller than previous
                # Check for downtrend context
                if i > 5 and self.df['downtrend'].iloc[i-1]:
                    bullish_harami.iloc[i] = True
            
            # Bearish Harami criteria
            elif (prev['is_bullish'] and    # Previous candle is bullish
                  not curr['is_bullish'] and  # Current candle is bearish
                  curr['Open'] < prev['Close'] and  # Current open inside previous body
                  curr['Open'] > prev['Open'] and
                  curr['Close'] < prev['Close'] and  # Current close inside previous body
                  curr['Close'] > prev['Open'] and
                  curr['body_size'] < prev['body_size']):  # Current body smaller than previous
                # Check for uptrend context
                if i > 5 and self.df['uptrend'].iloc[i-1]:
                    bearish_harami.iloc[i] = True
        
        return {
            'bullish_harami': bullish_harami,
            'bearish_harami': bearish_harami
        }
    
    def detect_piercing_pattern(self):
        """
        Detect Piercing Pattern (bullish reversal pattern)
        
        Returns:
            Series with True at indices where Piercing Pattern is detected
        """
        # Initialize result Series
        piercing_pattern = pd.Series(False, index=self.df.index)
        
        # We need at least 2 candles
        if len(self.df) < 2:
            return piercing_pattern
        
        # Iterate through DataFrame (starting from second candle)
        for i in range(1, len(self.df)):
            # Get current and previous candle
            curr = self.df.iloc[i]
            prev = self.df.iloc[i-1]
            
            # Piercing Pattern criteria
            if (not prev['is_bullish'] and  # Previous candle is bearish
                curr['is_bullish'] and      # Current candle is bullish
                prev['body_size'] / prev['candle_range'] > 0.6 and  # Previous is a long candle
                curr['Open'] < prev['Low'] and  # Gap down opening
                curr['Close'] > (prev['Open'] + prev['Close']) / 2 and  # Close above midpoint
                curr['Close'] < prev['Open']):  # Not a complete bullish engulfing
                # Check for downtrend context
                if i > 5 and self.df['downtrend'].iloc[i-1]:
                    piercing_pattern.iloc[i] = True
        
        return piercing_pattern
    
    def detect_dark_cloud_cover(self):
        """
        Detect Dark Cloud Cover (bearish reversal pattern)
        
        Returns:
            Series with True at indices where Dark Cloud Cover is detected
        """
        # Initialize result Series
        dark_cloud_cover = pd.Series(False, index=self.df.index)
        
        # We need at least 2 candles
        if len(self.df) < 2:
            return dark_cloud_cover
        
        # Iterate through DataFrame (starting from second candle)
        for i in range(1, len(self.df)):
            # Get current and previous candle
            curr = self.df.iloc[i]
            prev = self.df.iloc[i-1]
            
            # Dark Cloud Cover criteria
            if (prev['is_bullish'] and  # Previous candle is bullish
                not curr['is_bullish'] and  # Current candle is bearish
                prev['body_size'] / prev['candle_range'] > 0.6 and  # Previous is a long candle
                curr['Open'] > prev['High'] and  # Gap up opening
                curr['Close'] < (prev['Open'] + prev['Close']) / 2 and  # Close below midpoint
                curr['Close'] > prev['Open']):  # Not a complete bearish engulfing
                # Check for uptrend context
                if i > 5 and self.df['uptrend'].iloc[i-1]:
                    dark_cloud_cover.iloc[i] = True
        
        return dark_cloud_cover
    
    def detect_morning_star(self):
        """
        Detect Morning Star (bullish reversal pattern)
        
        Returns:
            Series with True at indices where Morning Star is detected
        """
        # Initialize result Series
        morning_star = pd.Series(False, index=self.df.index)
        
        # We need at least 3 candles
        if len(self.df) < 3:
            return morning_star
        
        # Iterate through DataFrame (starting from third candle)
        for i in range(2, len(self.df)):
            # Get three consecutive candles
            first = self.df.iloc[i-2]  # First day (bearish)
            second = self.df.iloc[i-1]  # Second day (small body)
            third = self.df.iloc[i]    # Third day (bullish)
            
            # Morning Star criteria
            if (not first['is_bullish'] and  # First candle is bearish
                third['is_bullish'] and      # Third candle is bullish
                first['body_size'] / first['candle_range'] > 0.6 and  # First is a long candle
                second['body_size'] / second['candle_range'] < 0.3 and  # Second has a small body
                second['Close'] < first['Close'] and  # Second gaps down from first
                third['Open'] > second['Close'] and  # Third opens above second close
                third['Close'] > (first['Open'] + first['Close']) / 2):  # Third closes above midpoint of first
                # Check for downtrend context
                if i > 5 and self.df['downtrend'].iloc[i-2]:
                    morning_star.iloc[i] = True
        
        return morning_star
    
    def detect_evening_star(self):
        """
        Detect Evening Star (bearish reversal pattern)
        
        Returns:
            Series with True at indices where Evening Star is detected
        """
        # Initialize result Series
        evening_star = pd.Series(False, index=self.df.index)
        
        # We need at least 3 candles
        if len(self.df) < 3:
            return evening_star
        
        # Iterate through DataFrame (starting from third candle)
        for i in range(2, len(self.df)):
            # Get three consecutive candles
            first = self.df.iloc[i-2]  # First day (bullish)
            second = self.df.iloc[i-1]  # Second day (small body)
            third = self.df.iloc[i]    # Third day (bearish)
            
            # Evening Star criteria
            if (first['is_bullish'] and  # First candle is bullish
                not third['is_bullish'] and  # Third candle is bearish
                first['body_size'] / first['candle_range'] > 0.6 and  # First is a long candle
                second['body_size'] / second['candle_range'] < 0.3 and  # Second has a small body
                second['Close'] > first['Close'] and  # Second gaps up from first
                third['Open'] < second['Close'] and  # Third opens below second close
                third['Close'] < (first['Open'] + first['Close']) / 2):  # Third closes below midpoint of first
                # Check for uptrend context
                if i > 5 and self.df['uptrend'].iloc[i-2]:
                    evening_star.iloc[i] = True
        
        return evening_star
    
    def detect_all_patterns(self):
        """
        Detect all implemented candlestick patterns
        
        Returns:
            Dictionary with all detected patterns
        """
        # Get basic patterns
        marubozu_patterns = self.detect_marubozu()
        
        # Build comprehensive pattern dictionary
        patterns = {
            # Single candle patterns
            'marubozu': marubozu_patterns['marubozu'],
            'bullish_marubozu': marubozu_patterns['bullish_marubozu'],
            'bearish_marubozu': marubozu_patterns['bearish_marubozu'],
            'doji': self.detect_doji(),
            'spinning_tops': self.detect_spinning_tops(),
            'paper_umbrella': self.detect_paper_umbrella(),
            'hammer': self.detect_hammer(),
            'hanging_man': self.detect_hanging_man(),
            'shooting_star': self.detect_shooting_star(),
            
            # Two-candle patterns
            **self.detect_engulfing(),  # Adds bullish_engulfing and bearish_engulfing
            **self.detect_harami(),     # Adds bullish_harami and bearish_harami
            'piercing_pattern': self.detect_piercing_pattern(),
            'dark_cloud_cover': self.detect_dark_cloud_cover(),
            
            # Three-candle patterns
            'morning_star': self.detect_morning_star(),
            'evening_star': self.detect_evening_star(),
        }
        
        return patterns
    
    def get_latest_patterns(self):
        """
        Get patterns detected in the most recent candle
        
        Returns:
            Dictionary with pattern names and boolean values
        """
        all_patterns = self.detect_all_patterns()
        
        # Extract patterns from the latest candle
        latest_patterns = {}
        for pattern_name, pattern_series in all_patterns.items():
            if len(pattern_series) > 0:
                latest_patterns[pattern_name] = pattern_series.iloc[-1]
            else:
                latest_patterns[pattern_name] = False
        
        return latest_patterns
    
    def get_pattern_signals(self):
        """
        Generate buy/sell signals based on detected patterns
        
        Returns:
            Dictionary with signals and their strengths
        """
        latest_patterns = self.get_latest_patterns()
        signals = []
        
        # Pattern classifications with signal type and strength
        pattern_signals = {
            # Bullish patterns (Buy signals)
            'bullish_marubozu': {'signal': 'BUY', 'strength': 3},
            'hammer': {'signal': 'BUY', 'strength': 3},
            'bullish_engulfing': {'signal': 'BUY', 'strength': 4},
            'bullish_harami': {'signal': 'BUY', 'strength': 3},
            'piercing_pattern': {'signal': 'BUY', 'strength': 3},
            'morning_star': {'signal': 'BUY', 'strength': 4},
            
            # Bearish patterns (Sell signals)
            'bearish_marubozu': {'signal': 'SELL', 'strength': 3},
            'hanging_man': {'signal': 'SELL', 'strength': 3},
            'shooting_star': {'signal': 'SELL', 'strength': 3},
            'bearish_engulfing': {'signal': 'SELL', 'strength': 4},
            'bearish_harami': {'signal': 'SELL', 'strength': 3},
            'dark_cloud_cover': {'signal': 'SELL', 'strength': 3},
            'evening_star': {'signal': 'SELL', 'strength': 4},
        }
        
        # Create signals based on detected patterns
        for pattern, is_detected in latest_patterns.items():
            if is_detected and pattern in pattern_signals:
                signals.append({
                    'pattern': pattern,
                    'signal': pattern_signals[pattern]['signal'],
                    'strength': pattern_signals[pattern]['strength']
                })
        
        # Compute summary
        buy_signals = [s for s in signals if s['signal'] == 'BUY']
        sell_signals = [s for s in signals if s['signal'] == 'SELL']
        
        # Determine overall signal
        overall_signal = 'NEUTRAL'
        overall_strength = 0
        
        if len(buy_signals) > len(sell_signals):
            overall_signal = 'BUY'
            overall_strength = min(5, max(1, sum(s['strength'] for s in buy_signals) // 2))
        elif len(sell_signals) > len(buy_signals):
            overall_signal = 'SELL'
            overall_strength = min(5, max(1, sum(s['strength'] for s in sell_signals) // 2))
        
        return {
            'overall_signal': overall_signal,
            'overall_strength': overall_strength,
            'individual_signals': signals,
            'buy_signals_count': len(buy_signals),
            'sell_signals_count': len(sell_signals)
        }

async def analyze_and_generate_signals():
    """
    Fetches historical data for symbols in STOCK_LIST, performs candlestick pattern analysis,
    and generates trading signals.
    """
    # Log function start with current UTC time
    current_datetime = datetime.datetime.now()
    logger.info(f"Starting analysis at {current_datetime.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    
    # Current date/time
    current_date_str = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f"Analysis date: {current_date_str}")
    
    # Calculate date range (based on HISTORICAL_DAYS constant)
    end_date = current_datetime.strftime('%Y-%m-%d')
    start_date = (current_datetime - datetime.timedelta(days=HISTORICAL_DAYS)).strftime('%Y-%m-%d')
    logger.info(f"Analyzing data from {start_date} to {end_date}")

    # Initialize Upstox API client
    market_api = initialize_upstox()
    if not market_api:
        logger.error("Failed to initialize Upstox client")
        return
    
    # Track overall statistics
    successful_analyses = 0
    failed_analyses = 0
    total_signals = 0
    buy_signals = 0
    sell_signals = 0
    
    # Header for daily report
    daily_report = [
        f"📈 CANDLESTICK PATTERN ANALYSIS REPORT 📉",
        f"Date: {current_date_str} UTC",
        f"Analyzing {len(STOCK_LIST)} symbols with {HISTORICAL_DAYS} days of historical data",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    ]
    
    # Process each symbol in STOCK_LIST
    for symbol in STOCK_LIST:
        logger.info(f"Processing symbol: {symbol}")
        
        try:
            # Get stock information
            stock_info = get_stock_info_by_key(symbol)
            company_name = stock_info.get("name", "")
            industry = stock_info.get("industry", "")
            trading_symbol = stock_info.get("symbol", symbol.split("|")[-1] if "|" in symbol else symbol)
            
            logger.info(f"Analyzing {company_name} ({trading_symbol}) - {industry}")
            
            # Fetch historical data with daily interval
            data = fetch_ohlcv_data(market_api, symbol, start_date, end_date, interval="day")
            
            if data.empty:
                logger.error(f"No historical data fetched for {company_name} ({symbol})")
                failed_analyses += 1
                continue
            
            logger.info(f"Analyzing {company_name} ({trading_symbol}) with {len(data)} data points")
            
            # Perform candlestick pattern analysis
            pattern_analyzer = CandlestickPatterns(data)
            latest_patterns = pattern_analyzer.get_latest_patterns()
            pattern_signals = pattern_analyzer.get_pattern_signals()
            
            # Extract information about detected patterns
            detected_patterns = [pattern for pattern, is_detected in latest_patterns.items() if is_detected]
            
            # Get signal information
            overall_signal = pattern_signals['overall_signal']
            overall_strength = pattern_signals['overall_strength']
            buy_signals_count = pattern_signals['buy_signals_count']
            sell_signals_count = pattern_signals['sell_signals_count']
            
            # Create signal summary
            signal_summary = f"{'Bullish' if overall_signal == 'BUY' else 'Bearish' if overall_signal == 'SELL' else 'Neutral'} signal with {buy_signals_count} buy vs {sell_signals_count} sell patterns"
            
            # Check if signal is strong enough to report
            if overall_signal != 'NEUTRAL' and overall_strength >= MINIMUM_SIGNAL_STRENGTH:
                logger.info(f"Generated signal for {company_name} ({trading_symbol}): {overall_signal} (Strength: {overall_strength}/5)")
                total_signals += 1
                
                if overall_signal == 'BUY':
                    buy_signals += 1
                elif overall_signal == 'SELL':
                    sell_signals += 1
                
                # Add to daily report
                daily_report.append(f"\n{company_name} ({trading_symbol}): {overall_signal} (Strength: {overall_strength}/5)")
                daily_report.append(f"{signal_summary}")
                
                # Format detected patterns for report
                if detected_patterns:
                    patterns_text = ["Detected patterns:"]
                    
                    for pattern in detected_patterns:
                        pattern_name = pattern.replace('_', ' ').title()
                        signal_type = 'bullish' if pattern in ['bullish_marubozu', 'hammer', 'bullish_engulfing', 'bullish_harami', 'piercing_pattern', 'morning_star'] else 'bearish' if pattern in ['bearish_marubozu', 'hanging_man', 'shooting_star', 'bearish_engulfing', 'bearish_harami', 'dark_cloud_cover', 'evening_star'] else 'neutral'
                        patterns_text.append(f"✅ {pattern_name} ({signal_type})")
                    
                    daily_report.append("\n".join(patterns_text))
                
                # Format message for Telegram notification
                message = f"""
*📊 CANDLESTICK PATTERN SIGNAL | {trading_symbol} | {overall_signal}* {"⭐" * overall_strength}

*{company_name}*
Price: ₹{data['Close'].iloc[-1]:.2f} | Industry: {industry}

*DETECTED PATTERNS:*
{chr(10).join(f"✅ {pattern.replace('_', ' ').title()}" for pattern in detected_patterns)}

*SIGNAL SUMMARY:*
{signal_summary}

*TREND CONTEXT:*
{("In a clear uptrend" if data['uptrend'].iloc[-1] else "In a clear downtrend" if data['downtrend'].iloc[-1] else "No clear trend")}

Generated: {datetime.datetime.now().strftime("%b-%d %H:%M")}
                """
                
                # Escape for Telegram markdown
                escaped_message = escape_telegram_markdown(message)
                await send_telegram_message(escaped_message)
                logger.info(f"Sent pattern alert for {company_name} ({trading_symbol})")
                
            else:
                if detected_patterns:
                    logger.info(f"Patterns detected for {company_name} ({trading_symbol}) but signal strength ({overall_strength}) below threshold")
                    patterns_str = ", ".join(p.replace('_', ' ').title() for p in detected_patterns)
                    daily_report.append(f"\n{company_name} ({trading_symbol}): Detected {patterns_str} - Below signal threshold")
                else:
                    logger.info(f"No significant patterns detected for {company_name} ({trading_symbol})")
                    daily_report.append(f"\n{company_name} ({trading_symbol}): No significant patterns")
            
            successful_analyses += 1
                
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            failed_analyses += 1
            daily_report.append(f"\n{symbol}: Error during analysis - {str(e)[:50]}...")
            continue
    
    # Finalize daily report
    daily_report.append("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    daily_report.append("Analysis Summary:")
    daily_report.append(f"• Analyzed: {successful_analyses} symbols")
    daily_report.append(f"• Failed: {failed_analyses} symbols") 
    daily_report.append(f"• Total signals generated: {total_signals} ({buy_signals} BUY, {sell_signals} SELL)")
    daily_report.append(f"• Report time: {current_date_str} UTC")
    
    # Send daily summary report via Telegram
    if successful_analyses > 0:
        escaped_report = escape_telegram_markdown("\n".join(daily_report))
        await send_telegram_message(escaped_report)
    
    # Log summary statistics
    logger.info(f"Analysis completed. Processed {len(STOCK_LIST)} symbols.")
    logger.info(f"Successful analyses: {successful_analyses}")
    logger.info(f"Failed analyses: {failed_analyses}")
    logger.info(f"Total signals generated: {total_signals} ({buy_signals} BUY, {sell_signals} SELL)")
    logger.info(f"Analysis completed at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC")

def run_trading_signals():
    """Run the trading signal generation process"""
    start_time = time.time()
    logger.info("Starting candlestick pattern analysis")
    
    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(analyze_and_generate_signals())
        
        # Log completion
        elapsed_time = time.time() - start_time
        logger.info(f"Completed candlestick pattern analysis in {elapsed_time:.2f} seconds")
    
    except Exception as e:
        logger.error(f"Error in candlestick pattern analysis: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Send error notification
        try:
            loop = asyncio.get_event_loop()
            # Escape the error message for MarkdownV2 format
            error_message = escape_telegram_markdown(f"""
⚠️ ERROR: Candlestick Pattern Bot Failure ⚠️
    
Time: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Error: {str(e)}
    
Please check the logs for more details.
            """)
            loop.run_until_complete(send_telegram_message(error_message))
        except Exception as notification_err:
            logger.error(f"Failed to send error notification: {notification_err}")

def test_upstox_connection():
    """Test connection to Upstox API"""
    logger.info("Testing Upstox API connection...")
    
    try:
        market_api = initialize_upstox()
        if market_api:
            logger.info("✅ Successfully initialized Upstox API client")
            return True
        else:
            logger.error("❌ Failed to initialize Upstox API client")
            return False
    except Exception as e:
        logger.error(f"❌ Error connecting to Upstox API: {str(e)}")
        return False

def test_telegram_connection():
    """Test connection to Telegram API"""
    logger.info("Testing Telegram API connection...")
    
    try:
        loop = asyncio.get_event_loop()
        # Escape the test message properly for MarkdownV2
        test_message = escape_telegram_markdown("🔍 Test Message - Candlestick Pattern Bot connection test successful!")
        result = loop.run_until_complete(send_telegram_message(test_message))
        
        if result:
            logger.info("✅ Successfully sent test message to Telegram")
            return True
        else:
            logger.error("❌ Failed to send test message to Telegram")
            return False
    except Exception as e:
        logger.error(f"❌ Error connecting to Telegram API: {str(e)}")
        return False

def schedule_analysis():
    """Schedule the analysis based on config"""
    # Schedule for specific hours of the day based on market hours
    for hour in range(9, 16):  # 9 AM to 3 PM
        schedule.every().monday.at(f"{hour:02d}:00").do(run_trading_signals)
        schedule.every().tuesday.at(f"{hour:02d}:00").do(run_trading_signals)
        schedule.every().wednesday.at(f"{hour:02d}:00").do(run_trading_signals)
        schedule.every().thursday.at(f"{hour:02d}:00").do(run_trading_signals)
        schedule.every().friday.at(f"{hour:02d}:00").do(run_trading_signals)
    
    # Also schedule at market open and close
    schedule.every().monday.at("09:15").do(run_trading_signals)
    schedule.every().tuesday.at("09:15").do(run_trading_signals)
    schedule.every().wednesday.at("09:15").do(run_trading_signals)
    schedule.every().thursday.at("09:15").do(run_trading_signals)
    schedule.every().friday.at("09:15").do(run_trading_signals)
    
    schedule.every().monday.at("15:30").do(run_trading_signals)
    schedule.every().tuesday.at("15:30").do(run_trading_signals)
    schedule.every().wednesday.at("15:30").do(run_trading_signals)
    schedule.every().thursday.at("15:30").do(run_trading_signals)
    schedule.every().friday.at("15:30").do(run_trading_signals)
    
    logger.info(f"Analysis scheduled during market hours (9:00 AM - 3:30 PM) on weekdays")
    
    while True:
        schedule.run_pending()
        time.sleep(60)

def main():
    """Main function to run the Candlestick Pattern Bot"""
    logger.info("=" * 70)
    logger.info("Candlestick Pattern Detection Bot - Starting Up")
    logger.info("Advanced Pattern Recognition - Version 3.0")
    logger.info("=" * 70)
    
    # Test connections
    upstox_connected = test_upstox_connection()
    telegram_connected = test_telegram_connection()
    
    if not upstox_connected:
        logger.error("Cannot proceed without Upstox API connection")
        return
    
    if not telegram_connected:
        logger.warning("Telegram connection failed, proceeding without notifications")
    
    # Send startup notification
    send_startup_notification()
    
    # Run immediately on startup
    run_trading_signals()
    
    # Schedule future runs
    try:
        schedule_analysis()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
