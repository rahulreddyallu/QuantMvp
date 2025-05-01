"""
Enhanced Quantitative Analysis Engine
Combines technical indicators and candlestick pattern recognition

Version: 3.5.0
Author: rahulreddyallu
Date: 2025-05-01
"""

import os
import time
import datetime
import logging
import sys
import traceback
import asyncio
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Union, Tuple, Any, Optional

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


# ===============================================================
# Utility Functions
# ===============================================================

def setup_logging(config):
    """
    Setup logging configuration
    
    Args:
        config: Configuration dictionary with logging parameters
        
    Returns:
        Logger object
    """
    # Create logs directory
    os.makedirs(config.get('LOG_DIRECTORY', 'logs'), exist_ok=True)

    # Setup logging
    log_filename = f"{config.get('LOG_DIRECTORY', 'logs')}/trading_bot_{datetime.datetime.now().strftime('%Y%m%d')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def get_stock_info_by_key(instrument_key, stock_info_dict):
    """
    Get stock info from instrument key (e.g., NSE_EQ|INE117A01022)
    
    Args:
        instrument_key: The instrument key to lookup
        stock_info_dict: Dictionary mapping ISINs to stock information
        
    Returns:
        Dictionary with stock information
    """
    # Create a reverse mapping from symbol to ISIN
    symbol_to_isin = {info["symbol"]: isin for isin, info in stock_info_dict.items()}
    
    parts = instrument_key.split('|')
    if len(parts) == 2:
        isin = parts[1]
        if isin in stock_info_dict:
            return stock_info_dict[isin]
    
    # Try direct symbol match as fallback
    if instrument_key in symbol_to_isin:
        isin = symbol_to_isin[instrument_key]
        return stock_info_dict[isin]
    
    return {"name": "", "industry": "", "symbol": instrument_key, "series": ""}

def escape_telegram_markdown(text):
    """
    Escape special characters for Telegram MarkdownV2 formatting.
    
    Args:
        text: The text to escape
        
    Returns:
        Escaped text string
    """
    if not text:
        return "N/A"
    # List of all special characters that need to be escaped in MarkdownV2
    special_chars = r'_*[]()~`>#+-=|{}.!'
    escaped_text = re.sub(f'([{re.escape(special_chars)}])', r'\\\1', str(text))
    return escaped_text


# ===============================================================
# Market Data Handling
# ===============================================================

def initialize_upstox(config):
    """
    Initialize connection to Upstox API
    
    Args:
        config: Configuration dictionary with API credentials
    
    Returns:
        Tuple of (MarketQuoteApi, ApiClient) if successful, (None, None) otherwise
    """
    try:
        api_client = ApiClient()
        api_client.configuration.access_token = config.get('UPSTOX_ACCESS_TOKEN', '')
        market_api = MarketQuoteApi(api_client)
        logger.info("✅ Successfully initialized Upstox API client")
        return market_api, api_client
    except Exception as e:
        logger.error(f"Error initializing Upstox API client: {e}")
        return None, None

def fetch_ohlcv_data(market_api, symbol, start_date, end_date, interval="day", api_version="2.0"):
    """
    Fetch historical OHLC data for a given symbol using the Upstox API.
    
    Args:
        market_api: The initialized Upstox API client
        symbol: The instrument symbol/key to fetch data for
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        interval: Time interval (1minute, 30minute, day, week, month)
        api_version: API version to use
        
    Returns:
        Pandas DataFrame with OHLCV data or empty DataFrame if failed
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
                    api_version=api_version
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
                    logger.error(traceback.format_exc())
                    return pd.DataFrame()
                
    except Exception as e:
        logger.error(f"Error fetching historical OHLC data: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return pd.DataFrame()


# ===============================================================
# Candlestick Pattern Recognition
# ===============================================================

class CandlestickPatterns:
    """Complete candlestick pattern detection with precise validation criteria"""
    
    def __init__(self, df):
        """
        Initialize with DataFrame containing OHLCV data
        
        Args:
            df: DataFrame with OHLCV data
        """
        # Ensure we have a valid DataFrame
        if df is None or len(df) == 0:
            raise ValueError("Empty DataFrame provided")
            
        # Make a deep copy to avoid modifying the original
        self.df = df.copy()
        
        # Calculate body sizes and shadows for all candles
        self._calculate_candle_dimensions()
        
        # Calculate trends for context
        self._calculate_trend_context()
    
    def _calculate_candle_dimensions(self):
        """
        Calculate candle body, upper shadow, and lower shadow sizes
        """
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
            # Calculate volume moving average
            self.df['volume_ma20'] = self.df['Volume'].rolling(window=20).mean()
            
            # Flag high volume candles (50% above average)
            self.df['high_volume'] = self.df['Volume'] > (self.df['volume_ma20'] * 1.5)
            
            # Calculate volume ratio to average
            self.df['volume_ratio'] = self.df['Volume'] / self.df['volume_ma20'].replace(0, np.nan)
    
    def _calculate_trend_context(self):
        """
        Calculate trend indicators for pattern context with volume confirmation
        """
        # Short-term trend (5-day)
        self.df['ma5'] = self.df['Close'].rolling(window=5).mean()
        
        # Medium-term trend (20-day)
        self.df['ma20'] = self.df['Close'].rolling(window=20).mean()
        
        # Basic trend determination
        price_uptrend = (self.df['ma5'] > self.df['ma5'].shift(3)) & (self.df['Close'] > self.df['ma20'])
        price_downtrend = (self.df['ma5'] < self.df['ma5'].shift(3)) & (self.df['Close'] < self.df['ma20'])
        
        # Volume confirmation if available
        if 'Volume' in self.df.columns:
            # Calculate volume trend (increasing or decreasing volume)
            self.df['volume_ma5'] = self.df['Volume'].rolling(window=5).mean()
            volume_increasing = self.df['volume_ma5'] > self.df['volume_ma5'].shift(3)
            
            # Strong uptrend: Price rising with increasing volume
            self.df['uptrend'] = price_uptrend & (
                volume_increasing | 
                # Or high volume on up days
                (self.df['high_volume'] & self.df['is_bullish'])
            )
            
            # Strong downtrend: Price falling with increasing volume
            self.df['downtrend'] = price_downtrend & (
                volume_increasing | 
                # Or high volume on down days
                (self.df['high_volume'] & ~self.df['is_bullish'])
            )
        else:
            # If no volume data, use just price
            self.df['uptrend'] = price_uptrend
            self.df['downtrend'] = price_downtrend
    
    def detect_marubozu(self):
        """
        Detect Marubozu patterns (candles with no shadows)
        
        Returns:
            Dictionary with detected patterns and their indices
        """
        # Ensure we have data
        if len(self.df) == 0:
            return {'marubozu': pd.Series(dtype=bool), 
                    'bullish_marubozu': pd.Series(dtype=bool), 
                    'bearish_marubozu': pd.Series(dtype=bool)}
            
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
        # Ensure we have data
        if len(self.df) == 0:
            return pd.Series(dtype=bool)
            
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
        # Ensure we have data
        if len(self.df) == 0:
            return pd.Series(dtype=bool)
            
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
        # Ensure we have data
        if len(self.df) == 0:
            return pd.Series(dtype=bool)
            
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
        # Ensure we have data and sufficient history
        if len(self.df) < 5:
            return pd.Series(False, index=self.df.index)
            
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
        # Ensure we have data and sufficient history
        if len(self.df) < 5:
            return pd.Series(False, index=self.df.index)
            
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
        # Ensure we have data and sufficient history
        if len(self.df) < 5:
            return pd.Series(False, index=self.df.index)
            
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
        if len(self.df) < 6:  # Need at least 6 candles (5 for trend context + current)
            return {'bullish_engulfing': bullish_engulfing, 'bearish_engulfing': bearish_engulfing}
        
        # Iterate through DataFrame (starting from second candle)
        for i in range(1, len(self.df)):
            # Ensure we don't go out of bounds
            if i >= len(self.df):
                break
                
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
        if len(self.df) < 6:  # Need at least 6 candles (5 for trend context + current)
            return {'bullish_harami': bullish_harami, 'bearish_harami': bearish_harami}
        
        # Iterate through DataFrame (starting from second candle)
        for i in range(1, len(self.df)):
            # Ensure we don't go out of bounds
            if i >= len(self.df):
                break
                
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
        
        # We need at least 2 candles to detect piercing patterns
        if len(self.df) < 6:  # Need at least 6 candles (5 for trend context + current)
            return piercing_pattern
        
        # Iterate through DataFrame (starting from second candle)
        for i in range(1, len(self.df)):
            # Ensure we don't go out of bounds
            if i >= len(self.df):
                break
                
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
        
        # We need at least 2 candles to detect dark cloud cover
        if len(self.df) < 6:  # Need at least 6 candles (5 for trend context + current)
            return dark_cloud_cover
        
        # Iterate through DataFrame (starting from second candle)
        for i in range(1, len(self.df)):
            # Ensure we don't go out of bounds
            if i >= len(self.df):
                break
                
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
        
        # We need at least 3 candles to detect morning star
        if len(self.df) < 7:  # Need at least 7 candles (5 for trend context + 2 previous)
            return morning_star
        
        # Iterate through DataFrame (starting from third candle)
        for i in range(2, len(self.df)):
            # Ensure we don't go out of bounds
            if i >= len(self.df):
                break
                
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
        
        # We need at least 3 candles to detect evening star
        if len(self.df) < 7:  # Need at least 7 candles (5 for trend context + 2 previous)
            return evening_star
        
        # Iterate through DataFrame (starting from third candle)
        for i in range(2, len(self.df)):
            # Ensure we don't go out of bounds
            if i >= len(self.df):
                break
                
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
            
            # Neutral patterns with context signals
            'doji': {'signal': 'NEUTRAL', 'strength': 1},
            'spinning_tops': {'signal': 'NEUTRAL', 'strength': 1}
        }
        
        # Add market insight for neutral patterns
        if latest_patterns.get('doji', False):
            # Doji indicates indecision - signal depends on trend context
            if self.df['uptrend'].iloc[-1]:
                signals.append({
                    'pattern': 'doji',
                    'signal': 'CAUTION',
                    'strength': 2,
                    'note': 'Doji in uptrend suggests potential reversal'
                })
            elif self.df['downtrend'].iloc[-1]:
                signals.append({
                    'pattern': 'doji',
                    'signal': 'WATCH',
                    'strength': 2,
                    'note': 'Doji in downtrend suggests potential reversal'
                })
        
        if latest_patterns.get('spinning_tops', False):
            # Spinning tops also indicate indecision
            if self.df['uptrend'].iloc[-1]:
                signals.append({
                    'pattern': 'spinning_tops',
                    'signal': 'CAUTION',
                    'strength': 1,
                    'note': 'Spinning Top in uptrend suggests weakening momentum'
                })
            elif self.df['downtrend'].iloc[-1]:
                signals.append({
                    'pattern': 'spinning_tops',
                    'signal': 'WATCH',
                    'strength': 1,
                    'note': 'Spinning Top in downtrend suggests slowing momentum'
                })
        
        # Create signals based on detected patterns
        for pattern, is_detected in latest_patterns.items():
            if is_detected and pattern in pattern_signals and pattern_signals[pattern]['signal'] != 'NEUTRAL':
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


# ===============================================================
# Technical Indicators Analysis
# ===============================================================

class TechnicalIndicators:
    """Calculate and interpret technical indicators for trading signals"""
    
    def __init__(self, df):
        """
        Initialize with DataFrame containing OHLCV data
        
        Args:
            df: DataFrame with OHLCV data
        """
        # Ensure we have a valid DataFrame
        if df is None or len(df) == 0:
            raise ValueError("Empty DataFrame provided")
            
        # Make a deep copy to avoid modifying the original
        self.df = df.copy()
        
        # Initialize results dictionary to store indicator calculations
        self.results = {}
        
    def calculate_all(self):
        """
        Calculate all technical indicators
        
        Returns:
            DataFrame with all indicator columns added
        """
        # Price Action Indicators
        self.calculate_support_resistance()
        self.calculate_fibonacci_retracement()
        
        # Volume Indicators
        self.calculate_volume_ratio()
        self.calculate_obv()
        self.calculate_vwap()
        
        # Moving Averages
        self.calculate_sma(periods=[5, 10, 20, 50, 200])
        self.calculate_ema(periods=[5, 12, 26, 50])
        
        # Oscillators
        self.calculate_rsi()
        self.calculate_macd()
        self.calculate_stochastic()
        self.calculate_stochastic_rsi()
        self.calculate_adx()
        self.calculate_aroon()
        
        # Volatility Indicators
        self.calculate_bollinger_bands()
        self.calculate_atr()
        self.calculate_atr_bands()
        self.calculate_supertrend()
        
        # Specialized Indicators
        self.calculate_alligator()
        self.calculate_cpr()
        
        # Dow Theory Patterns
        self.detect_double_patterns()
        self.detect_triple_patterns()
        self.detect_trading_range()
        self.detect_flag_patterns()
        
        # Return the updated DataFrame
        return self.df
        
    #
    # 1. SUPPORT AND RESISTANCE
    #
    
    def calculate_support_resistance(self, lookback=100, price_tolerance=0.02, window_size=5):
        """
        Calculate key support and resistance levels
        
        Args:
            lookback: Number of periods to analyze
            price_tolerance: Percentage tolerance to group levels
            window_size: Size of window to identify swings
            
        Returns:
            Dictionary with support and resistance levels
        """
        # Ensure we have enough data
        lookback = min(lookback, len(self.df))
        recent_data = self.df.iloc[-lookback:].copy()
        
        # Find swing highs and lows
        high_series = recent_data['High']
        low_series = recent_data['Low']
        
        # Function to find local extrema
        def is_swing_high(x):
            if len(x) < window_size:
                return False
            mid_point = len(x) // 2
            return x[mid_point] == max(x)
            
        def is_swing_low(x):
            if len(x) < window_size:
                return False
            mid_point = len(x) // 2
            return x[mid_point] == min(x)
        
        # Find swing points
        recent_data['is_swing_high'] = high_series.rolling(window=window_size, center=True).apply(
            is_swing_high, raw=True
        ).fillna(0).astype(bool)
        
        recent_data['is_swing_low'] = low_series.rolling(window=window_size, center=True).apply(
            is_swing_low, raw=True
        ).fillna(0).astype(bool)
        
        # Extract price levels
        swing_highs = recent_data[recent_data['is_swing_high']]['High'].tolist()
        swing_lows = recent_data[recent_data['is_swing_low']]['Low'].tolist()
        
        # Group similar levels
        def cluster_levels(price_levels):
            if not price_levels:
                return []
                
            # Sort levels
            sorted_levels = sorted(price_levels)
            clusters = []
            current_cluster = [sorted_levels[0]]
            
            # Group levels that are within tolerance
            for level in sorted_levels[1:]:
                if abs(level - current_cluster[-1]) / current_cluster[-1] <= price_tolerance:
                    current_cluster.append(level)
                else:
                    # Calculate the average of the cluster and add to clusters
                    clusters.append(sum(current_cluster) / len(current_cluster))
                    current_cluster = [level]
                    
            # Add the last cluster
            if current_cluster:
                clusters.append(sum(current_cluster) / len(current_cluster))
                
            return clusters
        
        # Get clustered levels
        resistance_levels = cluster_levels(swing_highs)
        support_levels = cluster_levels(swing_lows)
        
        # Store in dataframe for latest candle
        if resistance_levels:
            # Find nearest resistance
            current_price = self.df['Close'].iloc[-1]
            resistance_above = [level for level in resistance_levels if level > current_price]
            nearest_resistance = min(resistance_above) if resistance_above else None
            
            self.df['nearest_resistance'] = nearest_resistance
        
        if support_levels:
            # Find nearest support
            current_price = self.df['Close'].iloc[-1]
            support_below = [level for level in support_levels if level < current_price]
            nearest_support = max(support_below) if support_below else None
            
            self.df['nearest_support'] = nearest_support
            
        # Calculate distance to support/resistance as percentage
        if 'nearest_support' in self.df.columns and 'nearest_resistance' in self.df.columns:
            current_price = self.df['Close'].iloc[-1]
            
            if not pd.isna(self.df['nearest_support'].iloc[-1]):
                support_distance = (current_price - self.df['nearest_support'].iloc[-1]) / current_price * 100
                self.df['support_distance_pct'] = support_distance
            
            if not pd.isna(self.df['nearest_resistance'].iloc[-1]):
                resistance_distance = (self.df['nearest_resistance'].iloc[-1] - current_price) / current_price * 100
                self.df['resistance_distance_pct'] = resistance_distance
        
        return {
            'support_levels': support_levels,
            'resistance_levels': resistance_levels
        }
    
    #
    # 2. FIBONACCI RETRACEMENT
    #
    
    def calculate_fibonacci_retracement(self, lookback=100):
        """
        Calculate Fibonacci retracement levels for the most recent significant trend
        
        Args:
            lookback: Number of periods to look back
            
        Returns:
            Dictionary with Fibonacci levels
        """
        # Fibonacci retracement levels
        fib_levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
        
        # Get subset of data
        lookback = min(lookback, len(self.df))
        recent_data = self.df.iloc[-lookback:].copy()
        
        # Find highest high and lowest low in the period
        highest_high = recent_data['High'].max()
        highest_high_idx = recent_data['High'].idxmax()
        
        lowest_low = recent_data['Low'].min()
        lowest_low_idx = recent_data['Low'].idxmin()
        
        # Determine if uptrend or downtrend based on which came first
        is_uptrend = lowest_low_idx < highest_high_idx
        
        # Calculate retracement levels
        if is_uptrend:
            # For uptrend: retrace from high to low
            price_range = highest_high - lowest_low
            levels = {level: highest_high - (price_range * level) for level in fib_levels}
            trend_direction = 'uptrend'
        else:
            # For downtrend: retrace from low to high
            price_range = highest_high - lowest_low
            levels = {level: lowest_low + (price_range * level) for level in fib_levels}
            trend_direction = 'downtrend'
        
        # Store key levels in DataFrame for the most recent candle
        for level_name, level_value in levels.items():
            self.df[f'fib_{level_name}'] = level_value
            
        # Store trend direction
        self.df['fib_trend'] = trend_direction
        
        # Identify nearest Fibonacci level
        current_price = self.df['Close'].iloc[-1]
        level_diffs = [(level_name, abs(level_value - current_price)) 
                      for level_name, level_value in levels.items()]
        nearest_level = min(level_diffs, key=lambda x: x[1])
        
        self.df['nearest_fib'] = nearest_level[0]
        self.df['nearest_fib_price'] = levels[nearest_level[0]]
        
        return {
            'trend': trend_direction,
            'levels': levels,
            'nearest_level': nearest_level[0]
        }
        
    #
    # 3. VOLUME ANALYSIS
    #
    
    def calculate_volume_ratio(self, periods=[10, 20, 50]):
        """
        Calculate volume relative to moving averages
        
        Args:
            periods: List of periods for volume moving averages
            
        Returns:
            Dictionary with volume ratio values
        """
        # Skip if Volume data is not available
        if 'Volume' not in self.df.columns:
            return {}
            
        # Calculate volume moving averages
        for period in periods:
            self.df[f'volume_ma{period}'] = self.df['Volume'].rolling(window=period).mean()
            
        # Calculate volume ratio (current volume / average volume)
        for period in periods:
            self.df[f'volume_ratio_{period}'] = self.df['Volume'] / self.df[f'volume_ma{period}']
            
        # Flag high and low volume (based on 20-period MA)
        self.df['high_volume'] = self.df['Volume'] > (self.df['volume_ma20'] * 1.5)  # 50% above average
        self.df['low_volume'] = self.df['Volume'] < (self.df['volume_ma20'] * 0.5)   # 50% below average
        
        # Volume increasing or decreasing (5-day volume trend)
        self.df['volume_ma5'] = self.df['Volume'].rolling(window=5).mean()
        self.df['volume_increasing'] = self.df['volume_ma5'] > self.df['volume_ma5'].shift(3)
        
        # Price/volume confirmation
        self.df['price_up_volume_up'] = (self.df['Close'] > self.df['Close'].shift(1)) & self.df['high_volume']
        self.df['price_down_volume_up'] = (self.df['Close'] < self.df['Close'].shift(1)) & self.df['high_volume']
        
        return {
            'volume_ratio_20': self.df['volume_ratio_20'],
            'high_volume': self.df['high_volume'],
            'volume_increasing': self.df['volume_increasing']
        }
    
    def calculate_obv(self):
        """
        Calculate On-Balance Volume
        
        Returns:
            Series with OBV values
        """
        # Skip if Volume data is not available
        if 'Volume' not in self.df.columns:
            return pd.Series()
            
        # Calculate price change direction
        price_change = self.df['Close'].diff()
        
        # Initialize OBV
        obv = pd.Series(0, index=self.df.index)
        
        # Calculate OBV values
        for i in range(1, len(self.df)):
            if price_change.iloc[i] > 0:
                obv.iloc[i] = obv.iloc[i-1] + self.df['Volume'].iloc[i]
            elif price_change.iloc[i] < 0:
                obv.iloc[i] = obv.iloc[i-1] - self.df['Volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        self.df['obv'] = obv
        
        # Calculate OBV EMA for trend
        self.df['obv_ema'] = self.df['obv'].ewm(span=20, adjust=False).mean()
        
        # OBV rising above its EMA is bullish
        self.df['obv_bullish'] = self.df['obv'] > self.df['obv_ema']
        
        # OBV divergence (price up, OBV down or vice versa)
        self.df['obv_bullish_divergence'] = (
            (self.df['Close'] < self.df['Close'].shift(1)) &
            (self.df['obv'] > self.df['obv'].shift(1))
        )
        
        self.df['obv_bearish_divergence'] = (
            (self.df['Close'] > self.df['Close'].shift(1)) &
            (self.df['obv'] < self.df['obv'].shift(1))
        )
        
        return self.df['obv']
    
    def calculate_vwap(self, reset_period='day'):
        """
        Calculate Volume Weighted Average Price
        
        Args:
            reset_period: When to reset the calculation ('day', 'week', or None for cumulative)
            
        Returns:
            Series with VWAP values
        """
        # Skip if Volume data is not available
        if 'Volume' not in self.df.columns:
            return pd.Series()
            
        # Calculate typical price
        self.df['typical_price'] = (self.df['High'] + self.df['Low'] + self.df['Close']) / 3
        
        # Calculate cumulative values
        if reset_period == 'day':
            # Group by date and calculate VWAP within each day
            self.df['date'] = pd.to_datetime(self.df.index).date
            grouped = self.df.groupby('date')
            
            vwap_list = []
            for _, group in grouped:
                # Calculate VWAP for this day
                tp_volume = group['typical_price'] * group['Volume']
                cumulative_tp_volume = tp_volume.cumsum()
                cumulative_volume = group['Volume'].cumsum()
                group_vwap = cumulative_tp_volume / cumulative_volume
                vwap_list.append(group_vwap)
            
            self.df['vwap'] = pd.concat(vwap_list)
        else:
            # Calculate cumulative VWAP without resetting
            tp_volume = self.df['typical_price'] * self.df['Volume']
            self.df['vwap'] = tp_volume.cumsum() / self.df['Volume'].cumsum()
        
        # Price above VWAP is bullish
        self.df['above_vwap'] = self.df['Close'] > self.df['vwap']
        
        return self.df['vwap']
    
    #
    # 4. MOVING AVERAGES
    #
    
    def calculate_sma(self, periods=[5, 10, 20, 50, 200]):
        """
        Calculate Simple Moving Average for specified periods
        
        Args:
            periods: List of periods to calculate SMAs for
            
        Returns:
            Dictionary with SMA values for each period
        """
        for period in periods:
            column_name = f'sma_{period}'
            self.df[column_name] = self.df['Close'].rolling(window=period).mean()
            
        # Calculate crossovers between key moving averages
        if all(f'sma_{p}' in self.df.columns for p in [20, 50]):
            self.df['sma_20_50_bullish_cross'] = (
                (self.df['sma_20'] > self.df['sma_50']) &
                (self.df['sma_20'].shift(1) <= self.df['sma_50'].shift(1))
            )
            
            self.df['sma_20_50_bearish_cross'] = (
                (self.df['sma_20'] < self.df['sma_50']) &
                (self.df['sma_20'].shift(1) >= self.df['sma_50'].shift(1))
            )
            
        # Price position relative to key moving averages
        if 'sma_200' in self.df.columns:
            self.df['price_above_sma200'] = self.df['Close'] > self.df['sma_200']
            
        return {f'sma_{period}': self.df[f'sma_{period}'] for period in periods}
    
    def calculate_ema(self, periods=[5, 12, 26, 50]):
        """
        Calculate Exponential Moving Average for specified periods
        
        Args:
            periods: List of periods to calculate EMAs for
            
        Returns:
            Dictionary with EMA values for each period
        """
        for period in periods:
            column_name = f'ema_{period}'
            self.df[column_name] = self.df['Close'].ewm(span=period, adjust=False).mean()
            
        # Calculate crossovers between key moving averages
        if all(f'ema_{p}' in self.df.columns for p in [12, 26]):
            self.df['ema_12_26_bullish_cross'] = (
                (self.df['ema_12'] > self.df['ema_26']) &
                (self.df['ema_12'].shift(1) <= self.df['ema_26'].shift(1))
            )
            
            self.df['ema_12_26_bearish_cross'] = (
                (self.df['ema_12'] < self.df['ema_26']) &
                (self.df['ema_12'].shift(1) >= self.df['ema_26'].shift(1))
            )
            
        return {f'ema_{period}': self.df[f'ema_{period}'] for period in periods}
    
    #
    # 5. RSI INDICATOR
    #
    
    def calculate_rsi(self, period=14):
        """
        Calculate Relative Strength Index
        
        Args:
            period: Period for RSI calculation (default: 14)
            
        Returns:
            Series with RSI values
        """
        # Calculate price changes
        delta = self.df['Close'].diff()
        
        # Separate gains (up) from losses (down)
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        
        # Calculate rolling averages
        avg_gain = up.rolling(period).mean()
        avg_loss = down.rolling(period).mean()
        
        # Calculate RS
        rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)  # Avoid division by zero
        
        # Calculate RSI
        self.df['rsi'] = 100 - (100 / (1 + rs))
        
        # Store RSI thresholds
        self.df['rsi_oversold'] = self.df['rsi'] < 30
        self.df['rsi_overbought'] = self.df['rsi'] > 70
        
        # Calculate price and RSI higher highs/lower lows for divergence
        self.df['price_higher_high'] = (
            (self.df['High'] > self.df['High'].shift(1)) &
            (self.df['High'].shift(1) > self.df['High'].shift(2))
        )
        
        self.df['price_lower_low'] = (
            (self.df['Low'] < self.df['Low'].shift(1)) &
            (self.df['Low'].shift(1) < self.df['Low'].shift(2))
        )
        
        self.df['rsi_higher_high'] = (
            (self.df['rsi'] > self.df['rsi'].shift(1)) &
            (self.df['rsi'].shift(1) > self.df['rsi'].shift(2))
        )
        
