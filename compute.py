"""
Enhanced Quantitative Analysis Engine
Combines technical indicators and candlestick pattern recognition

Version: 3.5.1 (Improved)
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
import uuid
import sys
import threading
from contextlib import asynccontextmanager
from typing import Dict, List, Union, Tuple, Any, Optional
from contextlib import asynccontextmanager

# Import dependencies (with assumption they are already installed)
# Rather than installing at runtime, fail with clear error messages
try:
    from aiogram import Bot, Dispatcher
except ImportError:
    logging.critical("Required dependency 'aiogram' not found. Please install it using: pip install aiogram")
    logging.critical("Exiting application due to missing dependencies")
    sys.exit(1)

try:
    from upstox_client.api_client import ApiClient
    from upstox_client.api.market_quote_api import MarketQuoteApi
    from upstox_client.api.history_api import HistoryApi
except ImportError:
    logging.critical("Required Upstox client dependencies not found. Please install them.")
    logging.critical("Exiting application due to missing dependencies")
    sys.exit(1)


# ===============================================================
# Custom Exceptions
# ===============================================================
# Add global lock and initialization flag

_INITIALIZATION_LOCK = threading.Lock()
_BOT_INITIALIZED = False
_LOGGING_INITIALIZED = False
_BOT_INSTANCE_UUID = str(uuid.uuid4())
_INIT_TIMESTAMP = None
_INIT_CALLER_INFO = []

# Global state tracking
_BOT_SETUP_COMPLETE = False     # First stage initialization
_BOT_RUNNING = False            # Second stage (scheduler running)


class TradingBotError(Exception):
    """Base exception for all trading bot errors"""
    pass

class DataFetchError(TradingBotError):
    """Exception raised for errors when fetching data"""
    def __init__(self, symbol, message="Failed to fetch data", cause=None):
        self.symbol = symbol
        self.cause = cause
        self.message = f"{message} for {symbol}: {cause}"
        super().__init__(self.message)

class PatternDetectionError(TradingBotError):
    """Exception raised for errors in pattern detection"""
    pass

class InvalidConfigurationError(TradingBotError):
    """Exception raised for invalid configuration"""
    pass

class APIConnectionError(TradingBotError):
    """Exception raised for API connection issues"""
    pass

class EmptyDataError(DataFetchError):
    """Exception raised when fetched data is empty"""
    def __init__(self, symbol, message="No data returned"):
        super().__init__(symbol, message)

def setup_logging(config):
    """
    Create a completely separate logger for each instance
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Logger instance
    """
    # Generate a unique logger name using the instance UUID
    instance_id = config.get('INSTANCE_UUID', str(uuid.uuid4()))[:8]
    logger_name = f'compute_{instance_id}'
    
    # Get a unique logger for this instance
    logger = logging.getLogger(logger_name)
    
    # If the logger already has handlers, it's already configured
    if logger.handlers:
        return logger
    
    # Get log directory
    log_dir = config.get('LOG_DIRECTORY', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Create a unique log file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'trading_bot_{timestamp}_{instance_id}.log')
    
    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Disable propagation to the root logger
    logger.propagate = False
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create handlers
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Set level and add handlers
    logger.setLevel(config.get('LOG_LEVEL', logging.INFO))
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
# ===============================================================
# Parameter Configuration
# ===============================================================

class TradingParameters:
    """Centralized configuration for all trading parameters and thresholds"""
    
    def __init__(self, config=None):
        """
        Initialize with parameters from config or use defaults
        
        Args:
            config: Optional configuration dictionary to override defaults
        """
        self.config = config or {}
        
        # Pattern detection thresholds
        self.PATTERN_THRESHOLDS = {
            # Doji thresholds
            "doji_body_threshold": self.config.get("doji_body_threshold", 0.1),
            
            # Marubozu thresholds
            "marubozu_shadow_threshold": self.config.get("marubozu_shadow_threshold", 0.05),
            "marubozu_body_pct": self.config.get("marubozu_body_pct", 0.95),
            
            # Spinning top thresholds
            "spinning_top_body_threshold": self.config.get("spinning_top_body_threshold", 0.25),
            "spinning_top_shadow_threshold": self.config.get("spinning_top_shadow_threshold", 0.35),
            
            # Hammer & shooting star thresholds
            "hammer_lower_shadow_ratio": self.config.get("hammer_lower_shadow_ratio", 2.0),
            "hammer_upper_shadow_threshold": self.config.get("hammer_upper_shadow_threshold", 0.1),
            
            # Paper umbrella thresholds
            "umbrella_lower_shadow_ratio": self.config.get("umbrella_lower_shadow_ratio", 2.0),
            "umbrella_upper_shadow_threshold": self.config.get("umbrella_upper_shadow_threshold", 0.1),
            
            # Engulfing pattern tolerance
            "engulfing_body_size_factor": self.config.get("engulfing_body_size_factor", 1.1),
            
            # Harami pattern thresholds  
            "harami_body_size_ratio": self.config.get("harami_body_size_ratio", 0.6),
            
            # Star pattern thresholds
            "star_body_size_threshold": self.config.get("star_body_size_threshold", 0.3),
            "star_body_size_factor": self.config.get("star_body_size_factor", 0.6),
        }
        
        # Technical indicator parameters
        self.INDICATOR_PARAMS = {
            # RSI parameters
            "rsi_period": self.config.get("rsi_period", 14),
            "rsi_oversold": self.config.get("rsi_oversold", 30),
            "rsi_overbought": self.config.get("rsi_overbought", 70),
            
            # MACD parameters
            "macd_fast": self.config.get("macd_fast", 12),
            "macd_slow": self.config.get("macd_slow", 26),
            "macd_signal": self.config.get("macd_signal", 9),
            
            # Bollinger Bands parameters
            "bb_period": self.config.get("bb_period", 20),
            "bb_std_dev": self.config.get("bb_std_dev", 2),
            
            # ATR parameters
            "atr_period": self.config.get("atr_period", 14),
            "atr_multiplier": self.config.get("atr_multiplier", 2),
            
            # Stochastic parameters
            "stoch_k_period": self.config.get("stoch_k_period", 14),
            "stoch_d_period": self.config.get("stoch_d_period", 3),
            "stoch_slowing": self.config.get("stoch_slowing", 3),
            "stoch_oversold": self.config.get("stoch_oversold", 20),
            "stoch_overbought": self.config.get("stoch_overbought", 80),
            
            # ADX parameters
            "adx_period": self.config.get("adx_period", 14),
            "adx_threshold": self.config.get("adx_threshold", 25),
            
            # Supertrend parameters
            "supertrend_period": self.config.get("supertrend_period", 10),
            "supertrend_multiplier": self.config.get("supertrend_multiplier", 3),
            
            # Support/Resistance parameters
            "sr_lookback": self.config.get("sr_lookback", 100),
            "sr_price_tolerance": self.config.get("sr_price_tolerance", 0.02),
            "sr_window_size": self.config.get("sr_window_size", 5),
            
            # Fibonacci parameters
            "fib_lookback": self.config.get("fib_lookback", 100),
            
            # Moving Average parameters
            "sma_periods": self.config.get("sma_periods", [5, 10, 20, 50, 200]),
            "ema_periods": self.config.get("ema_periods", [5, 12, 26, 50]),
            
            # Volume parameters
            "volume_ma_periods": self.config.get("volume_ma_periods", [10, 20, 50]),
            "high_volume_threshold": self.config.get("high_volume_threshold", 1.5),
            "low_volume_threshold": self.config.get("low_volume_threshold", 0.5),
        }
        
        # Signal generation parameters
        self.SIGNAL_PARAMS = {
            # Minimum RRR threshold for valid signals
            "min_rrr": self.config.get("min_rrr", 1.5),
            
            # ATR multipliers for targets and stops
            "target_multiplier": self.config.get("target_multiplier", 1.5),
            "stop_multiplier": self.config.get("stop_multiplier", 1.0),
            
            # Minimum signal strength threshold
            "min_signal_strength": self.config.get("min_signal_strength", 3),
            
            # Checklist thresholds
            "checklist_high_confidence_threshold": self.config.get("checklist_high_confidence_threshold", 4),
            "checklist_medium_confidence_threshold": self.config.get("checklist_medium_confidence_threshold", 3),
            "checklist_low_confidence_threshold": self.config.get("checklist_low_confidence_threshold", 2),
            
            # Pattern strength factors
            "pattern_strength_weights": self.config.get("pattern_strength_weights", {
                "bullish_marubozu": 3, "bearish_marubozu": 3,
                "hammer": 3, "hanging_man": 3, "shooting_star": 3,
                "bullish_engulfing": 4, "bearish_engulfing": 4,
                "bullish_harami": 3, "bearish_harami": 3,
                "piercing_pattern": 3, "dark_cloud_cover": 3,
                "morning_star": 4, "evening_star": 4,
                "doji": 1, "spinning_tops": 1
            }),
            
            # Support & Resistance significance thresholds
            "sr_near_threshold_pct": self.config.get("sr_near_threshold_pct", 2.0),
            "sr_very_near_threshold_pct": self.config.get("sr_very_near_threshold_pct", 1.0),
        }
        
        # Chart pattern detection parameters
        self.CHART_PATTERN_PARAMS = {
            # Double top/bottom parameters
            "double_pattern_tolerance": self.config.get("double_pattern_tolerance", 0.03),
            "double_pattern_lookback": self.config.get("double_pattern_lookback", 50),
            
            # Triple top/bottom parameters
            "triple_pattern_tolerance": self.config.get("triple_pattern_tolerance", 0.03),
            "triple_pattern_lookback": self.config.get("triple_pattern_lookback", 100),
            
            # Trading range parameters
            "range_lookback": self.config.get("range_lookback", 60),
            "range_tolerance": self.config.get("range_tolerance", 0.05),
            "range_min_touches": self.config.get("range_min_touches", 4),
            
            # Flag pattern parameters
            "flag_lookback": self.config.get("flag_lookback", 30),
            "flag_pole_threshold": self.config.get("flag_pole_threshold", 0.15),
            "flag_consolidation_threshold": self.config.get("flag_consolidation_threshold", 0.05),
            "flag_max_bars": self.config.get("flag_max_bars", 15),
        }
        
    def get_pattern_param(self, param_name):
        """Get a pattern detection parameter"""
        return self.PATTERN_THRESHOLDS.get(param_name)
        
    def get_indicator_param(self, param_name):
        """Get a technical indicator parameter"""
        return self.INDICATOR_PARAMS.get(param_name)
        
    def get_signal_param(self, param_name):
        """Get a signal generation parameter"""
        return self.SIGNAL_PARAMS.get(param_name)
        
    def get_chart_pattern_param(self, param_name):
        """Get a chart pattern detection parameter"""
        return self.CHART_PATTERN_PARAMS.get(param_name)


    # ===============================================================
    # Utility Functions
    # ===============================================================
# ===============================================================
# Helper Functions
# ===============================================================
# ===============================================================
# Telegram Integration
# ===============================================================

@asynccontextmanager
async def get_telegram_bot(token):
    """
    Context manager for Telegram bot to ensure proper resource management
    
    Args:
        token: Telegram bot token
        
    Yields:
        Bot instance
    """
    bot = Bot(token=token)
    try:
        yield bot
    finally:
        await bot.session.close()

def setup_logging(config):
    """
    Setup logging configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Logger instance
    """
    log_dir = config.get('LOG_DIRECTORY', 'logs')
    
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup logging
    log_level = config.get('LOG_LEVEL', logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Setup file handler
    log_file = os.path.join(log_dir, f'candlestick_bot_{datetime.datetime.now().strftime("%Y%m%d")}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Setup logger
    logger = logging.getLogger('compute')
    logger.setLevel(log_level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Prevent log messages from being propagated to the root logger
    logger.propagate = False
    
    return logger


async def send_telegram_message(message, config, logger, retry_attempts=5):
    """
    Send message to Telegram with retry mechanism and proper resource management
    
    Args:
        message: The message text to send
        config: Configuration with Telegram credentials
        logger: Logger instance
        retry_attempts: Number of retry attempts
        
    Returns:
        True if message was sent successfully, False otherwise
    """
    if not config.get('ENABLE_TELEGRAM_ALERTS', False):
        return False
    
    if not config.get('TELEGRAM_BOT_TOKEN', '') or not config.get('TELEGRAM_CHAT_ID', ''):
        logger.error("Telegram credentials are missing")
        return False
        
    delay = 1  # Initial delay in seconds
    
    for attempt in range(retry_attempts):
        try:
            async with get_telegram_bot(config.get('TELEGRAM_BOT_TOKEN', '')) as bot:
                # Use MarkdownV2 for formatted messages
                await bot.send_message(
                    chat_id=config.get('TELEGRAM_CHAT_ID', ''), 
                    text=message, 
                    parse_mode='MarkdownV2'
                )
                logger.info(f"Successfully sent telegram message (attempt {attempt+1})")
                return True
        except Exception as e:
            if "Too Many Requests" in str(e):
                retry_after = int(str(e).split("retry after ")[-1].split()[0]) if "retry after" in str(e) else delay
                logger.error(f"Error sending Telegram message: {e}. Retrying in {retry_after} seconds.")
                await asyncio.sleep(retry_after)
            else:
                logger.error(f"Error sending Telegram message: {e}. Retrying in {delay} seconds.")
                await asyncio.sleep(delay)
                delay *= 2  # Exponential backoff
    
    logger.error(f"Failed to send Telegram message after {retry_attempts} attempts")
    return False


def escape_telegram_markdown(text):
    """
    Escape special characters for Telegram MarkdownV2 format
    
    Args:
        text: Text to escape
        
    Returns:
        Escaped text safe for Telegram MarkdownV2
    """
    # Characters that need to be escaped in MarkdownV2
    escape_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
    
    # Escape each character with a backslash
    for char in escape_chars:
        text = text.replace(char, '\\' + char)
    
    return text


# ===============================================================
# System and Connection Test Functions
# ===============================================================

async def initialize_and_test(config):
    """
    Initialize the system and test connections
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (logger, upstox_ok, telegram_ok)
    """
    # Initialize logger
    logger = setup_logging(config)
    
    # Test connections
    upstox_ok = test_upstox_connection(config, logger)
    telegram_ok = await test_telegram_connection(config, logger) if config.get('ENABLE_TELEGRAM_ALERTS', False) else False
    
    return logger, upstox_ok, telegram_ok


async def test_telegram_connection(config, logger):
    """
    Test connection to Telegram API
    
    Args:
        config: Configuration dictionary with Telegram credentials
        logger: Logger instance
    
    Returns:
        True if connection is successful, False otherwise
    """
    logger.info("Testing Telegram API connection...")
    
    if not config.get('ENABLE_TELEGRAM_ALERTS', False):
        logger.info("❌ Telegram notifications are disabled in config")
        return False
    
    try:
        # Escape the test message properly for MarkdownV2
        test_message = escape_telegram_markdown("🔍 Test Message - Candlestick Pattern Bot connection test successful!")
        result = await send_telegram_message(test_message, config, logger)
        
        if result:
            logger.info("✅ Successfully sent test message to Telegram")
            return True
        else:
            logger.error("❌ Failed to send test message to Telegram")
            return False
    except Exception as e:
        logger.error(f"❌ Error connecting to Telegram API: {str(e)}")
        return False
   

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

    def initialize_upstox(config, logger):
        """
        Initialize connection to Upstox API
        
        Args:
            config: Configuration dictionary with API credentials
            logger: Logger instance
        
        Returns:
            Tuple of (MarketQuoteApi, ApiClient) if successful
            Raises APIConnectionError if initialization fails
        """
        try:
            api_client = ApiClient()
            api_client.configuration.access_token = config.get('UPSTOX_ACCESS_TOKEN', '')
            market_api = MarketQuoteApi(api_client)
            logger.info("✅ Successfully initialized Upstox API client")
            return market_api, api_client
        except Exception as e:
            logger.error(f"Error initializing Upstox API client: {e}")
            raise APIConnectionError(f"Failed to connect to Upstox API: {str(e)}")

    def fetch_ohlcv_data(market_api, symbol, start_date, end_date, interval="day", api_version="2.0", logger=None):
        """
        Fetch historical OHLC data for a given symbol using the Upstox API.
        
        Args:
            market_api: The initialized Upstox API client
            symbol: The instrument symbol/key to fetch data for
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Time interval (1minute, 30minute, day, week, month)
            api_version: API version to use
            logger: Logger instance
            
        Returns:
            Pandas DataFrame with OHLCV data
            
        Raises:
            DataFetchError: For API errors
            EmptyDataError: When no data is returned
            ValueError: For invalid parameters
        """
        try:
            # Validate dates
            try:
                datetime.datetime.strptime(start_date, "%Y-%m-%d")
                datetime.datetime.strptime(end_date, "%Y-%m-%d")
            except ValueError as e:
                raise ValueError(f"Invalid date format: {e}")
            
            # Validate interval
            valid_intervals = ['1minute', '5minute', '30minute', 'day', 'week', 'month']
            if interval not in valid_intervals:
                raise ValueError(f"Invalid interval: {interval}. Must be one of {valid_intervals}")
            
            if logger:
                logger.info(f"Fetching historical data for {symbol} from {start_date} to {end_date} with {interval} interval")
            
            # Create a HistoryApi instance
            history_api = HistoryApi(market_api.api_client)
            
            # Implement retries for API calls
            max_retries = 3
            retry_delay = 2  # seconds
            
            last_exception = None
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
                        
                        # Check if DataFrame is empty
                        if df.empty:
                            raise EmptyDataError(symbol, "Empty dataset returned by API")
                        
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
                            if logger:
                                logger.warning(f"Retrieved only {len(df)} candles for {symbol}, which may be insufficient for reliable pattern detection")
                        
                        if logger:
                            logger.info(f"Successfully fetched {len(df)} candles for {symbol}")
                        return df
                    else:
                        err_msg = "No candle data returned by API"
                        if hasattr(response, 'status'):
                            err_msg += f" (API status: {response.status})"
                        
                        last_exception = DataFetchError(symbol, err_msg)
                        
                        if logger:
                            logger.error(f"No candle data returned for {symbol}")
                            if hasattr(response, 'status'):
                                logger.error(f"API status: {response.status}")
                            if hasattr(response, 'data'):
                                logger.error(f"Response data type: {type(response.data)}")
                        
                        if attempt < max_retries - 1:
                            if logger:
                                logger.info(f"Retrying fetch for {symbol} (attempt {attempt+1}/{max_retries})...")
                            time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                        else:
                            raise last_exception
                except Exception as e:
                    last_exception = DataFetchError(symbol, "API error", str(e))
                    if logger:
                        logger.error(f"Error in attempt {attempt+1}/{max_retries} for {symbol}: {e}")
                    
                    if attempt < max_retries - 1:
                        if logger:
                            logger.info(f"Retrying fetch for {symbol}...")
                        time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                    else:
                        if logger:
                            logger.error(traceback.format_exc())
                        raise last_exception
            
            # If we've exhausted retries
            if last_exception:
                raise last_exception
            
            # This should never happen, but as a fallback
            raise DataFetchError(symbol, "Failed to fetch data after all retry attempts")
                    
        except (DataFetchError, EmptyDataError) as e:
            # Re-raise these as they're already our custom exceptions
            raise
        except Exception as e:
            if logger:
                logger.error(f"Error fetching historical OHLC data: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
            raise DataFetchError(symbol, "Unexpected error", str(e))


    # ===============================================================
    # Candlestick Pattern Recognition
    # ===============================================================

    class CandlestickPatterns:
        """Complete candlestick pattern detection with precise validation criteria"""
        
        def __init__(self, df, params=None):
            """
            Initialize with DataFrame containing OHLCV data
            
            Args:
                df: DataFrame with OHLCV data
                params: Optional TradingParameters instance for pattern thresholds
            """
            # Ensure we have a valid DataFrame
            if df is None or len(df) == 0:
                raise ValueError("Empty DataFrame provided")
                
            # Make a deep copy to avoid modifying the original
            self.df = df.copy()
            
            # Initialize parameters (use defaults if not provided)
            self.params = params or TradingParameters()
            
            # Calculate body sizes and shadows for all candles
            self._calculate_candle_dimensions()
            
            # Calculate trends for context
            self._calculate_trend_context()
        
        def _calculate_candle_dimensions(self):
            """
            Calculate candle body, upper shadow, and lower shadow sizes using vectorized operations
            """
            # Get body size and direction
            self.df['body_size'] = abs(self.df['Close'] - self.df['Open'])
            self.df['candle_range'] = self.df['High'] - self.df['Low']
            # Avoid division by zero with where
            self.df['body_pct'] = np.where(
                self.df['candle_range'] > 0,
                self.df['body_size'] / self.df['candle_range'],
                0
            )
            self.df['is_bullish'] = self.df['Close'] > self.df['Open']
            
            # Calculate shadows using vectorized numpy operations
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
            
            # Calculate shadow percentages relative to range (avoid division by zero)
            self.df['upper_shadow_pct'] = np.where(
                self.df['candle_range'] > 0,
                self.df['upper_shadow'] / self.df['candle_range'],
                0
            )
            self.df['lower_shadow_pct'] = np.where(
                self.df['candle_range'] > 0,
                self.df['lower_shadow'] / self.df['candle_range'],
                0
            )
            
            # Handle volume analysis
            if 'Volume' in self.df.columns:
                # Calculate volume moving average
                self.df['volume_ma20'] = self.df['Volume'].rolling(window=20).mean()
                
                # Flag high volume candles (50% above average)
                high_volume_threshold = self.params.get_indicator_param('high_volume_threshold')
                self.df['high_volume'] = self.df['Volume'] > (self.df['volume_ma20'] * high_volume_threshold)
                
                # Calculate volume ratio to average (avoid division by zero)
                self.df['volume_ratio'] = np.where(
                    self.df['volume_ma20'] > 0,
                    self.df['Volume'] / self.df['volume_ma20'],
                    0
                )
        
        def _calculate_trend_context(self):
            """
            Calculate trend indicators for pattern context with volume confirmation using vectorized operations
            """
            # Short-term trend (5-day)
            self.df['ma5'] = self.df['Close'].rolling(window=5).mean()
            
            # Medium-term trend (20-day)
            self.df['ma20'] = self.df['Close'].rolling(window=20).mean()
            
            # Basic trend determination (vectorized)
            price_uptrend = (self.df['ma5'] > self.df['ma5'].shift(3)) & (self.df['Close'] > self.df['ma20'])
            price_downtrend = (self.df['ma5'] < self.df['ma5'].shift(3)) & (self.df['Close'] < self.df['ma20'])
            
            # Volume confirmation if available
            if 'Volume' in self.df.columns:
                # Calculate volume trend (increasing or decreasing volume)
                self.df['volume_ma5'] = self.df['Volume'].rolling(window=5).mean()
                volume_increasing = self.df['volume_ma5'] > self.df['volume_ma5'].shift(3)
                
                # Strong uptrend: Price rising with increasing volume (vectorized condition)
                self.df['uptrend'] = price_uptrend & (
                    volume_increasing | 
                    # Or high volume on up days
                    (self.df['high_volume'] & self.df['is_bullish'])
                )
                
                # Strong downtrend: Price falling with increasing volume (vectorized condition)
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
            Detect Marubozu patterns (candles with no shadows) using vectorized operations
            
            Returns:
                Dictionary with detected patterns and their indices
            """
            # Ensure we have data
            if len(self.df) == 0:
                return {'marubozu': pd.Series(dtype=bool), 
                        'bullish_marubozu': pd.Series(dtype=bool), 
                        'bearish_marubozu': pd.Series(dtype=bool)}
                
            # Get parameters from config
            shadow_threshold = self.params.get_pattern_param('marubozu_shadow_threshold')
            body_pct_threshold = self.params.get_pattern_param('marubozu_body_pct')
            
            # Detect Bullish Marubozu (vectorized)
            bullish_marubozu = (
                (self.df['is_bullish']) &
                (self.df['upper_shadow_pct'] <= shadow_threshold) &
                (self.df['lower_shadow_pct'] <= shadow_threshold) &
                (self.df['body_pct'] >= body_pct_threshold)  # Body is at least 95% of range
            )
            
            # Detect Bearish Marubozu (vectorized)
            bearish_marubozu = (
                (~self.df['is_bullish']) &
                (self.df['upper_shadow_pct'] <= shadow_threshold) &
                (self.df['lower_shadow_pct'] <= shadow_threshold) &
                (self.df['body_pct'] >= body_pct_threshold)  # Body is at least 95% of range
            )
            
            # Combine results
            return {
                'marubozu': bullish_marubozu | bearish_marubozu,
                'bullish_marubozu': bullish_marubozu,
                'bearish_marubozu': bearish_marubozu
            }
        
        def detect_doji(self):
            """
            Detect Doji patterns (candles with small bodies and shadows) using vectorized operations
            
            Returns:
                Series with True at indices where Doji patterns are detected
            """
            # Ensure we have data
            if len(self.df) == 0:
                return pd.Series(False, index=self.df.index)
                
            # Get parameter from config
            body_threshold = self.params.get_pattern_param('doji_body_threshold')
            
            # Doji criteria: open and close are virtually equal (vectorized)
            doji = (
                (self.df['body_pct'] <= body_threshold) &
                (self.df['candle_range'] > 0)  # Ensure there is some trading range
            )
            
            return doji
        
        def detect_spinning_tops(self):
            """
            Detect Spinning Top patterns (small bodies, long shadows) using vectorized operations
            
            Returns:
                Series with True at indices where Spinning Tops are detected
            """
            # Ensure we have data
            if len(self.df) == 0:
                return pd.Series(False, index=self.df.index)
                
            # Get parameters from config
            body_threshold = self.params.get_pattern_param('spinning_top_body_threshold')
            shadow_threshold = self.params.get_pattern_param('spinning_top_shadow_threshold')
            
            # Spinning Top criteria (vectorized)
            spinning_tops = (
                (self.df['body_pct'] <= body_threshold) &
                (self.df['upper_shadow_pct'] >= shadow_threshold) &
                (self.df['lower_shadow_pct'] >= shadow_threshold) &
                (self.df['candle_range'] > 0)  # Ensure there is some trading range
            )
            
            return spinning_tops
        
        def detect_paper_umbrella(self):
            """
            Detect Paper Umbrella patterns (small body at top, long lower shadow) using vectorized operations
            
            Returns:
                Series with True at indices where Paper Umbrella patterns are detected
            """
            # Ensure we have data
            if len(self.df) == 0:
                return pd.Series(False, index=self.df.index)
                
            # Get parameters from config
            lower_shadow_ratio = self.params.get_pattern_param('umbrella_lower_shadow_ratio')
            upper_shadow_threshold = self.params.get_pattern_param('umbrella_upper_shadow_threshold')
            
            # Paper Umbrella criteria (vectorized)
            paper_umbrella = (
                (self.df['lower_shadow'] >= (self.df['body_size'] * lower_shadow_ratio)) &
                (self.df['upper_shadow_pct'] <= upper_shadow_threshold) &
                (self.df['candle_range'] > 0)  # Ensure there is some trading range
            )
            
            return paper_umbrella
        
        def detect_hammer(self):
            """
            Detect Hammer patterns (bullish reversal with small body at top, long lower shadow) using vectorized operations
            
            Returns:
                Series with True at indices where Hammer patterns are detected
            """
            # Ensure we have data and sufficient history
            if len(self.df) < 5:
                return pd.Series(False, index=self.df.index)
                
            # Get parameters from config
            lower_shadow_ratio = self.params.get_pattern_param('hammer_lower_shadow_ratio')
            upper_shadow_threshold = self.params.get_pattern_param('hammer_upper_shadow_threshold')
            
            # Detect hammer candle structure (vectorized)
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
            Detect Hanging Man patterns (bearish reversal with small body at top, long lower shadow) using vectorized operations
            
            Returns:
                Series with True at indices where Hanging Man patterns are detected
            """
            # Ensure we have data and sufficient history
            if len(self.df) < 5:
                return pd.Series(False, index=self.df.index)
                
            # Get parameters from config - using same as hammer since structure is similar
            lower_shadow_ratio = self.params.get_pattern_param('hammer_lower_shadow_ratio')
            upper_shadow_threshold = self.params.get_pattern_param('hammer_upper_shadow_threshold')
            
            # Detect hanging man candle structure (same as hammer, vectorized)
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
            Detect Shooting Star patterns (bearish reversal with small body at bottom, long upper shadow) using vectorized operations
            
            Returns:
                Series with True at indices where Shooting Star patterns are detected
            """
            # Ensure we have data and sufficient history
            if len(self.df) < 5:
                return pd.Series(False, index=self.df.index)
                
            # Get parameters from config
            # Using same ratio as hammer but for upper shadow
            upper_shadow_ratio = self.params.get_pattern_param('hammer_lower_shadow_ratio')  
            lower_shadow_threshold = self.params.get_pattern_param('hammer_upper_shadow_threshold')
            
            # Detect shooting star candle structure (vectorized)
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
            Detect Engulfing patterns (two-candle reversal pattern) using vectorized operations
            
            Returns:
                Dictionary with bullish_engulfing and bearish_engulfing Series
            """
            # Initialize result Series
            bullish_engulfing = pd.Series(False, index=self.df.index)
            bearish_engulfing = pd.Series(False, index=self.df.index)
            
            # We need at least 2 candles to detect engulfing patterns
            if len(self.df) < 6:  # Need at least 6 candles (5 for trend context + current)
                return {'bullish_engulfing': bullish_engulfing, 'bearish_engulfing': bearish_engulfing}
            
            # Create shifted dataframes for comparison
            df_prev = self.df.shift(1)
            
            # Bullish Engulfing (vectorized)
            bullish_engulfing_condition = (
                ~self.df['is_bullish'].shift(1) &  # Previous candle is bearish
                self.df['is_bullish'] &            # Current candle is bullish
                (self.df['Open'] <= df_prev['Close']) &  # Current open below previous close
                (self.df['Close'] >= df_prev['Open'])    # Current close above previous open
            )
            
            # Add trend context (at least 5 periods of history)
            downtrend_context = self.df['downtrend'].shift(1)
            bullish_engulfing_valid = bullish_engulfing_condition & downtrend_context
            
            # Bearish Engulfing (vectorized)
            bearish_engulfing_condition = (
                self.df['is_bullish'].shift(1) &    # Previous candle is bullish
                ~self.df['is_bullish'] &            # Current candle is bearish
                (self.df['Open'] >= df_prev['Close']) &  # Current open above previous close
                (self.df['Close'] <= df_prev['Open'])    # Current close below previous open
            )
            
            # Add trend context
            uptrend_context = self.df['uptrend'].shift(1)
            bearish_engulfing_valid = bearish_engulfing_condition & uptrend_context
            
            return {
                'bullish_engulfing': bullish_engulfing_valid,
                'bearish_engulfing': bearish_engulfing_valid
            }
        
        def detect_harami(self):
            """
            Detect Harami patterns (two-candle pattern where second candle is contained within first) using vectorized operations
            
            Returns:
                Dictionary with bullish_harami and bearish_harami Series
            """
            # Initialize result Series with index from self.df
            bullish_harami = pd.Series(False, index=self.df.index)
            bearish_harami = pd.Series(False, index=self.df.index)
            
            # We need at least 2 candles to detect harami patterns
            if len(self.df) < 6:  # Need at least 6 candles (5 for trend context + current)
                return {'bullish_harami': bullish_harami, 'bearish_harami': bearish_harami}
            
            # Create shifted dataframes for comparison
            df_prev = self.df.shift(1)
            
            # Get harami body size ratio parameter
            harami_body_size_ratio = self.params.get_pattern_param('harami_body_size_ratio')
            
            # Bullish Harami (vectorized)
            bullish_harami_condition = (
                ~df_prev['is_bullish'] &  # Previous candle is bearish
                self.df['is_bullish'] &   # Current candle is bullish
                (self.df['Open'] > df_prev['Close']) &  # Current open inside previous body
                (self.df['Open'] < df_prev['Open']) &
                (self.df['Close'] > df_prev['Close']) &  # Current close inside previous body
                (self.df['Close'] < df_prev['Open']) &
                (self.df['body_size'] < df_prev['body_size'] * harami_body_size_ratio)  # Current body smaller than previous
            )
            
            # Add trend context
            downtrend_context = self.df['downtrend'].shift(1)
            bullish_harami_valid = bullish_harami_condition & downtrend_context
            
            # Bearish Harami (vectorized)
            bearish_harami_condition = (
                df_prev['is_bullish'] &    # Previous candle is bullish
                ~self.df['is_bullish'] &   # Current candle is bearish
                (self.df['Open'] < df_prev['Close']) &  # Current open inside previous body
                (self.df['Open'] > df_prev['Open']) &
                (self.df['Close'] < df_prev['Close']) &  # Current close inside previous body
                (self.df['Close'] > df_prev['Open']) &
                (self.df['body_size'] < df_prev['body_size'] * harami_body_size_ratio)  # Current body smaller than previous
            )
            
            # Add trend context
            uptrend_context = self.df['uptrend'].shift(1)
            bearish_harami_valid = bearish_harami_condition & uptrend_context
            
            return {
                'bullish_harami': bullish_harami_valid,
                'bearish_harami': bearish_harami_valid
            }
        
        def detect_piercing_pattern(self):
            """
            Detect Piercing Pattern (bullish reversal pattern) using vectorized operations
            
            Returns:
                Series with True at indices where Piercing Pattern is detected
            """
            # Initialize result Series with index from self.df
            piercing_pattern = pd.Series(False, index=self.df.index)
            
            # We need at least 2 candles to detect piercing patterns
            if len(self.df) < 6:  # Need at least 6 candles (5 for trend context + current)
                return piercing_pattern
            
            # Create shifted dataframes for comparison
            df_prev = self.df.shift(1)
            
            # Piercing Pattern (vectorized)
            piercing_condition = (
                ~df_prev['is_bullish'] &  # Previous candle is bearish
                self.df['is_bullish'] &   # Current candle is bullish
                (df_prev['body_size'] / df_prev['candle_range'] > 0.6) &  # Previous is a long candle
                (self.df['Open'] < df_prev['Low']) &  # Gap down opening
                (self.df['Close'] > (df_prev['Open'] + df_prev['Close']) / 2) &  # Close above midpoint
                (self.df['Close'] < df_prev['Open'])  # Not a complete bullish engulfing
            )
            
            # Add trend context
            downtrend_context = self.df['downtrend'].shift(1)
            piercing_pattern_valid = piercing_condition & downtrend_context
            
            return piercing_pattern_valid
        
        def detect_dark_cloud_cover(self):
            """
            Detect Dark Cloud Cover (bearish reversal pattern) using vectorized operations
            
            Returns:
                Series with True at indices where Dark Cloud Cover is detected
            """
            # Initialize result Series with index from self.df
            dark_cloud_cover = pd.Series(False, index=self.df.index)
            
            # We need at least 2 candles to detect dark cloud cover
            if len(self.df) < 6:  # Need at least 6 candles (5 for trend context + current)
                return dark_cloud_cover
            
            # Create shifted dataframes for comparison
            df_prev = self.df.shift(1)
            
            # Dark Cloud Cover (vectorized)
            dark_cloud_condition = (
                df_prev['is_bullish'] &    # Previous candle is bullish
                ~self.df['is_bullish'] &   # Current candle is bearish
                (df_prev['body_size'] / df_prev['candle_range'] > 0.6) &  # Previous is a long candle
                (self.df['Open'] > df_prev['High']) &  # Gap up opening
                (self.df['Close'] < (df_prev['Open'] + df_prev['Close']) / 2) &  # Close below midpoint
                (self.df['Close'] > df_prev['Open'])  # Not a complete bearish engulfing
            )
            
            # Add trend context
            uptrend_context = self.df['uptrend'].shift(1)
            dark_cloud_cover_valid = dark_cloud_condition & uptrend_context
            
            return dark_cloud_cover_valid
        
        def detect_morning_star(self):
            """
            Detect Morning Star (bullish reversal pattern)
            
            Note: This pattern is complex and retains some loop logic but is optimized
            
            Returns:
                Series with True at indices where Morning Star is detected
            """
            # Initialize result Series with index from self.df
            morning_star = pd.Series(False, index=self.df.index)
            
            # We need at least 3 candles to detect morning star
            if len(self.df) < 7:  # Need at least 7 candles (5 for trend context + 2 previous)
                return morning_star
            
            # Get star body size threshold
            star_body_size_threshold = self.params.get_pattern_param('star_body_size_threshold')
            
            # Create shifted dataframes for comparison
            df_prev1 = self.df.shift(1)  # second day
            df_prev2 = self.df.shift(2)  # first day
            
            # Morning Star conditions (vectorized where possible)
            condition = (
                ~df_prev2['is_bullish'] &  # First candle is bearish
                self.df['is_bullish'] &    # Third candle is bullish
                (df_prev2['body_size'] / df_prev2['candle_range'] > 0.6) &  # First is a long candle
                (df_prev1['body_size'] / df_prev1['candle_range'] < star_body_size_threshold) &  # Second has a small body
                (df_prev1['Close'] < df_prev2['Close']) &  # Second gaps down from first
                (self.df['Open'] > df_prev1['Close']) &  # Third opens above second close
                (self.df['Close'] > (df_prev2['Open'] + df_prev2['Close']) / 2)  # Third closes above midpoint of first
            )
            
            # Add trend context
            downtrend_context = self.df['downtrend'].shift(2)
            morning_star_valid = condition & downtrend_context
            
            return morning_star_valid
        
        def detect_evening_star(self):
            """
            Detect Evening Star (bearish reversal pattern)
            
            Note: This pattern is complex and retains some loop logic but is optimized
            
            Returns:
                Series with True at indices where Evening Star is detected
            """
            # Initialize result Series with index from self.df
            evening_star = pd.Series(False, index=self.df.index)
            
            # We need at least 3 candles to detect evening star
            if len(self.df) < 7:  # Need at least 7 candles (5 for trend context + 2 previous)
                return evening_star
            
            # Get star body size threshold
            star_body_size_threshold = self.params.get_pattern_param('star_body_size_threshold')
            
            # Create shifted dataframes for comparison
            df_prev1 = self.df.shift(1)  # second day
            df_prev2 = self.df.shift(2)  # first day
            
            # Evening Star conditions (vectorized where possible)
            condition = (
                df_prev2['is_bullish'] &    # First candle is bullish
                ~self.df['is_bullish'] &    # Third candle is bearish
                (df_prev2['body_size'] / df_prev2['candle_range'] > 0.6) &  # First is a long candle
                (df_prev1['body_size'] / df_prev1['candle_range'] < star_body_size_threshold) &  # Second has a small body
                (df_prev1['Close'] > df_prev2['Close']) &  # Second gaps up from first
                (self.df['Open'] < df_prev1['Close']) &  # Third opens below second close
                (self.df['Close'] < (df_prev2['Open'] + df_prev2['Close']) / 2)  # Third closes below midpoint of first
            )
            
            # Add trend context
            uptrend_context = self.df['uptrend'].shift(2)
            evening_star_valid = condition & uptrend_context
            
            return evening_star_valid
        
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
            
            # Get pattern strength weights from params
            pattern_strength_weights = self.params.get_signal_param('pattern_strength_weights')
            
            # Pattern classifications with signal type and strength
            pattern_signals = {
                # Bullish patterns (Buy signals)
                'bullish_marubozu': {'signal': 'BUY', 'strength': pattern_strength_weights.get('bullish_marubozu', 3)},
                'hammer': {'signal': 'BUY', 'strength': pattern_strength_weights.get('hammer', 3)},
                'bullish_engulfing': {'signal': 'BUY', 'strength': pattern_strength_weights.get('bullish_engulfing', 4)},
                'bullish_harami': {'signal': 'BUY', 'strength': pattern_strength_weights.get('bullish_harami', 3)},
                'piercing_pattern': {'signal': 'BUY', 'strength': pattern_strength_weights.get('piercing_pattern', 3)},
                'morning_star': {'signal': 'BUY', 'strength': pattern_strength_weights.get('morning_star', 4)},
                
                # Bearish patterns (Sell signals)
                'bearish_marubozu': {'signal': 'SELL', 'strength': pattern_strength_weights.get('bearish_marubozu', 3)},
                'hanging_man': {'signal': 'SELL', 'strength': pattern_strength_weights.get('hanging_man', 3)},
                'shooting_star': {'signal': 'SELL', 'strength': pattern_strength_weights.get('shooting_star', 3)},
                'bearish_engulfing': {'signal': 'SELL', 'strength': pattern_strength_weights.get('bearish_engulfing', 4)},
                'bearish_harami': {'signal': 'SELL', 'strength': pattern_strength_weights.get('bearish_harami', 3)},
                'dark_cloud_cover': {'signal': 'SELL', 'strength': pattern_strength_weights.get('dark_cloud_cover', 3)},
                'evening_star': {'signal': 'SELL', 'strength': pattern_strength_weights.get('evening_star', 4)},
                
                # Neutral patterns with context signals
                'doji': {'signal': 'NEUTRAL', 'strength': pattern_strength_weights.get('doji', 1)},
                'spinning_tops': {'signal': 'NEUTRAL', 'strength': pattern_strength_weights.get('spinning_tops', 1)}
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
            buy_count = len(buy_signals)  # Define buy_count here
            sell_count = len(sell_signals)  # Define sell_count here
            
            # Determine overall signal
            overall_signal = 'NEUTRAL'
            overall_strength = 0
            
            if buy_count > sell_count:
                overall_signal = 'BUY'
                # Calculate strength based on buy signal strengths and margin over sell signals
                overall_strength = min(5, max(1, round(sum(s['strength'] for s in buy_signals) / (buy_count + 1))))
                
            elif sell_count > buy_count:
                overall_signal = 'SELL'
                # Calculate strength based on sell signal strengths and margin over buy signals
                overall_strength = min(5, max(1, round(sum(s['strength'] for s in sell_signals) / (sell_count + 1))))
            
            else:
                # If nearly equal, check signal strengths
                buy_strength = sum(s['strength'] for s in buy_signals)
                sell_strength = sum(s['strength'] for s in sell_signals)
                
                if buy_strength > sell_strength * 1.5:
                    overall_signal = 'BUY'
                    overall_strength = min(5, max(1, round(buy_strength / (len(buy_signals) + len(sell_signals) + 1))))
                elif sell_strength > buy_strength * 1.5:
                    overall_signal = 'SELL'
                    overall_strength = min(5, max(1, round(sell_strength / (len(buy_signals) + len(sell_signals) + 1))))
            
            return {
                'overall_signal': overall_signal,
                'overall_strength': overall_strength,
                'individual_signals': signals,
                'buy_signals_count': buy_count,
                'sell_signals_count': sell_count
            }

    # ===============================================================
    # Technical Indicators Analysis
    # ===============================================================

    class TechnicalIndicators:
        """Calculate and interpret technical indicators for trading signals"""
        
        def __init__(self, df, params=None):
            """
            Initialize with DataFrame containing OHLCV data
            
            Args:
                df: DataFrame with OHLCV data
                params: Optional TradingParameters instance for indicator thresholds
            """
            # Ensure we have a valid DataFrame
            if df is None or len(df) == 0:
                raise ValueError("Empty DataFrame provided")
                
            # Make a deep copy to avoid modifying the original
            self.df = df.copy()
            
            # Initialize parameters (use defaults if not provided)
            self.params = params or TradingParameters()
            
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
            self.calculate_sma(periods=self.params.get_indicator_param('sma_periods'))
            self.calculate_ema(periods=self.params.get_indicator_param('ema_periods'))
            
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
        
        def calculate_support_resistance(self):
            """
            Calculate key support and resistance levels
            
            Returns:
                Dictionary with support and resistance levels
            """
            # Get parameters from config
            lookback = self.params.get_indicator_param('sr_lookback')
            price_tolerance = self.params.get_indicator_param('sr_price_tolerance')
            window_size = self.params.get_indicator_param('sr_window_size')
            
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
        
        def calculate_fibonacci_retracement(self):
            """
            Calculate Fibonacci retracement levels for the most recent significant trend
            
            Returns:
                Dictionary with Fibonacci levels
            """
            # Get parameter from config
            lookback = self.params.get_indicator_param('fib_lookback')
            
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
        
        def calculate_volume_ratio(self):
            """
            Calculate volume relative to moving averages
            
            Returns:
                Dictionary with volume ratio values
            """
            # Skip if Volume data is not available
            if 'Volume' not in self.df.columns:
                return {}
                
            # Get parameters from config
            periods = self.params.get_indicator_param('volume_ma_periods')
            high_volume_threshold = self.params.get_indicator_param('high_volume_threshold')
            low_volume_threshold = self.params.get_indicator_param('low_volume_threshold')
                
            # Calculate volume moving averages
            for period in periods:
                self.df[f'volume_ma{period}'] = self.df['Volume'].rolling(window=period).mean()
                
            # Calculate volume ratio (current volume / average volume)
            for period in periods:
                # Vectorized calculation with handling for zero values
                self.df[f'volume_ratio_{period}'] = np.where(
                    self.df[f'volume_ma{period}'] > 0,
                    self.df['Volume'] / self.df[f'volume_ma{period}'],
                    0
                )
                
            # Flag high and low volume (based on 20-period MA)
            self.df['high_volume'] = self.df['Volume'] > (self.df['volume_ma20'] * high_volume_threshold)
            self.df['low_volume'] = self.df['Volume'] < (self.df['volume_ma20'] * low_volume_threshold)
            
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
            Calculate On-Balance Volume using vectorized operations
            
            Returns:
                Series with OBV values
            """
            # Skip if Volume data is not available
            if 'Volume' not in self.df.columns:
                return pd.Series()
                
            # Initialize OBV with zeros
            self.df['obv'] = 0
            
            # Calculate price direction
            price_direction = np.sign(self.df['Close'].diff())
            
            # Set first value to 0
            price_direction.iloc[0] = 0
            
            # Calculate OBV (vectorized)
            self.df['obv'] = (price_direction * self.df['Volume']).cumsum()
            
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
                    # Avoid division by zero with where 
                    group_vwap = np.where(
                        cumulative_volume > 0,
                        cumulative_tp_volume / cumulative_volume,
                        group['typical_price']  # Use typical price as fallback
                    )
                    vwap_list.append(pd.Series(group_vwap, index=group.index))
                
                self.df['vwap'] = pd.concat(vwap_list)
            else:
                # Calculate cumulative VWAP without resetting
                tp_volume = self.df['typical_price'] * self.df['Volume']
                cumulative_tp_volume = tp_volume.cumsum()
                cumulative_volume = self.df['Volume'].cumsum()
                
                # Avoid division by zero with where
                self.df['vwap'] = np.where(
                    cumulative_volume > 0,
                    cumulative_tp_volume / cumulative_volume,
                    self.df['typical_price']  # Use typical price as fallback
                )
            
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
            sma_columns = {}
            
            for period in periods:
                column_name = f'sma_{period}'
                self.df[column_name] = self.df['Close'].rolling(window=period).mean()
                sma_columns[column_name] = self.df[column_name]
                
            # Calculate crossovers between key moving averages (vectorized)
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
                
            return sma_columns
        
        def calculate_ema(self, periods=[5, 12, 26, 50]):
            """
            Calculate Exponential Moving Average for specified periods
            
            Args:
                periods: List of periods to calculate EMAs for
                
            Returns:
                Dictionary with EMA values for each period
            """
            ema_columns = {}
            
            for period in periods:
                column_name = f'ema_{period}'
                self.df[column_name] = self.df['Close'].ewm(span=period, adjust=False).mean()
                ema_columns[column_name] = self.df[column_name]
                
            # Calculate crossovers between key moving averages (vectorized)
            if all(f'ema_{p}' in self.df.columns for p in [12, 26]):
                self.df['ema_12_26_bullish_cross'] = (
                    (self.df['ema_12'] > self.df['ema_26']) &
                    (self.df['ema_12'].shift(1) <= self.df['ema_26'].shift(1))
                )
                
                self.df['ema_12_26_bearish_cross'] = (
                    (self.df['ema_12'] < self.df['ema_26']) &
                    (self.df['ema_12'].shift(1) >= self.df['ema_26'].shift(1))
                )
                
            return ema_columns
        
        #
        # 5. RSI INDICATOR
        #
        
        def calculate_rsi(self):
            """
            Calculate Relative Strength Index using vectorized operations
            
            Returns:
                Series with RSI values
            """
            # Get parameters from config
            period = self.params.get_indicator_param('rsi_period')
            oversold_threshold = self.params.get_indicator_param('rsi_oversold')
            overbought_threshold = self.params.get_indicator_param('rsi_overbought')
            
            # Calculate price changes
            delta = self.df['Close'].diff()
            
            # Separate gains (up) from losses (down)
            up = delta.clip(lower=0)
            down = -delta.clip(upper=0)
            
            # Calculate rolling averages
            avg_gain = up.rolling(period).mean()
            avg_loss = down.rolling(period).mean()
            
            # Calculate RS (handle division by zero with where)
            rs = np.where(
                avg_loss > 0,
                avg_gain / avg_loss,
                100.0  # If no losses, RSI should be high
            )
            
            # Calculate RSI
            self.df['rsi'] = 100 - (100 / (1 + rs))
            
            # Store RSI thresholds
            self.df['rsi_oversold'] = self.df['rsi'] < oversold_threshold
            self.df['rsi_overbought'] = self.df['rsi'] > overbought_threshold
            
            # Calculate price and RSI higher highs/lower lows for divergence (vectorized)
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
            
            self.df['rsi_lower_low'] = (
                (self.df['rsi'] < self.df['rsi'].shift(1)) &
                (self.df['rsi'].shift(1) < self.df['rsi'].shift(2))
            )
            
            # Bullish divergence: price makes lower lows but RSI makes higher lows
            self.df['rsi_bullish_divergence'] = (
                self.df['price_lower_low'] &
                ~self.df['rsi_lower_low']
            )
            
            # Bearish divergence: price makes higher highs but RSI makes lower highs
            self.df['rsi_bearish_divergence'] = (
                self.df['price_higher_high'] &
                ~self.df['rsi_higher_high']
            )
            
            return self.df['rsi']
        
        #
        # 6. MACD INDICATOR
        #
        
        def calculate_macd(self):
            """
            Calculate Moving Average Convergence Divergence using vectorized operations
            
            Returns:
                Dictionary with MACD line, signal line, and histogram
            """
            # Get parameters from config
            fast = self.params.get_indicator_param('macd_fast')
            slow = self.params.get_indicator_param('macd_slow')
            signal = self.params.get_indicator_param('macd_signal')
            
            # Calculate MACD components
            fast_ema = self.df['Close'].ewm(span=fast, adjust=False).mean()
            slow_ema = self.df['Close'].ewm(span=slow, adjust=False).mean()
            
            # MACD Line
            self.df['macd_line'] = fast_ema - slow_ema
            
            # Signal Line
            self.df['macd_signal'] = self.df['macd_line'].ewm(span=signal, adjust=False).mean()
            
            # MACD Histogram
            self.df['macd_histogram'] = self.df['macd_line'] - self.df['macd_signal']
            
            # MACD crossovers (bullish/bearish signals) (vectorized)
            self.df['macd_bullish_crossover'] = (
                (self.df['macd_line'] > self.df['macd_signal']) & 
                (self.df['macd_line'].shift(1) <= self.df['macd_signal'].shift(1))
            )
            
            self.df['macd_bearish_crossover'] = (
                (self.df['macd_line'] < self.df['macd_signal']) & 
                (self.df['macd_line'].shift(1) >= self.df['macd_signal'].shift(1))
            )
            
            # MACD histogram direction
            self.df['macd_histogram_rising'] = self.df['macd_histogram'] > self.df['macd_histogram'].shift(1)
            
            # MACD divergence (price makes new high but MACD doesn't) (vectorized)
            self.df['macd_bullish_divergence'] = (
                (self.df['price_lower_low']) &
                (self.df['macd_line'] > self.df['macd_line'].shift(1))
            )
            
            self.df['macd_bearish_divergence'] = (
                (self.df['price_higher_high']) &
                (self.df['macd_line'] < self.df['macd_line'].shift(1))
            )
            
            return {
                'macd_line': self.df['macd_line'],
                'macd_signal': self.df['macd_signal'],
                'macd_histogram': self.df['macd_histogram']
            }
        
        #
        # 7. BOLLINGER BANDS
        #
        
        def calculate_bollinger_bands(self):
            """
            Calculate Bollinger Bands using vectorized operations
            
            Returns:
                Dictionary with upper band, middle band (SMA), and lower band
            """
            # Get parameters from config
            period = self.params.get_indicator_param('bb_period')
            std_dev = self.params.get_indicator_param('bb_std_dev')
            
            # Middle band is SMA
            self.df['bb_middle'] = self.df['Close'].rolling(window=period).mean()
            
            # Calculate standard deviation
            rolling_std = self.df['Close'].rolling(window=period).std()
            
            # Calculate upper and lower bands
            self.df['bb_upper'] = self.df['bb_middle'] + (rolling_std * std_dev)
            self.df['bb_lower'] = self.df['bb_middle'] - (rolling_std * std_dev)
            
            # Calculate % B (position within bands) - handle division by zero
            bb_range = self.df['bb_upper'] - self.df['bb_lower']
            self.df['bb_pct_b'] = np.where(
                bb_range > 0,
                (self.df['Close'] - self.df['bb_lower']) / bb_range,
                0.5  # Default to middle when bands are equal
            )
            
            # Calculate bandwidth (indicator of volatility) - handle division by zero
            self.df['bb_bandwidth'] = np.where(
                self.df['bb_middle'] > 0,
                (self.df['bb_upper'] - self.df['bb_lower']) / self.df['bb_middle'],
                0  # Default to 0 when middle band is zero
            )
            
            # Identify overbought/oversold conditions
            self.df['bb_overbought'] = self.df['Close'] > self.df['bb_upper']
            self.df['bb_oversold'] = self.df['Close'] < self.df['bb_lower']
            
            # Squeeze condition (low volatility)
            self.df['bb_squeeze'] = self.df['bb_bandwidth'] < self.df['bb_bandwidth'].rolling(window=50).quantile(0.2)
            
            # Direction after band touch
            self.df['bb_upper_touch'] = self.df['High'] >= self.df['bb_upper']
            self.df['bb_lower_touch'] = self.df['Low'] <= self.df['bb_lower']
            
            return {
                'bb_upper': self.df['bb_upper'],
                'bb_middle': self.df['bb_middle'],
                'bb_lower': self.df['bb_lower']
            }
        
        #
        # 8. STOCHASTIC OSCILLATOR
        #
        
        def calculate_stochastic(self):
            """
            Calculate Stochastic Oscillator using vectorized operations
            
            Returns:
                Dictionary with %K and %D values
            """
            # Get parameters from config
            k_period = self.params.get_indicator_param('stoch_k_period')
            d_period = self.params.get_indicator_param('stoch_d_period')
            slowing = self.params.get_indicator_param('stoch_slowing')
            overbought = self.params.get_indicator_param('stoch_overbought')
            oversold = self.params.get_indicator_param('stoch_oversold')
            
            # Calculate %K
            lowest_low = self.df['Low'].rolling(window=k_period).min()
            highest_high = self.df['High'].rolling(window=k_period).max()
            
            # Calculate the range and handle division by zero
            price_range = highest_high - lowest_low
            
            # Fast %K
            stoch_k_fast = np.where(
                price_range > 0,
                100 * ((self.df['Close'] - lowest_low) / price_range),
                50  # Default to middle when price range is zero
            )
            
            # Slow %K (with slowing)
            self.df['stoch_k'] = pd.Series(stoch_k_fast).rolling(window=slowing).mean()
            
            # %D is the SMA of %K
            self.df['stoch_d'] = self.df['stoch_k'].rolling(window=d_period).mean()
            
            # Identify overbought/oversold conditions
            self.df['stoch_overbought'] = self.df['stoch_k'] > overbought
            self.df['stoch_oversold'] = self.df['stoch_k'] < oversold
            
            # Identify crossovers (vectorized)
            self.df['stoch_bullish_crossover'] = (
                (self.df['stoch_k'] > self.df['stoch_d']) & 
                (self.df['stoch_k'].shift(1) <= self.df['stoch_d'].shift(1))
            )
            
            self.df['stoch_bearish_crossover'] = (
                (self.df['stoch_k'] < self.df['stoch_d']) & 
                (self.df['stoch_k'].shift(1) >= self.df['stoch_d'].shift(1))
            )
            
            # Strong signals (crossover in overbought/oversold zone)
            self.df['stoch_strong_buy'] = self.df['stoch_bullish_crossover'] & self.df['stoch_oversold']
            self.df['stoch_strong_sell'] = self.df['stoch_bearish_crossover'] & self.df['stoch_overbought']
            
            # Calculate Stochastic divergence (vectorized)
            self.df['stoch_higher_high'] = (
                (self.df['stoch_k'] > self.df['stoch_k'].shift(1)) &
                (self.df['stoch_k'].shift(1) > self.df['stoch_k'].shift(2))
            )
            
            self.df['stoch_lower_low'] = (
                (self.df['stoch_k'] < self.df['stoch_k'].shift(1)) &
                (self.df['stoch_k'].shift(1) < self.df['stoch_k'].shift(2))
            )
            
            # Stochastic divergence
            self.df['stoch_bullish_divergence'] = (
                self.df['price_lower_low'] &
                ~self.df['stoch_lower_low']
            )
            
            self.df['stoch_bearish_divergence'] = (
                self.df['price_higher_high'] &
                ~self.df['stoch_higher_high']
            )
            
            return {
                'stoch_k': self.df['stoch_k'],
                'stoch_d': self.df['stoch_d']
            }
        
        #
        # 9. STOCHASTIC RSI
        #
        
        def calculate_stochastic_rsi(self):
            """
            Calculate Stochastic RSI using vectorized operations
            
            Returns:
                Dictionary with StochRSI K and D values
            """
            # Get parameters from config
            rsi_period = self.params.get_indicator_param('rsi_period')
            stoch_period = self.params.get_indicator_param('stoch_k_period')
            k_period = self.params.get_indicator_param('stoch_d_period')
            d_period = self.params.get_indicator_param('stoch_d_period')
            
            # Ensure RSI is calculated
            if 'rsi' not in self.df.columns:
                self.calculate_rsi()
            
            # Calculate Stochastic RSI
            min_rsi = self.df['rsi'].rolling(window=stoch_period).min()
            max_rsi = self.df['rsi'].rolling(window=stoch_period).max()
            
            # Handle division by zero
            rsi_range = max_rsi - min_rsi
            
            # Calculate the raw Stochastic RSI
            stoch_rsi = np.where(
                rsi_range > 0,
                100 * ((self.df['rsi'] - min_rsi) / rsi_range),
                50  # Default to middle when range is zero
            )
            
            # Smooth with moving averages
            self.df['stoch_rsi_k'] = pd.Series(stoch_rsi).rolling(window=k_period).mean()
            self.df['stoch_rsi_d'] = self.df['stoch_rsi_k'].rolling(window=d_period).mean()
            
            # Identify overbought/oversold conditions
            self.df['stoch_rsi_overbought'] = self.df['stoch_rsi_k'] > 80
            self.df['stoch_rsi_oversold'] = self.df['stoch_rsi_k'] < 20
            
            # Identify crossovers (vectorized)
            self.df['stoch_rsi_bullish_crossover'] = (
                (self.df['stoch_rsi_k'] > self.df['stoch_rsi_d']) & 
                (self.df['stoch_rsi_k'].shift(1) <= self.df['stoch_rsi_d'].shift(1))
            )
            
            self.df['stoch_rsi_bearish_crossover'] = (
                (self.df['stoch_rsi_k'] < self.df['stoch_rsi_d']) & 
                (self.df['stoch_rsi_k'].shift(1) >= self.df['stoch_rsi_d'].shift(1))
            )
            
            # Oversold/overbought crossovers (stronger signals)
            self.df['stoch_rsi_strong_buy'] = self.df['stoch_rsi_bullish_crossover'] & self.df['stoch_rsi_oversold']
            self.df['stoch_rsi_strong_sell'] = self.df['stoch_rsi_bearish_crossover'] & self.df['stoch_rsi_overbought']
            
            return {
                'stoch_rsi_k': self.df['stoch_rsi_k'],
                'stoch_rsi_d': self.df['stoch_rsi_d']
            }
        
        #
        # 10. ADX (AVERAGE DIRECTIONAL INDEX)
        #
        
        def calculate_adx(self):
            """
            Calculate Average Directional Index using vectorized operations
            
            Returns:
                Dictionary with ADX, +DI and -DI values
            """
            # Get parameters from config
            period = self.params.get_indicator_param('adx_period')
            adx_threshold = self.params.get_indicator_param('adx_threshold')
            
            # Calculate True Range
            high_low = self.df['High'] - self.df['Low']
            high_close = abs(self.df['High'] - self.df['Close'].shift(1))
            low_close = abs(self.df['Low'] - self.df['Close'].shift(1))
            
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()
            
            # Calculate +DM and -DM
            up_move = self.df['High'] - self.df['High'].shift(1)
            down_move = self.df['Low'].shift(1) - self.df['Low']
            
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            
            # Calculate smoothed +DM and -DM
            plus_di = 100 * pd.Series(plus_dm).rolling(window=period).mean() / atr
            minus_di = 100 * pd.Series(minus_dm).rolling(window=period).mean() / atr
            
            # Calculate DX - handle division by zero
            dx = np.where(
                (plus_di + minus_di) > 0,
                100 * abs(plus_di - minus_di) / (plus_di + minus_di),
                0  # Default to 0 when both DIs are zero
            )
            
            # Calculate ADX
            self.df['adx'] = pd.Series(dx).rolling(window=period).mean()
            self.df['plus_di'] = plus_di
            self.df['minus_di'] = minus_di
            
            # Identify strong trend
            self.df['adx_strong_trend'] = self.df['adx'] > adx_threshold
            self.df['adx_weak_trend'] = self.df['adx'] < 20
            
            # Identify positive/negative crossover (vectorized)
            self.df['adx_bullish_crossover'] = (
                (self.df['plus_di'] > self.df['minus_di']) & 
                (self.df['plus_di'].shift(1) <= self.df['minus_di'].shift(1)) &
                (self.df['adx'] > adx_threshold)
            )
            
            self.df['adx_bearish_crossover'] = (
                (self.df['plus_di'] < self.df['minus_di']) & 
                (self.df['plus_di'].shift(1) >= self.df['minus_di'].shift(1)) &
                (self.df['adx'] > adx_threshold)
            )
            
            # Strong ADX buy/sell signals
            self.df['adx_strong_buy'] = (
                self.df['adx'] > adx_threshold &
                self.df['plus_di'] > self.df['minus_di'] &
                self.df['adx'] > self.df['adx'].shift(1)
            )
            
            self.df['adx_strong_sell'] = (
                self.df['adx'] > adx_threshold &
                self.df['minus_di'] > self.df['plus_di'] &
                self.df['adx'] > self.df['adx'].shift(1)
            )
            
            return {
                'adx': self.df['adx'],
                'plus_di': self.df['plus_di'],
                'minus_di': self.df['minus_di']
            }
        
        #
        # 11. AROON INDICATOR
        #
        
        def calculate_aroon(self):
            """
            Calculate Aroon Indicator
            
            Returns:
                Dictionary with Aroon Up, Aroon Down, and Aroon Oscillator
            """
            # Get parameter from config
            period = 25  # Default Aroon period
            
            # Initialize arrays for Aroon values
            aroon_up = np.zeros(len(self.df))
            aroon_down = np.zeros(len(self.df))
            
            # For vectorization, we'll use rolling apply
            # Function to find days since high in a window
            def days_since_high(window):
                if len(window) < period:
                    return np.nan
                # Return the position of max value from the end (most recent)
                return period - 1 - np.argmax(window)
                
            # Function to find days since low in a window
            def days_since_low(window):
                if len(window) < period:
                    return np.nan
                # Return the position of min value from the end (most recent)
                return period - 1 - np.argmin(window)
            
            # Calculate days since high/low using rolling apply
            if len(self.df) >= period:
                days_high = self.df['High'].rolling(window=period).apply(
                    lambda x: days_since_high(x), raw=True)
                days_low = self.df['Low'].rolling(window=period).apply(
                    lambda x: days_since_low(x), raw=True)
                
                # Calculate Aroon values (vectorized)
                self.df['aroon_up'] = 100 * (period - days_high) / period
                self.df['aroon_down'] = 100 * (period - days_low) / period
                self.df['aroon_oscillator'] = self.df['aroon_up'] - self.df['aroon_down']
            else:
                # Not enough data for calculation
                self.df['aroon_up'] = np.nan
                self.df['aroon_down'] = np.nan
                self.df['aroon_oscillator'] = np.nan
            
            # Generate signals (vectorized)
            self.df['aroon_bullish'] = (self.df['aroon_up'] > 50) & (self.df['aroon_down'] < 30)
            self.df['aroon_bearish'] = (self.df['aroon_down'] > 50) & (self.df['aroon_up'] < 30)
            
            # Strong signals when Aroon crossing with extreme values
            self.df['aroon_strong_bull'] = (
                (self.df['aroon_up'] > 70) & 
                (self.df['aroon_down'] < 30) & 
                (self.df['aroon_up'] > self.df['aroon_down'])
            )
            
            self.df['aroon_strong_bear'] = (
                (self.df['aroon_down'] > 70) & 
                (self.df['aroon_up'] < 30) & 
                (self.df['aroon_down'] > self.df['aroon_up'])
            )
            
            return {
                'aroon_up': self.df['aroon_up'],
                'aroon_down': self.df['aroon_down'],
                'aroon_oscillator': self.df['aroon_oscillator']
            }
        
        #
        # 12. ATR (AVERAGE TRUE RANGE)
        #
        
        def calculate_atr(self):
            """
            Calculate Average True Range using vectorized operations
            
            Returns:
                Series with ATR values
            """
            # Get parameter from config
            period = self.params.get_indicator_param('atr_period')
            
            # Calculate True Range (vectorized)
            high_low = self.df['High'] - self.df['Low']
            high_close = abs(self.df['High'] - self.df['Close'].shift(1))
            low_close = abs(self.df['Low'] - self.df['Close'].shift(1))
            
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            self.df['atr'] = tr.rolling(window=period).mean()
            
            # Calculate ATR percentage (ATR relative to price)
            self.df['atr_pct'] = 100 * self.df['atr'] / self.df['Close']
            
            # Calculate potential stop loss levels based on ATR
            self.df['atr_stop_long'] = self.df['Close'] - (self.df['atr'] * 2)
            self.df['atr_stop_short'] = self.df['Close'] + (self.df['atr'] * 2)
            
            # Calculate dynamic price targets based on ATR
            self.df['atr_target_long'] = self.df['Close'] + (self.df['atr'] * 3)
            self.df['atr_target_short'] = self.df['Close'] - (self.df['atr'] * 3)
            
            # Calculate reward:risk ratios based on ATR (handling division by zero)
            long_risk = self.df['Close'] - self.df['atr_stop_long']
            short_risk = self.df['atr_stop_short'] - self.df['Close']
            
            self.df['atr_rrr_long'] = np.where(
                long_risk > 0,
                (self.df['atr_target_long'] - self.df['Close']) / long_risk,
                0  # Default to 0 when risk is zero or negative
            )
            
            self.df['atr_rrr_short'] = np.where(
                short_risk > 0,
                (self.df['Close'] - self.df['atr_target_short']) / short_risk,
                0  # Default to 0 when risk is zero or negative
            )
            
            return self.df['atr']
        
        #
        # 13. ATR BANDS
        #
        
        def calculate_atr_bands(self):
            """
            Calculate ATR Bands using vectorized operations
            
            Returns:
                Dictionary with upper and lower ATR bands
            """
            # Get parameters from config
            period = self.params.get_indicator_param('atr_period')
            multiplier = self.params.get_indicator_param('atr_multiplier')
            
            # Ensure ATR is calculated
            if 'atr' not in self.df.columns:
                self.calculate_atr()
            
            # Calculate moving average
            self.df['atr_ma'] = self.df['Close'].rolling(window=period).mean()
            
            # Calculate ATR bands
            self.df['atr_upper'] = self.df['atr_ma'] + (self.df['atr'] * multiplier)
            self.df['atr_lower'] = self.df['atr_ma'] - (self.df['atr'] * multiplier)
            
            # Generate signals (vectorized)
            self.df['atr_upper_break'] = self.df['Close'] > self.df['atr_upper']
            self.df['atr_lower_break'] = self.df['Close'] < self.df['atr_lower']
            
            # ATR band penetration signals
            self.df['atr_band_buy'] = (
                (self.df['Close'] > self.df['atr_upper']) & 
                (self.df['Close'].shift(1) <= self.df['atr_upper'].shift(1))
            )
            
            self.df['atr_band_sell'] = (
                (self.df['Close'] < self.df['atr_lower']) & 
                (self.df['Close'].shift(1) >= self.df['atr_lower'].shift(1))
            )
            
            return {
                'atr_upper': self.df['atr_upper'],
                'atr_lower': self.df['atr_lower']
            }
        
        #
        # 14. SUPERTREND INDICATOR
        #
        
        def calculate_supertrend(self):
            """
            Calculate SuperTrend indicator
            
            Returns:
                Series with SuperTrend values
            """
            # Get parameters from config
            period = self.params.get_indicator_param('supertrend_period')
            multiplier = self.params.get_indicator_param('supertrend_multiplier')
            
            # Ensure ATR is calculated
            if 'atr' not in self.df.columns:
                self.calculate_atr()
            
            # Calculate basic upper and lower bands
            self.df['basic_upper'] = ((self.df['High'] + self.df['Low']) / 2) + (multiplier * self.df['atr'])
            self.df['basic_lower'] = ((self.df['High'] + self.df['Low']) / 2) - (multiplier * self.df['atr'])
            
            # Initialize SuperTrend columns
            self.df['supertrend'] = 0.0
            self.df['supertrend_direction'] = True  # True for uptrend, False for downtrend
            
            # This is one algorithm that's difficult to fully vectorize
            # We'll use an optimized loop but retain the structure for correctness
            if len(self.df) > period:
                # Set initial values for the first valid day
                self.df.loc[self.df.index[period], 'supertrend'] = self.df.loc[self.df.index[period], 'basic_lower']
                self.df.loc[self.df.index[period], 'supertrend_direction'] = True
                
                # Calculate iteratively for the rest of the days
                for i in range(period+1, len(self.df)):
                    curr = self.df.iloc[i]
                    prev = self.df.iloc[i-1]
                    
                    # Algorithm remains the same but using direct DataFrame access
                    if prev['supertrend'] == prev['basic_upper']:
                        curr_upper = min(curr['basic_upper'], prev['supertrend'])
                    else:
                        curr_upper = curr['basic_upper']
                        
                    if prev['supertrend'] == prev['basic_lower']:
                        curr_lower = max(curr['basic_lower'], prev['supertrend'])
                    else:
                        curr_lower = curr['basic_lower']
                    
                    # Determine SuperTrend direction
                    if prev['supertrend'] == prev['basic_upper'] and curr['Close'] <= curr_upper:
                        # Continue downtrend
                        self.df.loc[self.df.index[i], 'supertrend'] = curr_upper
                        self.df.loc[self.df.index[i], 'supertrend_direction'] = False
                    elif prev['supertrend'] == prev['basic_upper'] and curr['Close'] > curr_upper:
                        # Change to uptrend
                        self.df.loc[self.df.index[i], 'supertrend'] = curr_lower
                        self.df.loc[self.df.index[i], 'supertrend_direction'] = True
                    elif prev['supertrend'] == prev['basic_lower'] and curr['Close'] >= curr_lower:
                        # Continue uptrend
                        self.df.loc[self.df.index[i], 'supertrend'] = curr_lower
                        self.df.loc[self.df.index[i], 'supertrend_direction'] = True
                    elif prev['supertrend'] == prev['basic_lower'] and curr['Close'] < curr_lower:
                        # Change to downtrend
                        self.df.loc[self.df.index[i], 'supertrend'] = curr_upper
                        self.df.loc[self.df.index[i], 'supertrend_direction'] = False
            
            # Create buy/sell signals (vectorized)
            if len(self.df) > 1:  # Need at least 2 rows to calculate signals
                self.df['supertrend_buy'] = (
                    (self.df['supertrend_direction'] == True) & 
                    (self.df['supertrend_direction'].shift(1) == False)
                )
                
                self.df['supertrend_sell'] = (
                    (self.df['supertrend_direction'] == False) & 
                    (self.df['supertrend_direction'].shift(1) == True)
                )
            else:
                self.df['supertrend_buy'] = False
                self.df['supertrend_sell'] = False
            
            return self.df['supertrend']
        
        #
        # 15. ALLIGATOR INDICATOR
        #
        
        def calculate_alligator(self):
            """
            Calculate Alligator indicator using vectorized operations
            
            Returns:
                Dictionary with jaw, teeth, and lips lines
            """
            # Get default values for alligator periods
            jaw = 13
            teeth = 8
            lips = 5
            
            # Calculate the median price
            self.df['median_price'] = (self.df['High'] + self.df['Low']) / 2
            
            # Calculate the three lines
            self.df['alligator_jaw'] = self.df['median_price'].rolling(window=jaw).mean().shift(8)
            self.df['alligator_teeth'] = self.df['median_price'].rolling(window=teeth).mean().shift(5)
            self.df['alligator_lips'] = self.df['median_price'].rolling(window=lips).mean().shift(3)
            
            # Determine if Alligator is sleeping (lines are intertwined)
            max_line = self.df[['alligator_jaw', 'alligator_teeth', 'alligator_lips']].max(axis=1)
            min_line = self.df[['alligator_jaw', 'alligator_teeth', 'alligator_lips']].min(axis=1)
            
            # If the difference between max and min is small, Alligator is sleeping
            self.df['alligator_sleeping'] = (max_line - min_line) < (self.df['Close'] * 0.01)  # 1% of price
            
            # Determine buy/sell signal (vectorized)
            self.df['alligator_buy'] = (
                ~self.df['alligator_sleeping'] &
                (self.df['Close'] > self.df['alligator_lips']) &
                (self.df['alligator_lips'] > self.df['alligator_teeth']) &
                (self.df['alligator_teeth'] > self.df['alligator_jaw'])
            )
            
            self.df['alligator_sell'] = (
                ~self.df['alligator_sleeping'] &
                (self.df['Close'] < self.df['alligator_lips']) &
                (self.df['alligator_lips'] < self.df['alligator_teeth']) &
                (self.df['alligator_teeth'] < self.df['alligator_jaw'])
            )
            
            # Define the feeding phase
            self.df['alligator_feeding'] = (
                ~self.df['alligator_sleeping'] &
                (
                    (self.df['alligator_buy'] & (self.df['Close'] > self.df['Close'].shift(1))) |
                    (self.df['alligator_sell'] & (self.df['Close'] < self.df['Close'].shift(1)))
                )
            )
            
            return {
                'jaw': self.df['alligator_jaw'],
                'teeth': self.df['alligator_teeth'],
                'lips': self.df['alligator_lips']
            }
        
        #
        # 16. CENTRAL PIVOT RANGE (CPR)
        #
        
        def calculate_cpr(self):
            """
            Calculate Central Pivot Range
            
            Returns:
                Dictionary with pivot, TC, and BC values
            """
            # Calculate the previous day's data
            self.df['prev_high'] = self.df['High'].shift(1)
            self.df['prev_low'] = self.df['Low'].shift(1)
            self.df['prev_close'] = self.df['Close'].shift(1)
            
            # Calculate pivot points
            self.df['pivot'] = (self.df['prev_high'] + self.df['prev_low'] + self.df['prev_close']) / 3
            self.df['bc'] = (self.df['prev_high'] + self.df['prev_low']) / 2
            self.df['tc'] = (self.df['pivot'] - self.df['bc']) + self.df['pivot']
            
            # Calculate traditional support and resistance levels
            self.df['r1'] = (2 * self.df['pivot']) - self.df['prev_low']
            self.df['s1'] = (2 * self.df['pivot']) - self.df['prev_high']
            self.df['r2'] = self.df['pivot'] + (self.df['prev_high'] - self.df['prev_low'])
            self.df['s2'] = self.df['pivot'] - (self.df['prev_high'] - self.df['prev_low'])
            
            # Calculate CPR width (indication of volatility/range)
            self.df['cpr_width'] = self.df['tc'] - self.df['bc']
            
            # Handle division by zero for percentage calculation
            self.df['cpr_width_pct'] = np.where(
                self.df['pivot'] > 0,
                100 * self.df['cpr_width'] / self.df['pivot'],
                0  # Default to 0 when pivot is zero
            )
            
            # Price position relative to CPR (vectorized)
            self.df['above_cpr'] = self.df['Close'] > self.df['tc']
            self.df['below_cpr'] = self.df['Close'] < self.df['bc']
            self.df['inside_cpr'] = (self.df['Close'] >= self.df['bc']) & (self.df['Close'] <= self.df['tc'])
            
            # CPR breakout signals
            self.df['cpr_breakout_up'] = (
                (self.df['Close'] > self.df['tc']) & 
                (self.df['Close'].shift(1) <= self.df['tc'].shift(1))
            )
            
            self.df['cpr_breakout_down'] = (
                (self.df['Close'] < self.df['bc']) & 
                (self.df['Close'].shift(1) >= self.df['bc'].shift(1))
            )
            
            return {
                'pivot': self.df['pivot'],
                'bc': self.df['bc'],
                'tc': self.df['tc']
            }
        
        #
        # 17. DOW THEORY PATTERNS
        #
        
        def detect_double_patterns(self):
            """
            Detect double top and double bottom patterns
            
            Returns:
                Dictionary with double top and double bottom flags
            """
            # Get parameters from config
            tolerance = self.params.get_chart_pattern_param('double_pattern_tolerance')
            lookback = self.params.get_chart_pattern_param('double_pattern_lookback')
            
            # Initialize result columns
            self.df['double_top'] = False
            self.df['double_bottom'] = False
            
            # Need at least 'lookback' periods of data
            if len(self.df) < lookback:
                return {'tops': [], 'bottoms': []}
            
            # Find local extrema (vectorized)
            self.df['is_local_max'] = (
                (self.df['High'] > self.df['High'].shift(1)) & 
                (self.df['High'] > self.df['High'].shift(2)) &
                (self.df['High'] > self.df['High'].shift(-1)) & 
                (self.df['High'] > self.df['High'].shift(-2))
            )
            
            self.df['is_local_min'] = (
                (self.df['Low'] < self.df['Low'].shift(1)) & 
                (self.df['Low'] < self.df['Low'].shift(2)) &
                (self.df['Low'] < self.df['Low'].shift(-1)) & 
                (self.df['Low'] < self.df['Low'].shift(-2))
            )
            
            # This algorithm is difficult to fully vectorize due to the need to
            # find and compare specific maxima/minima within a rolling window
            # We'll use an optimized loop that processes data in chunks
            
            # Process in window steps to reduce iterations
            step_size = 10
            for i in range(lookback, len(self.df), step_size):
                end_idx = min(i + step_size, len(self.df))
                
                for j in range(i, end_idx):
                    # Get window of data
                    window = self.df.iloc[j-lookback:j+1]
                    
                    # Double top detection
                    max_indices = window[window['is_local_max']].index.tolist()
                    if len(max_indices) >= 2:
                        # Check the most recent two maxima
                        last_two_maxima = max_indices[-2:]
                        first_max = window.loc[last_two_maxima[0]]['High']
                        second_max = window.loc[last_two_maxima[1]]['High']
                        
                        # Check if maxima are within tolerance
                        if abs(first_max - second_max) / first_max <= tolerance:
                            # Check if there's a valley between the two tops
                            between_indices = window.index[(window.index > last_two_maxima[0]) & (window.index < last_two_maxima[1])]
                            if len(between_indices) > 0:
                                between_low = window.loc[between_indices]['Low'].min()
                                if between_low < first_max * 0.95 and between_low < second_max * 0.95:
                                    # Double top confirmed
                                    self.df.loc[self.df.index[j], 'double_top'] = True
                    
                    # Double bottom detection
                    min_indices = window[window['is_local_min']].index.tolist()
                    if len(min_indices) >= 2:
                        # Check the most recent two minima
                        last_two_minima = min_indices[-2:]
                        first_min = window.loc[last_two_minima[0]]['Low']
                        second_min = window.loc[last_two_minima[1]]['Low']
                        
                        # Check if minima are within tolerance
                        if abs(first_min - second_min) / first_min <= tolerance:
                            # Check if there's a peak between the two bottoms
                            between_indices = window.index[(window.index > last_two_minima[0]) & (window.index < last_two_minima[1])]
                            if len(between_indices) > 0:
                                between_high = window.loc[between_indices]['High'].max()
                                if between_high > first_min * 1.05 and between_high > second_min * 1.05:
                                    # Double bottom confirmed
                                    self.df.loc[self.df.index[j], 'double_bottom'] = True
            
            return {
                'double_top': self.df['double_top'],
                'double_bottom': self.df['double_bottom']
            }
        
        def detect_triple_patterns(self):
            """
            Detect triple top and triple bottom patterns
            
            Returns:
                Dictionary with triple top and triple bottom flags
            """
            # Get parameters from config
            tolerance = self.params.get_chart_pattern_param('triple_pattern_tolerance')
            lookback = self.params.get_chart_pattern_param('triple_pattern_lookback')
            
            # Initialize result columns
            self.df['triple_top'] = False
            self.df['triple_bottom'] = False
            
            # Need at least 'lookback' periods of data
            if len(self.df) < lookback:
                return {'tops': [], 'bottoms': []}
            
            # Find local extrema (use the ones calculated in double_patterns if available)
            if 'is_local_max' not in self.df.columns:
                self.df['is_local_max'] = (
                    (self.df['High'] > self.df['High'].shift(1)) & 
                    (self.df['High'] > self.df['High'].shift(2)) &
                    (self.df['High'] > self.df['High'].shift(-1)) & 
                    (self.df['High'] > self.df['High'].shift(-2))
                )
                
                self.df['is_local_min'] = (
                    (self.df['Low'] < self.df['Low'].shift(1)) & 
                    (self.df['Low'] < self.df['Low'].shift(2)) &
                    (self.df['Low'] < self.df['Low'].shift(-1)) & 
                    (self.df['Low'] < self.df['Low'].shift(-2))
                )
            
            # Process in window steps to reduce iterations
            step_size = 10
            for i in range(lookback, len(self.df), step_size):
                end_idx = min(i + step_size, len(self.df))
                
                for j in range(i, end_idx):
                    # Get window of data
                    window = self.df.iloc[j-lookback:j+1]
                    
                    # Triple top detection
                    max_indices = window[window['is_local_max']].index.tolist()
                    if len(max_indices) >= 3:
                        # Check the most recent three maxima
                        last_three_maxima = max_indices[-3:]
                        first_max = window.loc[last_three_maxima[0]]['High']
                        second_max = window.loc[last_three_maxima[1]]['High']
                        third_max = window.loc[last_three_maxima[2]]['High']
                        
                        # Check if all maxima are within tolerance
                        if (abs(first_max - second_max) / first_max <= tolerance and
                            abs(first_max - third_max) / first_max <= tolerance and
                            abs(second_max - third_max) / second_max <= tolerance):
                            
                            # Check for valleys between the tops
                            between_indices_1 = window.index[(window.index > last_three_maxima[0]) & (window.index < last_three_maxima[1])]
                            between_indices_2 = window.index[(window.index > last_three_maxima[1]) & (window.index < last_three_maxima[2])]
                            
                            if len(between_indices_1) > 0 and len(between_indices_2) > 0:
                                valley1 = window.loc[between_indices_1]['Low'].min()
                                valley2 = window.loc[between_indices_2]['Low'].min()
                                
                                if (valley1 < first_max * 0.95 and valley1 < second_max * 0.95 and
                                    valley2 < second_max * 0.95 and valley2 < third_max * 0.95):
                                    # Triple top confirmed
                                    self.df.loc[self.df.index[j], 'triple_top'] = True

                    # Triple bottom detection
                    min_indices = window[window['is_local_min']].index.tolist()
                    if len(min_indices) >= 3:
                        # Check the most recent three minima
                        last_three_minima = min_indices[-3:]
                        first_min = window.loc[last_three_minima[0]]['Low']
                        second_min = window.loc[last_three_minima[1]]['Low']
                        third_min = window.loc[last_three_minima[2]]['Low']
                        
                        # Check if all minima are within tolerance
                        if (abs(first_min - second_min) / first_min <= tolerance and
                            abs(first_min - third_min) / first_min <= tolerance and
                            abs(second_min - third_min) / second_min <= tolerance):
                            
                            # Check for peaks between the bottoms
                            between_indices_1 = window.index[(window.index > last_three_minima[0]) & (window.index < last_three_minima[1])]
                            between_indices_2 = window.index[(window.index > last_three_minima[1]) & (window.index < last_three_minima[2])]
                            
                            if len(between_indices_1) > 0 and len(between_indices_2) > 0:
                                peak1 = window.loc[between_indices_1]['High'].max()
                                peak2 = window.loc[between_indices_2]['High'].max()
                                
                                if (peak1 > first_min * 1.05 and peak1 > second_min * 1.05 and
                                    peak2 > second_min * 1.05 and peak2 > third_min * 1.05):
                                    # Triple bottom confirmed
                                    self.df.loc[self.df.index[j], 'triple_bottom'] = True
            
            return {
                'triple_top': self.df['triple_top'],
                'triple_bottom': self.df['triple_bottom']
            }
        
        def detect_trading_range(self):
            """
            Detect if price is trading in a range (sideways market)
            
            Returns:
                Dictionary with trading range information
            """
            # Get parameters from config
            lookback = self.params.get_chart_pattern_param('range_lookback')
            tolerance = self.params.get_chart_pattern_param('range_tolerance')
            min_touches = self.params.get_chart_pattern_param('range_min_touches')
            
            # Initialize result columns
            self.df['in_range'] = False
            
            # Need at least 'lookback' periods of data
            if len(self.df) < lookback:
                return {'in_range': False}
            
            # Process in window steps to improve performance
            step_size = 10
            for i in range(lookback, len(self.df), step_size):
                end_idx = min(i + step_size, len(self.df))
                
                for j in range(i, end_idx):
                    # Get window of data
                    window = self.df.iloc[j-lookback:j+1]
                    
                    # Find highest high and lowest low in the window
                    highest_high = window['High'].max()
                    lowest_low = window['Low'].min()
                    
                    # Calculate range width
                    range_width = highest_high - lowest_low
                    range_midpoint = (highest_high + lowest_low) / 2
                    
                    # Skip division by zero
                    if range_midpoint == 0:
                        continue
                        
                    # Check if range is within tolerance (not too wide)
                    if range_width / range_midpoint <= tolerance:
                        # Count touches to the top and bottom of the range
                        upper_touches = sum((window['High'] >= highest_high * 0.98) & (window['High'] <= highest_high * 1.02))
                        lower_touches = sum((window['Low'] >= lowest_low * 0.98) & (window['Low'] <= lowest_low * 1.02))
                        
                        # Check if there are enough touches to confirm the range
                        if upper_touches + lower_touches >= min_touches:
                            self.df.loc[self.df.index[j], 'in_range'] = True
                            self.df.loc[self.df.index[j], 'range_top'] = highest_high
                            self.df.loc[self.df.index[j], 'range_bottom'] = lowest_low
                            self.df.loc[self.df.index[j], 'range_midpoint'] = range_midpoint
            
            # Range breakout signals
            if 'range_top' in self.df.columns and 'range_bottom' in self.df.columns:
                self.df['range_breakout_up'] = (
                    self.df['in_range'].shift(1) & 
                    (self.df['Close'] > self.df['range_top'].shift(1))
                )
                
                self.df['range_breakout_down'] = (
                    self.df['in_range'].shift(1) & 
                    (self.df['Close'] < self.df['range_bottom'].shift(1))
                )
            
            return {
                'in_range': self.df['in_range']
            }
        
        def detect_flag_patterns(self):
            """
            Detect bull and bear flag patterns
            
            Returns:
                Dictionary with bull and bear flag indicators
            """
            # Get parameters from config
            lookback = self.params.get_chart_pattern_param('flag_lookback')
            pole_threshold = self.params.get_chart_pattern_param('flag_pole_threshold')
            flag_threshold = self.params.get_chart_pattern_param('flag_consolidation_threshold')
            max_flag_bars = self.params.get_chart_pattern_param('flag_max_bars')
            
            # Initialize result columns
            self.df['bull_flag'] = False
            self.df['bear_flag'] = False
            
            # Need at least 'lookback' periods of data
            if len(self.df) < lookback:
                return {'bull_flag': False, 'bear_flag': False}
            
            # Process data in chunks for better performance
            step_size = 5
            for i in range(lookback, len(self.df), step_size):
                end_idx = min(i + step_size, len(self.df))
                
                for j in range(i, end_idx):
                    # Ensure we have enough bars to check for a flag
                    if j < max_flag_bars + 5:
                        continue
                        
                    # Get window of data for the pole and potential flag
                    potential_flag = self.df.iloc[j-max_flag_bars:j+1]
                    potential_pole = self.df.iloc[j-max_flag_bars-5:j-max_flag_bars]
                    
                    # Bull flag: Strong upward move followed by sideways/slight downward consolidation
                    if len(potential_pole) > 0:
                        # Calculate pole height (sharp upward move)
                        pole_low = potential_pole['Low'].min()
                        pole_high = potential_pole['High'].max()
                        pole_height = pole_high - pole_low
                        
                        # Skip division by zero
                        if pole_low == 0 or pole_high == 0:
                            continue
                        
                        # Calculate flag height (consolidation)
                        flag_high = potential_flag['High'].max()
                        flag_low = potential_flag['Low'].min()
                        flag_height = flag_high - flag_low
                        
                        # Check for bull flag pattern
                        if (pole_height / pole_low >= pole_threshold and  # Significant pole
                            flag_height / flag_low <= flag_threshold and  # Small consolidation
                            flag_high <= pole_high and  # Flag within or below pole high
                            flag_low >= pole_low):  # Flag above pole low
                            self.df.loc[self.df.index[j], 'bull_flag'] = True
                        
                        # Check for bear flag pattern (downward pole followed by consolidation)
                        if (pole_height / pole_high >= pole_threshold and  # Significant downward pole
                            flag_height / flag_high <= flag_threshold and  # Small consolidation
                            flag_low >= pole_low and  # Flag within or above pole low
                            flag_high <= pole_high):  # Flag below pole high
                            self.df.loc[self.df.index[j], 'bear_flag'] = True
            
            return {
                'bull_flag': self.df['bull_flag'],
                'bear_flag': self.df['bear_flag']
            }
        
        #
        # 18. REWARD TO RISK RATIO
        # 
        
        def calculate_reward_risk_ratio(self):
            """
            Calculate reward to risk ratio based on ATR for dynamic targets and stops
            
            Returns:
                Dictionary with RRR values for long and short trades
            """
            # Get parameters from config
            target_multiplier = self.params.get_signal_param('target_multiplier')
            stop_multiplier = self.params.get_signal_param('stop_multiplier')
            min_rrr = self.params.get_signal_param('min_rrr')
            
            # Ensure ATR is calculated
            if 'atr' not in self.df.columns:
                self.calculate_atr()
                
            # Calculate potential stops and targets
            self.df['long_entry'] = self.df['Close']
            self.df['long_stop'] = self.df['Close'] - (self.df['atr'] * stop_multiplier)
            self.df['long_target'] = self.df['Close'] + (self.df['atr'] * target_multiplier)
            
            self.df['short_entry'] = self.df['Close']
            self.df['short_stop'] = self.df['Close'] + (self.df['atr'] * stop_multiplier)
            self.df['short_target'] = self.df['Close'] - (self.df['atr'] * target_multiplier)
            
            # Calculate reward:risk ratios (handle division by zero)
            long_risk = self.df['long_entry'] - self.df['long_stop']
            short_risk = self.df['short_stop'] - self.df['short_entry']
            
            self.df['long_rrr'] = np.where(
                long_risk > 0,
                (self.df['long_target'] - self.df['long_entry']) / long_risk,
                0
            )
            
            self.df['short_rrr'] = np.where(
                short_risk > 0,
                (self.df['short_entry'] - self.df['short_target']) / short_risk,
                0
            )
            
            # Check if RRR meets minimum threshold
            self.df['long_rrr_valid'] = self.df['long_rrr'] >= min_rrr
            self.df['short_rrr_valid'] = self.df['short_rrr'] >= min_rrr
            
            return {
                'long_rrr': self.df['long_rrr'],
                'short_rrr': self.df['short_rrr'],
                'long_rrr_valid': self.df['long_rrr_valid'],
                'short_rrr_valid': self.df['short_rrr_valid']
            }
        
        #
        # SIGNAL GENERATION METHODS
        #
        
        def get_latest_indicator_signals(self):
            """
            Get trading signals based on the latest values of all indicators
            
            Returns:
                Dictionary with indicator names and their signals
            """
            if len(self.df) == 0:
                return {}
            
            # Get the latest row of data
            latest = self.df.iloc[-1]
            
            # Dictionary to store indicator signals
            signals = {}
            
            # Get parameters for support & resistance signals
            sr_near_threshold = self.params.get_signal_param('sr_near_threshold_pct')
            sr_very_near_threshold = self.params.get_signal_param('sr_very_near_threshold_pct')
            
            # Support and Resistance signals
            if 'nearest_support' in self.df.columns and 'nearest_resistance' in self.df.columns:
                if not pd.isna(latest['nearest_support']) and not pd.isna(latest['nearest_resistance']):
                    support_distance = latest.get('support_distance_pct', 0)
                    resistance_distance = latest.get('resistance_distance_pct', 0)
                    
                    signals['support_resistance'] = {
                        'signal': 'BUY' if support_distance < sr_near_threshold else 
                                'SELL' if resistance_distance < sr_near_threshold else 'NEUTRAL',
                        'strength': 4 if support_distance < sr_very_near_threshold or resistance_distance < sr_very_near_threshold else 
                                3 if support_distance < sr_near_threshold or resistance_distance < sr_near_threshold else 1,
                        'description': f'Price near support ({support_distance:.2f}% away)' if support_distance < resistance_distance 
                                    else f'Price near resistance ({resistance_distance:.2f}% away)'
                    }
            
            # Fibonacci signals
            if 'nearest_fib' in self.df.columns:
                if not pd.isna(latest['nearest_fib']):
                    fib_level = latest['nearest_fib']
                    fib_trend = latest.get('fib_trend', 'unknown')
                    
                    # Interpret the signal based on trend and level
                    if fib_trend == 'uptrend':
                        if fib_level <= 0.382:  # Strong retracement level in uptrend
                            signals['fibonacci'] = {
                                'signal': 'BUY',
                                'strength': 3,
                                'description': f'Price at {fib_level} Fibonacci retracement in uptrend'
                            }
                        elif fib_level >= 0.618:  # Deep retracement, potential reversal
                            signals['fibonacci'] = {
                                'signal': 'SELL',
                                'strength': 2,
                                'description': f'Price at deep {fib_level} Fibonacci retracement in uptrend'
                            }
                    elif fib_trend == 'downtrend':
                        if fib_level <= 0.382:  # Shallow retracement in downtrend
                            signals['fibonacci'] = {
                                'signal': 'SELL',
                                'strength': 3,
                                'description': f'Price at {fib_level} Fibonacci retracement in downtrend'
                            }
                        elif fib_level >= 0.618:  # Deep retracement, potential reversal
                            signals['fibonacci'] = {
                                'signal': 'BUY',
                                'strength': 2,
                                'description': f'Price at deep {fib_level} Fibonacci retracement in downtrend'
                            }
            
            # Volume signals
            if 'Volume' in self.df.columns and 'volume_ratio_20' in self.df.columns:
                volume_ratio = latest['volume_ratio_20']
                price_change = latest['Close'] > latest['Open']
                
                signals['volume'] = {
                    'signal': 'BUY' if price_change and volume_ratio > 1.5 else 
                            'SELL' if not price_change and volume_ratio > 1.5 else 'NEUTRAL',
                    'strength': 3 if volume_ratio > 2.0 else 
                            2 if volume_ratio > 1.5 else 1,
                    'description': f'High volume ({volume_ratio:.2f}x average) with {"up" if price_change else "down"} price'
                }
            
            # OBV signals
            if 'obv_bullish' in self.df.columns:
                signals['obv'] = {
                    'signal': 'BUY' if latest['obv_bullish'] else 'SELL',
                    'strength': 2,
                    'description': 'OBV rising above EMA' if latest['obv_bullish'] else 'OBV falling below EMA'
                }
            
            # VWAP signals
            if 'above_vwap' in self.df.columns:
                signals['vwap'] = {
                    'signal': 'BUY' if latest['above_vwap'] else 'SELL',
                    'strength': 2,
                    'description': 'Price above VWAP' if latest['above_vwap'] else 'Price below VWAP'
                }
            
            # Moving Average signals
            if 'sma_20' in self.df.columns and 'sma_50' in self.df.columns:
                signals['ma_crossover'] = {
                    'signal': 'BUY' if latest['sma_20'] > latest['sma_50'] else 'SELL',
                    'strength': 3 if latest.get('sma_20_50_bullish_cross', False) or latest.get('sma_20_50_bearish_cross', False) else 2,
                    'description': '20-period SMA above 50-period SMA' if latest['sma_20'] > latest['sma_50'] else '20-period SMA below 50-period SMA'
                }
            
            # Price vs Moving Average signals
            if 'sma_200' in self.df.columns:
                signals['price_vs_ma'] = {
                    'signal': 'BUY' if latest['Close'] > latest['sma_200'] else 'SELL',
                    'strength': 2,
                    'description': 'Price above 200-period SMA' if latest['Close'] > latest['sma_200'] else 'Price below 200-period SMA'
                }
            
            # RSI signals
            if 'rsi' in self.df.columns:
                if latest['rsi_oversold']:
                    signals['rsi'] = {
                        'signal': 'BUY',
                        'strength': 4,
                        'description': f'RSI oversold at {latest["rsi"]:.2f}'
                    }
                elif latest['rsi_overbought']:
                    signals['rsi'] = {
                        'signal': 'SELL',
                        'strength': 4,
                        'description': f'RSI overbought at {latest["rsi"]:.2f}'
                    }
                elif latest.get('rsi_bullish_divergence', False):
                    signals['rsi'] = {
                        'signal': 'BUY',
                        'strength': 3,
                        'description': 'Bullish RSI divergence'
                    }
                elif latest.get('rsi_bearish_divergence', False):
                    signals['rsi'] = {
                        'signal': 'SELL',
                        'strength': 3,
                        'description': 'Bearish RSI divergence'
                    }
                else:
                    signals['rsi'] = {
                        'signal': 'BUY' if latest['rsi'] < 40 else 'SELL' if latest['rsi'] > 60 else 'NEUTRAL',
                        'strength': 1,
                        'description': f'RSI at {latest["rsi"]:.2f}'
                    }
            
            # MACD signals
            if 'macd_bullish_crossover' in self.df.columns and 'macd_bearish_crossover' in self.df.columns:
                if latest['macd_bullish_crossover']:
                    signals['macd'] = {
                        'signal': 'BUY',
                        'strength': 4,
                        'description': 'MACD line crossed above signal line'
                    }
                elif latest['macd_bearish_crossover']:
                    signals['macd'] = {
                        'signal': 'SELL',
                        'strength': 4,
                        'description': 'MACD line crossed below signal line'
                    }
                elif latest.get('macd_bullish_divergence', False):
                    signals['macd'] = {
                        'signal': 'BUY',
                        'strength': 3,
                        'description': 'Bullish MACD divergence'
                    }
                elif latest.get('macd_bearish_divergence', False):
                    signals['macd'] = {
                        'signal': 'SELL',
                        'strength': 3,
                        'description': 'Bearish MACD divergence'
                    }
                else:
                    # Check MACD histogram direction
                    if 'macd_histogram' in self.df.columns:
                        hist_direction = latest.get('macd_histogram_rising', False)
                        signals['macd'] = {
                            'signal': 'BUY' if hist_direction else 'SELL',
                            'strength': 2,
                            'description': 'MACD histogram increasing' if hist_direction else 'MACD histogram decreasing'
                        }
            
            # Bollinger Bands signals
            if 'bb_oversold' in self.df.columns and 'bb_overbought' in self.df.columns:
                if latest['bb_oversold']:
                    signals['bollinger'] = {
                        'signal': 'BUY',
                        'strength': 3,
                        'description': 'Price below lower Bollinger Band'
                    }
                elif latest['bb_overbought']:
                    signals['bollinger'] = {
                        'signal': 'SELL',
                        'strength': 3,
                        'description': 'Price above upper Bollinger Band'
                    }
                elif latest.get('bb_squeeze', False):
                    signals['bollinger'] = {
                        'signal': 'NEUTRAL',
                        'strength': 2,
                        'description': 'Bollinger Band squeeze (low volatility)'
                    }
                else:
                    # Check if price is moving away from middle band
                    if 'bb_middle' in self.df.columns:
                        distance_to_middle = abs(latest['Close'] - latest['bb_middle'])
                        prev_distance = abs(self.df['Close'].iloc[-2] - self.df['bb_middle'].iloc[-2])
                        moving_outward = distance_to_middle > prev_distance
                        
                        signals['bollinger'] = {
                            'signal': 'BUY' if latest['Close'] > latest['bb_middle'] and moving_outward else 
                                    'SELL' if latest['Close'] < latest['bb_middle'] and moving_outward else 'NEUTRAL',
                            'strength': 1,
                            'description': 'Price moving away from middle band'
                        }
            
            # Stochastic signals
            if 'stoch_bullish_crossover' in self.df.columns and 'stoch_bearish_crossover' in self.df.columns:
                if latest.get('stoch_strong_buy', False):
                    signals['stochastic'] = {
                        'signal': 'BUY',
                        'strength': 4,
                        'description': '%K crossed above %D in oversold territory'
                    }
                elif latest.get('stoch_strong_sell', False):
                    signals['stochastic'] = {
                        'signal': 'SELL',
                        'strength': 4,
                        'description': '%K crossed below %D in overbought territory'
                    }
                elif latest['stoch_bullish_crossover']:
                    signals['stochastic'] = {
                        'signal': 'BUY',
                        'strength': 2,
                        'description': '%K crossed above %D'
                    }
                elif latest['stoch_bearish_crossover']:
                    signals['stochastic'] = {
                        'signal': 'SELL',
                        'strength': 2,
                        'description': '%K crossed below %D'
                    }
                elif latest.get('stoch_bullish_divergence', False):
                    signals['stochastic'] = {
                        'signal': 'BUY',
                        'strength': 3,
                        'description': 'Bullish Stochastic divergence'
                    }
                elif latest.get('stoch_bearish_divergence', False):
                    signals['stochastic'] = {
                        'signal': 'SELL',
                        'strength': 3,
                        'description': 'Bearish Stochastic divergence'
                    }
            
            # StochRSI signals
            if 'stoch_rsi_bullish_crossover' in self.df.columns and 'stoch_rsi_bearish_crossover' in self.df.columns:
                if latest.get('stoch_rsi_strong_buy', False):
                    signals['stoch_rsi'] = {
                        'signal': 'BUY',
                        'strength': 4,
                        'description': 'StochRSI %K crossed above %D in oversold territory'
                    }
                elif latest.get('stoch_rsi_strong_sell', False):
                    signals['stoch_rsi'] = {
                        'signal': 'SELL',
                        'strength': 4,
                        'description': 'StochRSI %K crossed below %D in overbought territory'
                    }
                elif latest['stoch_rsi_bullish_crossover']:
                    signals['stoch_rsi'] = {
                        'signal': 'BUY',
                        'strength': 3,
                        'description': 'StochRSI %K crossed above %D'
                    }
                elif latest['stoch_rsi_bearish_crossover']:
                    signals['stoch_rsi'] = {
                        'signal': 'SELL',
                        'strength': 3,
                        'description': 'StochRSI %K crossed below %D'
                    }
            
            # ADX signals
            if 'adx' in self.df.columns and 'adx_strong_trend' in self.df.columns:
                adx_threshold = self.params.get_indicator_param('adx_threshold')
                if latest.get('adx_strong_buy', False):
                    signals['adx'] = {
                        'signal': 'BUY',
                        'strength': 4,
                        'description': f'Strong uptrend (ADX: {latest["adx"]:.2f}) with increasing +DI'
                    }
                elif latest.get('adx_strong_sell', False):
                    signals['adx'] = {
                        'signal': 'SELL',
                        'strength': 4,
                        'description': f'Strong downtrend (ADX: {latest["adx"]:.2f}) with increasing -DI'
                    }
                elif latest['adx_bullish_crossover']:
                    signals['adx'] = {
                        'signal': 'BUY',
                        'strength': 4,
                        'description': 'Strong trend with +DI crossing above -DI'
                    }
                elif latest['adx_bearish_crossover']:
                    signals['adx'] = {
                        'signal': 'SELL',
                        'strength': 4,
                        'description': 'Strong trend with +DI crossing below -DI'
                    }
                elif latest['adx_strong_trend']:
                    signals['adx'] = {
                        'signal': 'BUY' if latest['plus_di'] > latest['minus_di'] else 'SELL',
                        'strength': 3,
                        'description': f'Strong trend (ADX: {latest["adx"]:.2f}) with {"positive" if latest["plus_di"] > latest["minus_di"] else "negative"} direction'
                    }
                elif latest['adx_weak_trend']:
                    signals['adx'] = {
                        'signal': 'NEUTRAL',
                        'strength': 1,
                        'description': f'Weak trend (ADX: {latest["adx"]:.2f})'
                    }
            
            # Aroon signals
            if 'aroon_bullish' in self.df.columns and 'aroon_bearish' in self.df.columns:
                if latest.get('aroon_strong_bull', False):
                    signals['aroon'] = {
                        'signal': 'BUY',
                        'strength': 4,
                        'description': 'Strong bullish trend (Aroon Up > 70, Aroon Down < 30)'
                    }
                elif latest.get('aroon_strong_bear', False):
                    signals['aroon'] = {
                        'signal': 'SELL',
                        'strength': 4,
                        'description': 'Strong bearish trend (Aroon Down > 70, Aroon Up < 30)'
                    }
                elif latest['aroon_bullish']:
                    signals['aroon'] = {
                        'signal': 'BUY',
                        'strength': 3,
                        'description': 'Bullish trend (Aroon Up > 50, Aroon Down < 30)'
                    }
                elif latest['aroon_bearish']:
                    signals['aroon'] = {
                        'signal': 'SELL',
                        'strength': 3,
                        'description': 'Bearish trend (Aroon Down > 50, Aroon Up < 30)'
                    }
            
            # ATR and Supertrend signals
            if 'supertrend_buy' in self.df.columns and 'supertrend_sell' in self.df.columns:
                if latest['supertrend_buy']:
                    signals['supertrend'] = {
                        'signal': 'BUY',
                        'strength': 4,
                        'description': 'Supertrend changed from bearish to bullish'
                    }
                elif latest['supertrend_sell']:
                    signals['supertrend'] = {
                        'signal': 'SELL',
                        'strength': 4,
                        'description': 'Supertrend changed from bullish to bearish'
                    }
                else:
                    signals['supertrend'] = {
                        'signal': 'BUY' if latest['supertrend_direction'] else 'SELL',
                        'strength': 3,
                        'description': 'In supertrend uptrend' if latest['supertrend_direction'] else 'In supertrend downtrend'
                    }
            
            # Alligator signals
            if 'alligator_buy' in self.df.columns and 'alligator_sell' in self.df.columns:
                if latest['alligator_sleeping']:
                    signals['alligator'] = {
                        'signal': 'NEUTRAL',
                        'strength': 1,
                        'description': 'Alligator sleeping (flat market)'
                    }
                elif latest['alligator_buy']:
                    signals['alligator'] = {
                        'signal': 'BUY',
                        'strength': 3,
                        'description': 'Alligator in feeding phase (uptrend)'
                    }
                elif latest['alligator_sell']:
                    signals['alligator'] = {
                        'signal': 'SELL',
                        'strength': 3,
                        'description': 'Alligator in feeding phase (downtrend)'
                    }
            
            # CPR signals
            if 'above_cpr' in self.df.columns and 'below_cpr' in self.df.columns:
                if latest.get('cpr_breakout_up', False):
                    signals['cpr'] = {
                        'signal': 'BUY',
                        'strength': 3,
                        'description': 'Breakout above CPR top'
                    }
                elif latest.get('cpr_breakout_down', False):
                    signals['cpr'] = {
                        'signal': 'SELL',
                        'strength': 3,
                        'description': 'Breakout below CPR bottom'
                    }
                elif latest['above_cpr']:
                    signals['cpr'] = {
                        'signal': 'BUY',
                        'strength': 2,
                        'description': 'Price above CPR'
                    }
                elif latest['below_cpr']:
                    signals['cpr'] = {
                        'signal': 'SELL',
                        'strength': 2,
                        'description': 'Price below CPR'
                    }
                elif latest['inside_cpr']:
                    signals['cpr'] = {
                        'signal': 'NEUTRAL',
                        'strength': 1,
                        'description': 'Price inside CPR'
                    }
            
            # Chart pattern signals (Dow Theory)
            pattern_signals = {}
            
            if latest.get('double_top', False):
                pattern_signals['double_top'] = {
                    'signal': 'SELL',
                    'strength': 4,
                    'description': 'Double top pattern detected'
                }
            
            if latest.get('double_bottom', False):
                pattern_signals['double_bottom'] = {
                    'signal': 'BUY',
                    'strength': 4,
                    'description': 'Double bottom pattern detected'
                }
            
            if latest.get('triple_top', False):
                pattern_signals['triple_top'] = {
                    'signal': 'SELL',
                    'strength': 5,
                    'description': 'Triple top pattern detected'
                }
            
            if latest.get('triple_bottom', False):
                pattern_signals['triple_bottom'] = {
                    'signal': 'BUY',
                    'strength': 5,
                    'description': 'Triple bottom pattern detected'
                }
            
            if latest.get('in_range', False):
                range_width = latest.get('range_top', 0) - latest.get('range_bottom', 0)
                if range_width > 0:
                    range_position = (latest['Close'] - latest.get('range_bottom', 0)) / range_width
                else:
                    range_position = 0.5
                
                pattern_signals['trading_range'] = {
                    'signal': 'BUY' if range_position < 0.3 else 'SELL' if range_position > 0.7 else 'NEUTRAL',
                    'strength': 2,
                    'description': f'Trading in range ({range_position:.2%} from bottom)'
                }
            
            if latest.get('range_breakout_up', False):
                pattern_signals['range_breakout'] = {
                    'signal': 'BUY',
                    'strength': 4,
                    'description': 'Bullish breakout from trading range'
                }
            
            if latest.get('range_breakout_down', False):
                pattern_signals['range_breakout'] = {
                    'signal': 'SELL',
                    'strength': 4,
                    'description': 'Bearish breakdown from trading range'
                }
            
            if latest.get('bull_flag', False):
                pattern_signals['flag'] = {
                    'signal': 'BUY',
                    'strength': 4,
                    'description': 'Bull flag pattern detected'
                }
            
            if latest.get('bear_flag', False):
                pattern_signals['flag'] = {
                    'signal': 'SELL',
                    'strength': 4,
                    'description': 'Bear flag pattern detected'
                }
            
            # Add pattern signals to the main signals dictionary
            signals.update(pattern_signals)
            
            # Reward to Risk ratio signals
            if 'long_rrr_valid' in self.df.columns and 'short_rrr_valid' in self.df.columns:
                long_rrr = latest.get('long_rrr', 0)
                short_rrr = latest.get('short_rrr', 0)
                
                signals['rrr'] = {
                    'signal': 'BUY' if latest['long_rrr_valid'] and not latest['short_rrr_valid'] else 
                            'SELL' if latest['short_rrr_valid'] and not latest['long_rrr_valid'] else
                            'BUY' if long_rrr > short_rrr else
                            'SELL' if short_rrr > long_rrr else 'NEUTRAL',
                    'strength': 3 if (latest['long_rrr_valid'] and long_rrr > 2.0) or 
                                (latest['short_rrr_valid'] and short_rrr > 2.0) else 2,
                    'description': f'Long RRR: {long_rrr:.2f}, Short RRR: {short_rrr:.2f}'
                }
            
            return signals

        def get_indicator_based_signal(self):
            """
            Get an overall trading signal based on all indicators
            
            Returns:
                Dictionary with overall signal and details
            """
            # Get signals from all indicators
            indicator_signals = self.get_latest_indicator_signals()
            
            if not indicator_signals:
                return {
                    'overall_signal': 'NEUTRAL',
                    'overall_strength': 0,
                    'confidence_score': 0,
                    'buy_count': 0,
                    'sell_count': 0,
                    'neutral_count': 0,
                    'details': {}
                }
            
            # Count buy, sell, and neutral signals
            buy_signals = [sig for sig in indicator_signals.values() if sig['signal'] == 'BUY']
            sell_signals = [sig for sig in indicator_signals.values() if sig['signal'] == 'SELL']
            neutral_signals = [sig for sig in indicator_signals.values() if sig['signal'] == 'NEUTRAL']
            
            buy_count = len(buy_signals)
            sell_count = len(sell_signals)
            neutral_count = len(neutral_signals)
            
            # Calculate total strength of buy and sell signals
            buy_strength = sum(sig['strength'] for sig in buy_signals)
            sell_strength = sum(sig['strength'] for sig in sell_signals)
            
            # Calculate overall signal
            if buy_count > sell_count and buy_strength > sell_strength:
                overall_signal = 'BUY'
                # Calculate strength based on buy signal strengths and margin over sell signals
                overall_strength = min(5, max(1, round(buy_strength / (buy_count + 1))))
                confidence_score = min(10, max(1, round((buy_strength - sell_strength) / 2)))
            elif sell_count > buy_count and sell_strength > buy_strength:
                overall_signal = 'SELL'
                # Calculate strength based on sell signal strengths and margin over buy signals
                overall_strength = min(5, max(1, round(sell_strength / (sell_count + 1))))
                confidence_score = min(10, max(1, round((sell_strength - buy_strength) / 2)))
            else:
                # If nearly equal, check signal strengths
                if buy_strength > sell_strength * 1.5:
                    overall_signal = 'BUY'
                    overall_strength = min(5, max(1, round(buy_strength / (buy_count + sell_count + 1))))
                    confidence_score = min(10, max(1, round((buy_strength - sell_strength) / 3)))
                elif sell_strength > buy_strength * 1.5:
                    overall_signal = 'SELL'
                    overall_strength = min(5, max(1, round(sell_strength / (buy_count + sell_count + 1))))
                    confidence_score = min(10, max(1, round((sell_strength - buy_strength) / 3)))
                else:
                    overall_signal = 'NEUTRAL'
                    overall_strength = 0
                    confidence_score = 0
            
            # Check the RRR for the suggested signal
            if overall_signal == 'BUY' and 'long_rrr_valid' in self.df.columns:
                if not self.df['long_rrr_valid'].iloc[-1]:
                    # RRR below threshold, reduce strength
                    overall_strength = max(1, overall_strength - 1)
                    confidence_score = max(1, confidence_score - 2)
            elif overall_signal == 'SELL' and 'short_rrr_valid' in self.df.columns:
                if not self.df['short_rrr_valid'].iloc[-1]:
                    # RRR below threshold, reduce strength
                    overall_strength = max(1, overall_strength - 1)
                    confidence_score = max(1, confidence_score - 2)
            
            return {
                'overall_signal': overall_signal,
                'overall_strength': overall_strength,
                'confidence_score': confidence_score,
                'buy_count': buy_count,
                'sell_count': sell_count,
                'neutral_count': neutral_count,
                'buy_strength': buy_strength,
                'sell_strength': sell_strength,
                'details': indicator_signals
            }
        
        def validate_pattern_signal(self, pattern_signal):
            """
            Validate candlestick pattern signal using technical indicators
            
            Args:
                pattern_signal: Dictionary with pattern signal information
                
            Returns:
                Dictionary with validated signal information
            """
            # Get indicator signal
            indicator_signal = self.get_indicator_based_signal()
            
            # Extract signal directions
            pattern_direction = pattern_signal['overall_signal']
            indicator_direction = indicator_signal['overall_signal']
            
            # Check if signals agree
            signals_agree = pattern_direction == indicator_direction
            
            # Combined signal calculation
            if signals_agree and pattern_direction != 'NEUTRAL':
                # If signals agree, strengthen the signal
                combined_strength = min(5, pattern_signal['overall_strength'] + 1)
                combined_signal = pattern_direction
                confidence_score = min(10, indicator_signal['confidence_score'] + 2)
            elif pattern_direction != 'NEUTRAL' and indicator_direction != 'NEUTRAL':
                # If signals disagree, compare strengths
                if pattern_signal['overall_strength'] >= indicator_signal['overall_strength']:
                    # Pattern signal is stronger or equal
                    combined_strength = max(1, pattern_signal['overall_strength'] - 1)
                    combined_signal = pattern_direction
                    confidence_score = max(1, indicator_signal['confidence_score'] - 2)
                else:
                    # Indicator signal is stronger
                    combined_strength = max(1, indicator_signal['overall_strength'] - 1)
                    combined_signal = indicator_direction
                    confidence_score = max(1, indicator_signal['confidence_score'] - 2)
            else:
                # If either signal is neutral, use the non-neutral one
                if pattern_direction != 'NEUTRAL':
                    combined_signal = pattern_direction
                    combined_strength = pattern_signal['overall_strength']
                    confidence_score = max(1, indicator_signal['confidence_score'] - 1)
                elif indicator_direction != 'NEUTRAL':
                    combined_signal = indicator_direction
                    combined_strength = indicator_signal['overall_strength']
                    confidence_score = max(1, indicator_signal['confidence_score'] - 1)
                else:
                    combined_signal = 'NEUTRAL'
                    combined_strength = 0
                    confidence_score = 0
            
            # Get supporting and opposing indicators
            supporting_indicators = []
            opposing_indicators = []
            
            for name, indicator in indicator_signal['details'].items():
                if indicator['signal'] == combined_signal:
                    supporting_indicators.append({
                        'name': name,
                        'strength': indicator['strength'],
                        'description': indicator['description']
                    })
                elif indicator['signal'] != 'NEUTRAL' and indicator['signal'] != combined_signal:
                    opposing_indicators.append({
                        'name': name,
                        'strength': indicator['strength'],
                        'description': indicator['description']
                    })
            
            # Build result with detailed information
            result = {
                'combined_signal': combined_signal,
                'combined_strength': combined_strength,
                'confidence_score': confidence_score,
                'signals_agree': signals_agree,
                'pattern_signal': pattern_signal,
                'indicator_signal': indicator_signal,
                'confirmation': signals_agree and pattern_direction != 'NEUTRAL',
                'supporting_indicators': supporting_indicators,
                'opposing_indicators': opposing_indicators,
                'supporting_count': len(supporting_indicators),
                'opposing_count': len(opposing_indicators)
            }
            
            return result
        
        def get_trading_checklist_result(self, pattern_signal):
            """
            Generate a trading checklist result based on the current pattern and indicators
            
            Args:
                pattern_signal: Dictionary with candlestick pattern signal
                
            Returns:
                Dictionary with checklist results and recommendation
            """
            # Get parameters from config
            checklist_high_confidence = self.params.get_signal_param('checklist_high_confidence_threshold')
            checklist_medium_confidence = self.params.get_signal_param('checklist_medium_confidence_threshold')
            checklist_low_confidence = self.params.get_signal_param('checklist_low_confidence_threshold')
            min_rrr = self.params.get_signal_param('min_rrr')
            sr_near_threshold = self.params.get_signal_param('sr_near_threshold_pct')
            
            # Get current price and latest data
            if len(self.df) == 0:
                return {'recommendation': 'NEUTRAL', 'checklist_passed': False, 'reason': 'No data available'}
            
            latest = self.df.iloc[-1]
            current_price = latest['Close']
            
            # 1. Check for recognizable candlestick pattern
            has_pattern = pattern_signal['overall_signal'] != 'NEUTRAL'
            
            # 2. S&R alignment with pattern
            sr_aligned = False
            if has_pattern:
                if pattern_signal['overall_signal'] == 'BUY' and 'nearest_support' in self.df.columns:
                    support_price = latest.get('nearest_support', 0)
                    if not pd.isna(support_price):
                        # For buy signal, low should be near support
                        sr_aligned = (latest['Low'] - support_price) / support_price < sr_near_threshold / 100  # Convert to ratio
                elif pattern_signal['overall_signal'] == 'SELL' and 'nearest_resistance' in self.df.columns:
                    resistance_price = latest.get('nearest_resistance', float('inf'))
                    if not pd.isna(resistance_price):
                        # For sell signal, high should be near resistance
                        sr_aligned = (resistance_price - latest['High']) / latest['High'] < sr_near_threshold / 100  # Convert to ratio
            
            # 3. Volume confirmation
            volume_confirms = latest.get('high_volume', False)
            
            # 4. Indicator confirmation (from validate_pattern_signal)
            indicator_validation = self.validate_pattern_signal(pattern_signal)
            indicators_confirm = indicator_validation['confirmation']
            
            # 5. Reward to Risk Ratio check
            rrr_confirms = False
            
            if pattern_signal['overall_signal'] == 'BUY' and 'long_rrr' in self.df.columns:
                rrr_confirms = latest['long_rrr'] >= min_rrr
                rrr_value = latest['long_rrr']
            elif pattern_signal['overall_signal'] == 'SELL' and 'short_rrr' in self.df.columns:
                rrr_confirms = latest['short_rrr'] >= min_rrr
                rrr_value = latest['short_rrr']
            else:
                rrr_value = 0
            
            # Calculate how many checklist items passed
            checklist_items = [has_pattern, sr_aligned, volume_confirms, indicators_confirm, rrr_confirms]
            passed_count = sum(checklist_items)
            
            # Prepare checklist results
            checklist_results = {
                'has_pattern': has_pattern,
                'sr_aligned': sr_aligned,
                'volume_confirms': volume_confirms,
                'indicators_confirm': indicators_confirm,
                'rrr_confirms': rrr_confirms,
                'rrr_value': rrr_value,
                'passed_count': passed_count,
                'total_count': len(checklist_items)
            }
            
            # Determine recommendation
            if has_pattern:  # Must have a pattern
                if passed_count >= checklist_high_confidence:  # At least 4 out of 5 items passed
                    recommendation = pattern_signal['overall_signal']
                    confidence = 'HIGH'
                    position_size = 'FULL'
                elif passed_count == checklist_medium_confidence:  # 3 items passed
                    recommendation = pattern_signal['overall_signal']
                    confidence = 'MEDIUM'
                    position_size = 'REDUCED'
                elif passed_count == checklist_low_confidence and (sr_aligned or rrr_confirms):  # 2 items with either S&R or RRR
                    recommendation = pattern_signal['overall_signal']
                    confidence = 'LOW'
                    position_size = 'SMALL'
                else:
                    recommendation = 'NEUTRAL'
                    confidence = 'VERY_LOW'
                    position_size = 'NONE'
            else:
                recommendation = 'NEUTRAL'
                confidence = 'NONE'
                position_size = 'NONE'
            
            return {
                'recommendation': recommendation,
                'confidence': confidence,
                'position_size': position_size,
                'checklist_results': checklist_results,
                'validation_details': indicator_validation,
                'pattern_signal': pattern_signal
            }
                                
    # ===============================================================
    # Notification/Alert System
    # ===============================================================

    @asynccontextmanager
    async def get_telegram_bot(token):
        """
        Context manager for Telegram bot to ensure proper resource cleanup
        
        Args:
            token: Telegram bot token
            
        Yields:
            Telegram Bot instance
        """
        try:
            bot = Bot(token=token)
            yield bot
        finally:
            if bot and hasattr(bot, "session") and not bot.session.closed:
                await bot.session.close()

    async def send_telegram_message(message, config, logger, retry_attempts=5):
        """
        Send message to Telegram with retry mechanism and proper resource management
        
        Args:
            message: The message text to send
            config: Configuration with Telegram credentials
            logger: Logger instance
            retry_attempts: Number of retry attempts
            
        Returns:
            True if message was sent successfully, False otherwise
        """
        if not config.get('ENABLE_TELEGRAM_ALERTS', False):
            return False
        
        if not config.get('TELEGRAM_BOT_TOKEN', '') or not config.get('TELEGRAM_CHAT_ID', ''):
            logger.error("Telegram credentials are missing")
            return False
            
        delay = 1  # Initial delay in seconds
        
        for attempt in range(retry_attempts):
            try:
                async with get_telegram_bot(config.get('TELEGRAM_BOT_TOKEN', '')) as bot:
                    # Use MarkdownV2 for formatted messages
                    await bot.send_message(
                        chat_id=config.get('TELEGRAM_CHAT_ID', ''), 
                        text=message, 
                        parse_mode='MarkdownV2'
                    )
                    logger.info(f"Successfully sent telegram message (attempt {attempt+1})")
                    return True
            except Exception as e:
                if "Too Many Requests" in str(e):
                    retry_after = int(str(e).split("retry after ")[-1].split()[0]) if "retry after" in str(e) else delay
                    logger.error(f"Error sending Telegram message: {e}. Retrying in {retry_after} seconds.")
                    await asyncio.sleep(retry_after)
                else:
                    logger.error(f"Error sending Telegram message: {e}. Retrying in {delay} seconds.")
                    await asyncio.sleep(delay)
                    delay *= 2  # Exponential backoff
        
        logger.error(f"Failed to send Telegram message after {retry_attempts} attempts")
        return False


    # ===============================================================
    # Main Trading Signal Generation Logic
    # ===============================================================

    async def analyze_and_generate_signals(config, logger):
        """
        Fetches historical data for symbols in config's STOCK_LIST, performs candlestick pattern analysis,
        and generates trading signals.
        
        Args:
            config: Configuration dictionary with API credentials and settings
            logger: Logger instance
            
        Returns:
            Dictionary with analysis results
            Raises exception if analysis fails
        """
        # Log function start with current UTC time
        current_datetime = datetime.datetime.now()
        logger.info(f"Starting analysis at {current_datetime.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        
        # Create trading parameters object for configuration
        params = TradingParameters(config)
        
        # Current date/time
        current_date_str = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
        logger.info(f"Analysis date: {current_date_str}")
        
        # Calculate date range (based on HISTORICAL_DAYS constant)
        end_date = current_datetime.strftime('%Y-%m-%d')
        start_date = (current_datetime - datetime.timedelta(days=config.get('HISTORICAL_DAYS', 100))).strftime('%Y-%m-%d')
        logger.info(f"Analyzing data from {start_date} to {end_date}")

        # Initialize Upstox API client
        try:
            market_api, api_client = initialize_upstox(config, logger)
        except APIConnectionError as e:
            logger.error(f"Failed to initialize Upstox client: {str(e)}")
            raise
        
        # Track overall statistics
        analysis_results = {
            'successful_analyses': 0,
            'failed_analyses': 0,
            'total_signals': 0,
            'buy_signals': 0,
            'sell_signals': 0,
            'symbols_analyzed': [],
            'signals_generated': []
        }
        
        # Header for daily report
        daily_report = [
            f"📈 CANDLESTICK PATTERN ANALYSIS REPORT 📉",
            f"Date: {current_date_str} UTC",
            f"Analyzing {len(config.get('STOCK_LIST', []))} symbols with {config.get('HISTORICAL_DAYS', 100)} days of historical data",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        ]
        
        # Process each symbol in STOCK_LIST
        for symbol in config.get('STOCK_LIST', []):
            logger.info(f"Processing symbol: {symbol}")
            
            try:
                # Get stock information
                stock_info = get_stock_info_by_key(symbol, config.get('STOCK_INFO', {}))
                company_name = stock_info.get("name", "")
                industry = stock_info.get("industry", "")
                trading_symbol = stock_info.get("symbol", symbol.split("|")[-1] if "|" in symbol else symbol)
                
                logger.info(f"Analyzing {company_name} ({trading_symbol}) - {industry}")
                
                # Fetch historical data with daily interval
                try:
                    data = fetch_ohlcv_data(
                        market_api, 
                        symbol, 
                        start_date, 
                        end_date, 
                        interval=config.get('CHART_INTERVAL', 'day'),
                        logger=logger
                    )
                except (DataFetchError, EmptyDataError) as e:
                    logger.error(f"Failed to fetch data for {symbol}: {str(e)}")
                    analysis_results['failed_analyses'] += 1
                    daily_report.append(f"\n{company_name} ({trading_symbol}): Error fetching data - {str(e)}")
                    continue
                
                logger.info(f"Analyzing {company_name} ({trading_symbol}) with {len(data)} data points")
                
                # Create parameter instance
                trading_params = TradingParameters(config)
                
                # Perform candlestick pattern analysis
                pattern_analyzer = CandlestickPatterns(data, params=trading_params)
                latest_patterns = pattern_analyzer.get_latest_patterns()
                pattern_signals = pattern_analyzer.get_pattern_signals()
                
                # Perform technical indicator analysis
                indicator_analyzer = TechnicalIndicators(data, params=trading_params)
                indicator_analyzer.calculate_all()
                
                # Get the validation and trading checklist
                trading_check = indicator_analyzer.get_trading_checklist_result(pattern_signals)
                
                # Extract information about detected patterns
                detected_patterns = [pattern for pattern, is_detected in latest_patterns.items() if is_detected]
                
                # Get signal information
                overall_signal = trading_check['recommendation']
                overall_strength = trading_check['validation_details']['combined_strength']
                confidence = trading_check['confidence']
                
                # Original pattern stats
                buy_signals_count = pattern_signals['buy_signals_count']
                sell_signals_count = pattern_signals['sell_signals_count']
                
                # Create signal summary
                signal_summary = f"{'Bullish' if overall_signal == 'BUY' else 'Bearish' if overall_signal == 'SELL' else 'Neutral'} " + \
                                f"signal with {confidence} confidence ({buy_signals_count} buy vs {sell_signals_count} sell patterns)"
                
                # Check if signal is strong enough to report
                min_signal_strength = params.get_signal_param('min_signal_strength')
                if overall_signal != 'NEUTRAL' and overall_strength >= min_signal_strength:
                    logger.info(f"Generated signal for {company_name} ({trading_symbol}): {overall_signal} (Strength: {overall_strength}/5)")
                    analysis_results['total_signals'] += 1
                    
                    if overall_signal == 'BUY':
                        analysis_results['buy_signals'] += 1
                    elif overall_signal == 'SELL':
                        analysis_results['sell_signals'] += 1
                    
                    # Store signal in results
                    analysis_results['signals_generated'].append({
                        'symbol': trading_symbol,
                        'company': company_name,
                        'signal': overall_signal,
                        'strength': overall_strength,
                        'confidence': confidence,
                        'price': data['Close'].iloc[-1],
                        'patterns': detected_patterns[:3]  # Top 3 patterns
                    })
                    
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
                        
                    # Get supporting indicators
                    supporting_indicators = trading_check['validation_details']['supporting_indicators']
                    if supporting_indicators:
                        indicators_text = ["Supporting indicators:"]
                        for ind in supporting_indicators[:3]:  # Top 3 indicators
                            indicators_text.append(f"- {ind['name'].title()}: {ind['description']}")
                        daily_report.append("\n".join(indicators_text))
                    
                    # Format message for Telegram notification
                    message = f"""
    *📊 CANDLESTICK PATTERN SIGNAL | {trading_symbol} | {overall_signal}* {"⭐" * overall_strength}

    *{company_name}*
    Price: ₹{data['Close'].iloc[-1]:.2f} | Industry: {industry}

    *DETECTED PATTERNS:*
    {chr(10).join(f"✅ {pattern.replace('_', ' ').title()}" for pattern in detected_patterns[:3])}

    *SIGNAL SUMMARY:*
    {signal_summary}

    *TREND CONTEXT:*
    {("In a clear uptrend" if not data.get('uptrend', pd.Series()).empty and data.get('uptrend', pd.Series()).iloc[-1] else 
    "In a clear downtrend" if not data.get('downtrend', pd.Series()).empty and data.get('downtrend', pd.Series()).iloc[-1] else 
    "No clear trend")}

    

    *SUPPORTING INDICATORS:*
    {chr(10).join(f"• {ind['name'].title()}: {ind['description']}" for ind in supporting_indicators[:3])}

    *TRADING CHECKLIST:*
    {chr(10).join([
        f"• Pattern Recognition: {'✅' if trading_check['checklist_results']['has_pattern'] else '❌'}",
        f"• S/R Alignment: {'✅' if trading_check['checklist_results']['sr_aligned'] else '❌'}",
        f"• Volume Confirmation: {'✅' if trading_check['checklist_results']['volume_confirms'] else '❌'}",
        f"• Indicator Confirmation: {'✅' if trading_check['checklist_results']['indicators_confirm'] else '❌'}",
        f"• Risk:Reward Valid: {'✅' if trading_check['checklist_results']['rrr_confirms'] else '❌'}"
    ])}

    Generated: {datetime.datetime.now().strftime("%b-%d %H:%M")}
                    """
                    
                    # Escape for Telegram markdown
                    escaped_message = escape_telegram_markdown(message)
                    await send_telegram_message(escaped_message, config, logger)
                    logger.info(f"Sent pattern alert for {company_name} ({trading_symbol})")
                    
                else:
                    if detected_patterns:
                        logger.info(f"Patterns detected for {company_name} ({trading_symbol}) but signal strength ({overall_strength}) below threshold")
                        patterns_str = ", ".join(p.replace('_', ' ').title() for p in detected_patterns)
                        daily_report.append(f"\n{company_name} ({trading_symbol}): Detected {patterns_str} - Below signal threshold")
                    else:
                        logger.info(f"No significant patterns detected for {company_name} ({trading_symbol})")
                        daily_report.append(f"\n{company_name} ({trading_symbol}): No significant patterns")
                
                analysis_results['successful_analyses'] += 1
                analysis_results['symbols_analyzed'].append(trading_symbol)
                    
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                analysis_results['failed_analyses'] += 1
                daily_report.append(f"\n{symbol}: Error during analysis - {str(e)[:50]}...")
                continue
        
        # Finalize daily report
        daily_report.append("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        daily_report.append("Analysis Summary:")
        daily_report.append(f"• Analyzed: {analysis_results['successful_analyses']} symbols")
        daily_report.append(f"• Failed: {analysis_results['failed_analyses']} symbols") 
        daily_report.append(f"• Total signals generated: {analysis_results['total_signals']} ({analysis_results['buy_signals']} BUY, {analysis_results['sell_signals']} SELL)")
        daily_report.append(f"• Report time: {current_date_str} UTC")
        
        # Store report in results
        analysis_results['daily_report'] = daily_report
        
        # Send daily summary report via Telegram
        if analysis_results['successful_analyses'] > 0 and config.get('ENABLE_DAILY_REPORT', True):
            escaped_report = escape_telegram_markdown("\n".join(daily_report))
            await send_telegram_message(escaped_report, config, logger)
        
        # Log summary statistics
        logger.info(f"Analysis completed. Processed {len(config.get('STOCK_LIST', []))} symbols.")
        logger.info(f"Successful analyses: {analysis_results['successful_analyses']}")
        logger.info(f"Failed analyses: {analysis_results['failed_analyses']}")
        logger.info(f"Total signals generated: {analysis_results['total_signals']} ({analysis_results['buy_signals']} BUY, {analysis_results['sell_signals']} SELL)")
        logger.info(f"Analysis completed at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        
        return analysis_results


# ===============================================================
# System and Connection Test Functions
# ===============================================================
# ===============================================================
# Market Data Handling - Upstox Connection
# ===============================================================

def initialize_upstox(config, logger):
    """
    Initialize connection to Upstox API
    
    Args:
        config: Configuration dictionary with API credentials
        logger: Logger instance
    
    Returns:
        Tuple of (MarketQuoteApi, ApiClient) if successful
        Raises APIConnectionError if initialization fails
    """
    try:
        api_client = ApiClient()
        api_client.configuration.access_token = config.get('UPSTOX_ACCESS_TOKEN', '')
        market_api = MarketQuoteApi(api_client)
        logger.info("✅ Successfully initialized Upstox API client")
        return market_api, api_client
    except Exception as e:
        logger.error(f"Error initializing Upstox API client: {e}")
        raise APIConnectionError(f"Failed to connect to Upstox API: {str(e)}")
        
def test_upstox_connection(config):
    """
    Test connection to Upstox API (compatibility version)
    
    Args:
        config: Configuration dictionary with API credentials
    
    Returns:
        True if connection is successful, False otherwise
    """
    # Set up logger
    logger = setup_logging(config)
    logger.info("Testing Upstox API connection...")
    
    try:
        market_api, api_client = initialize_upstox(config, logger)
        logger.info("✅ Successfully initialized Upstox API client")
        return True
    except Exception as e:
        logger.error(f"❌ Error connecting to Upstox API: {str(e)}")
        return False

def test_upstox_connection(config, logger=None):
    """
    Test connection to Upstox API
    
    Args:
        config: Configuration dictionary with API credentials
        logger: Logger instance
    
    Returns:
        True if connection is successful, False otherwise
    """
    logger.info("Testing Upstox API connection...")
    
    try:
        market_api, api_client = initialize_upstox(config, logger)
        logger.info("✅ Successfully initialized Upstox API client")
        return True
    except APIConnectionError as e:
        logger.error(f"❌ Error connecting to Upstox API: {str(e)}")
        return False

async def test_telegram_connection(config, logger):
    """
    Test connection to Telegram API
    
    Args:
        config: Configuration dictionary with Telegram credentials
        logger: Logger instance
    
    Returns:
        True if connection is successful, False otherwise
    """
    logger.info("Testing Telegram API connection...")
    
    if not config.get('ENABLE_TELEGRAM_ALERTS', False):
        logger.info("❌ Telegram notifications are disabled in config")
        return False
    
    try:
        # Escape the test message properly for MarkdownV2
        test_message = escape_telegram_markdown("🔍 Test Message - Candlestick Pattern Bot connection test successful!")
        result = await send_telegram_message(test_message, config, logger)
        
        if result:
            logger.info("✅ Successfully sent test message to Telegram")
            return True
        else:
            logger.error("❌ Failed to send test message to Telegram")
            return False
    except Exception as e:
        logger.error(f"❌ Error connecting to Telegram API: {str(e)}")
        return False

async def send_startup_notification(config, logger):
    """
    Send a startup notification via Telegram
    
    Args:
        config: Configuration dictionary with Telegram credentials
        logger: Logger instance
    
    Returns:
        True if notification was sent successfully, False otherwise
    """
    if not config.get('ENABLE_TELEGRAM_ALERTS', False):
        logger.info("Telegram alerts disabled, skipping startup notification")
        return False
        
    try:
        # Current time in UTC format
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Construct the notification message
        message = f"""
🚀 Enhanced Trading Signal Bot Started 🚀

Version: {config.get('VERSION', '3.5.1')}
Started at: {current_time}
Analysis Frequency: Every {config.get('ANALYSIS_FREQUENCY', 1)} hour(s)
Stocks Monitored: {len(config.get('STOCK_LIST', []))} stocks

Key Features:
• Complete candlestick pattern recognition system
• 20+ technical indicators implemented
• Advanced trend context validation
• Enhanced signal confirmation algorithm
• Full trading checklist evaluation

Bot is now actively monitoring for trading signals.
        """
        
        # Escape the message for Telegram's MarkdownV2 format
        escaped_message = escape_telegram_markdown(message)
        
        # Send the message
        result = await send_telegram_message(escaped_message, config, logger)
        
        if result:
            logger.info("Startup notification sent successfully")
        else:
            logger.warning("Startup notification delivery failed")
            
        return result
        
    except Exception as e:
        logger.error(f"Failed to send startup notification: {str(e)}")
        logger.debug(traceback.format_exc())
        return False

 # ===============================================================
# Main Execution Function
# ===============================================================

async def run_trading_signals(config, logger):
    """
    Run the trading signal generation process
    
    Args:
        config: Configuration dictionary with all settings
        logger: Logger instance
            
    Returns:
        Dictionary with analysis results or None if failed
    """
    start_time = time.time()
    logger.info("Starting candlestick pattern analysis")
    
    try:
        # Run the analysis
        # The placeholder has been removed - use the global implementation
        result = await analyze_and_generate_signals(config, logger)
        
        # Log completion
        elapsed_time = time.time() - start_time
        logger.info(f"Completed candlestick pattern analysis in {elapsed_time:.2f} seconds")
        return result
    
    except Exception as e:
        logger.error(f"Error in candlestick pattern analysis: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Send error notification if enabled
        if config.get('ENABLE_TELEGRAM_ALERTS', False):
            try:
                # Format the current time
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Create error message
                error_message = f"""
⚠️ ERROR: Candlestick Pattern Bot Failure ⚠️
    
Time: {current_time}
Error: {str(e)}
    
Please check the logs for more details.
                """
                
                # Escape the message for Telegram's MarkdownV2 format
                escaped_message = escape_telegram_markdown(error_message)
                
                # Send the notification
                await send_telegram_message(escaped_message, config, logger)
            except Exception as notification_err:
                logger.error(f"Failed to send error notification: {notification_err}")
        
        return None

# ===============================================================
# Execution Scheduler with APScheduler
# ===============================================================

def create_scheduler(config, logger):
    """
    Create an APScheduler instance for scheduled analysis
    
    Args:
        config: Configuration dictionary with scheduling parameters
        logger: Logger instance
        
    Returns:
        Configured APScheduler instance or None if error
    """
    try:
        # Import APScheduler components
        try:
            from apscheduler.schedulers.asyncio import AsyncIOScheduler
            from apscheduler.triggers.cron import CronTrigger
            from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED
        except ImportError:
            logger.critical("Required dependency 'apscheduler' not found. Please install it using: pip install apscheduler")
            logger.critical("Exiting application due to missing dependencies")
            return None
        
        # Create scheduler
        scheduler = AsyncIOScheduler()
        logger.info("Created AsyncIOScheduler for job scheduling")
        
        # Define async wrapper for job execution
        async def scheduled_analysis_job():
            logger.info("Running scheduled trading signals analysis")
            await run_trading_signals(config, logger)
        
        # Schedule for specific hours of the day based on market hours
        market_open_hour = config.get('MARKET_OPEN_HOUR', 9)
        market_close_hour = config.get('MARKET_CLOSE_HOUR', 15)
        market_days = config.get('MARKET_DAYS', ['mon', 'tue', 'wed', 'thu', 'fri'])
        frequency = config.get('ANALYSIS_FREQUENCY', 1)  # hours
        
        # Create comma-separated string of days
        days_str = ','.join(market_days)
        
        # Add job listener for logging
        def job_listener(event):
            if event.exception:
                logger.error(f"Job failed: {event.exception}")
            else:
                logger.info(f"Job executed successfully")
        
        scheduler.add_listener(job_listener, EVENT_JOB_ERROR | EVENT_JOB_EXECUTED)
        
        # Schedule by intervals during market hours
        hours_added = 0
        for hour in range(market_open_hour, market_close_hour + 1, frequency):
            job_id = f"analysis_job_{hour}"
            scheduler.add_job(
                scheduled_analysis_job,
                CronTrigger(hour=hour, minute=0, day_of_week=days_str),
                id=job_id,
                replace_existing=True,
                name=f"Analysis Job at {hour}:00"
            )
            logger.info(f"Scheduled analysis job at {hour}:00 on {days_str}")
            hours_added += 1
        
        # Also schedule at market open and close for important signals
        if config.get('RUN_AT_MARKET_OPEN', True):
            scheduler.add_job(
                scheduled_analysis_job,
                CronTrigger(hour=market_open_hour, minute=15, day_of_week=days_str),
                id="market_open_job",
                replace_existing=True,
                name="Market Open Analysis"
            )
            logger.info(f"Scheduled market open analysis at {market_open_hour}:15 on {days_str}")
        
        if config.get('RUN_AT_MARKET_CLOSE', True):
            scheduler.add_job(
                scheduled_analysis_job,
                CronTrigger(hour=market_close_hour, minute=30, day_of_week=days_str),
                id="market_close_job",
                replace_existing=True,
                name="Market Close Analysis"
            )
            logger.info(f"Scheduled market close analysis at {market_close_hour}:30 on {days_str}")
        
        # Verify that at least one job has been scheduled before returning
        if len(scheduler.get_jobs()) == 0:
            logger.error("No jobs were scheduled. Check market hours and frequency settings.")
            return None
            
        return scheduler
        
    except Exception as e:
        logger.error(f"Error creating scheduler: {str(e)}")
        logger.error(traceback.format_exc())
        return None


    # ===============================================================
    # Entry Point
    # ===============================================================

    async def initialize_and_test(config):
        """
        Initialize the system and test connections
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Tuple of (logger, upstox_ok, telegram_ok)
        """
        # Initialize logger
        logger = setup_logging(config)
        
        # Test connections
        upstox_ok = test_upstox_connection(config, logger)
        telegram_ok = await test_telegram_connection(config, logger) if config.get('ENABLE_TELEGRAM_ALERTS', False) else False
        
        return logger, upstox_ok, telegram_ok

async def main_async(config):
    """
    Main async function to run the Candlestick Pattern Bot
    
    Args:
        config: Configuration dictionary
        
    Returns:
        0 for successful execution, 1 for errors
    """
    global _BOT_INITIALIZED
    
    # Use threading lock to ensure only one instance initializes
    with _INITIALIZATION_LOCK:
        if _BOT_INITIALIZED:
            print("WARNING: Bot already initialized, ignoring duplicate start request")
            return 0
        _BOT_INITIALIZED = True
    
    # Generate unique run ID for this execution
    run_id = str(uuid.uuid4())[:8]
    
    try:
        # Set up logging
        logger = setup_logging(config)
        
        # Update config with current information
        config['RUN_ID'] = run_id
        config['CURRENT_TIME'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        config['CURRENT_USER'] = config.get('USERNAME', 'rahulreddyallu')
        
        # Log startup with unique run ID
        logger.info("=" * 70)
        logger.info(f"[RUN:{run_id}] Quantitative Trading Bot v{config.get('VERSION', '3.5.1')} - Starting")
        logger.info(f"[RUN:{run_id}] Bot run by: {config['CURRENT_USER']} at {config['CURRENT_TIME']} UTC")
        logger.info("=" * 70)
        logger.info(f"[RUN:{run_id}] Configured to analyze {len(config.get('STOCK_LIST', []))} stocks with {config.get('HISTORICAL_DAYS', 100)} days of historical data")
        
        # Single test for Upstox connection
        logger.info(f"[RUN:{run_id}] Testing Upstox API connection...")
        upstox_ok = test_upstox_connection(config, logger)
        
        # Single test for Telegram connection
        telegram_ok = False
        if config.get('ENABLE_TELEGRAM_ALERTS', False):
            logger.info(f"[RUN:{run_id}] Testing Telegram API connection...")
            try:
                # Set a timeout for the Telegram test
                test_message = escape_telegram_markdown(f"🔍 Test Message - Bot {run_id} connection test")
                telegram_task = asyncio.create_task(send_telegram_message(test_message, config, logger))
                result = await asyncio.wait_for(telegram_task, timeout=10.0)
                telegram_ok = result
                if result:
                    logger.info(f"[RUN:{run_id}] ✅ Successfully sent test message to Telegram")
                else:
                    logger.error(f"[RUN:{run_id}] ❌ Failed to send test message to Telegram")
            except asyncio.TimeoutError:
                logger.error(f"[RUN:{run_id}] Telegram connection test timed out after 10 seconds")
                telegram_ok = False
            except Exception as e:
                logger.error(f"[RUN:{run_id}] ❌ Error connecting to Telegram API: {str(e)}")
                logger.error(traceback.format_exc())
                telegram_ok = False
        
        # Check connections
        if not upstox_ok:
            logger.error(f"[RUN:{run_id}] Cannot proceed without Upstox API connection")
            return 1
        
        if not telegram_ok and config.get('ENABLE_TELEGRAM_ALERTS', False):
            logger.warning(f"[RUN:{run_id}] Telegram connection failed - notifications will not be sent")
        
        # Send startup notification if Telegram is enabled
        if config.get('ENABLE_TELEGRAM_ALERTS', False) and telegram_ok:
            logger.info(f"[RUN:{run_id}] Sending startup notification...")
            try:
                notification_task = asyncio.create_task(send_startup_notification(config, logger))
                await asyncio.wait_for(notification_task, timeout=10.0)
                logger.info(f"[RUN:{run_id}] Startup notification sent successfully")
            except asyncio.TimeoutError:
                logger.error(f"[RUN:{run_id}] Startup notification timed out after 10 seconds")
            except Exception as e:
                logger.error(f"[RUN:{run_id}] Error sending startup notification: {str(e)}")
                logger.error(traceback.format_exc())
        
        # Run initial analysis if configured
        if config.get('RUN_ON_STARTUP', True):
            logger.info(f"[RUN:{run_id}] Running initial analysis on startup")
            try:
                analysis_task = asyncio.create_task(run_trading_signals(config, logger))
                await asyncio.wait_for(analysis_task, timeout=60.0)
                logger.info(f"[RUN:{run_id}] Initial analysis completed successfully")
            except asyncio.TimeoutError:
                logger.error(f"[RUN:{run_id}] Initial analysis timed out after 60 seconds")
            except Exception as e:
                logger.error(f"[RUN:{run_id}] Error in initial analysis: {str(e)}")
                logger.error(traceback.format_exc())
        
        # Start scheduler if configured
        if config.get('SCHEDULED_MODE', True):
            try:
                logger.info(f"[RUN:{run_id}] Creating scheduler...")
                scheduler = create_scheduler(config, logger)
                
                if scheduler is None:
                    logger.error(f"[RUN:{run_id}] Failed to create scheduler - exiting scheduled mode")
                else:
                    logger.info(f"[RUN:{run_id}] Starting scheduler...")
                    scheduler.start()
                    logger.info(f"[RUN:{run_id}] Scheduler started successfully - waiting for scheduled events")
                    
                    # Keep the event loop running
                    try:
                        logger.info(f"[RUN:{run_id}] Entering main event loop")
                        counter = 0
                        while True:
                            await asyncio.sleep(60)
                            counter += 1
                            if counter % 60 == 0:  # Log once per hour
                                logger.info(f"[RUN:{run_id}] Bot running normally - uptime: {counter} minutes")
                    except (KeyboardInterrupt, SystemExit):
                        logger.info(f"[RUN:{run_id}] Bot stopped by user")
                        scheduler.shutdown()
                        return 0
            except Exception as e:
                logger.error(f"[RUN:{run_id}] Unexpected error in scheduler: {str(e)}")
                logger.error(traceback.format_exc())
                return 1
        else:
            logger.info(f"[RUN:{run_id}] Scheduled mode disabled - exiting after initial analysis")
        
        logger.info(f"[RUN:{run_id}] Bot completed execution successfully")
        return 0
        
    except Exception as e:
        # Handle any uncaught exceptions
        try:
            logger.error(f"[RUN:{run_id}] Unhandled exception in main_async: {str(e)}")
            logger.error(traceback.format_exc())
        except NameError:
            # If logger isn't defined yet, use basic logging
            print(f"FATAL ERROR: {str(e)}")
            print(traceback.format_exc())
        return 1

def main(config=None):
    """
    Main function with two-stage initialization support
    
    Args:
        config: Configuration dictionary
        
    Returns:
        0 for successful execution, 1 for errors
    """
    global _BOT_SETUP_COMPLETE, _BOT_RUNNING, _BOT_INSTANCE_UUID
    
    # Generate a unique ID for this specific call to main()
    call_id = str(uuid.uuid4())[:6]
    call_time = datetime.datetime.now().isoformat()
    
    # Capture stack trace to see who called main()
    import inspect
    caller_stack = inspect.stack()
    caller_info = f"Called from: {caller_stack[1].filename}:{caller_stack[1].lineno}"
    
    # Print diagnostic info
    print(f"\n=== BOT STARTUP REQUEST [{call_id}] at {call_time} ===")
    print(f"Instance UUID: {_BOT_INSTANCE_UUID[:8]}")
    print(f"Setup complete: {_BOT_SETUP_COMPLETE}, Running: {_BOT_RUNNING}")
    print(f"Caller: {caller_info}")
    
    # Acquire the lock to check/update initialization state
    with _INITIALIZATION_LOCK:
        # Check if we're already fully running
        if _BOT_RUNNING:
            print(f"[{call_id}] Bot is already running with scheduler - ignoring duplicate request")
            return 0
            
        # Check if we're in the second initialization phase
        if _BOT_SETUP_COMPLETE and not _BOT_RUNNING:
            print(f"[{call_id}] Setup is complete - proceeding to start scheduler (stage 2)")
            _BOT_RUNNING = True
        
        # Check if we're in the first initialization phase
        elif not _BOT_SETUP_COMPLETE:
            print(f"[{call_id}] First initialization call - setting up (stage 1)")
            _BOT_SETUP_COMPLETE = True
        
        # More than 2 calls detected
        else:
            print(f"[{call_id}] WARNING: Unexpected initialization state")
    
    try:
        # Set up default configuration if not provided
        if config is None:
            config = {
                'VERSION': '3.5.1',
                'LOG_DIRECTORY': 'logs',
                'HISTORICAL_DAYS': 100,
                'CHART_INTERVAL': 'day',
                'ENABLE_TELEGRAM_ALERTS': False,
                'ENABLE_DAILY_REPORT': True,
                'MARKET_OPEN_HOUR': 9,
                'MARKET_CLOSE_HOUR': 15,
                'MARKET_DAYS': ['mon', 'tue', 'wed', 'thu', 'fri'],
                'ANALYSIS_FREQUENCY': 1,
                'RUN_ON_STARTUP': True,
                'SCHEDULED_MODE': True,
                'RUN_AT_MARKET_OPEN': True,
                'RUN_AT_MARKET_CLOSE': True,
                'STOCK_LIST': [],
                'STOCK_INFO': {}
            }
        
        # Add instance and time information to config
        config['INSTANCE_UUID'] = _BOT_INSTANCE_UUID
        config['STARTUP_TIME'] = "2025-05-01 18:03:30"  # Using the time you provided
        config['CURRENT_USER'] = "rahulreddyallu"
        config['INITIALIZATION_STAGE'] = 2 if _BOT_RUNNING else 1
        
        # Initialize library logging with minimal settings
        logging.basicConfig(level=logging.WARNING, 
                           format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Determine the best way to run the async code
        try:
            # If we're in IPython or a Jupyter notebook
            import nest_asyncio
            nest_asyncio.apply()
            print(f"[{call_id}] Applied nest_asyncio patch for Jupyter environment")
            
            # Get the event loop
            loop = asyncio.get_event_loop()
            
            # Run the async main function
            print(f"[{call_id}] Running main_async function (stage {config['INITIALIZATION_STAGE']})")
            return loop.run_until_complete(main_async(config))
            
        except ImportError:
            # Standard Python environment
            print(f"[{call_id}] Standard Python environment detected")
            
            # Run the async main function
            print(f"[{call_id}] Running main_async function with asyncio.run()")
            return asyncio.run(main_async(config))
            
    except Exception as e:
        print(f"[{call_id}] FATAL ERROR in main function: {str(e)}")
        print(traceback.format_exc())
        
        # Reset initialization state on fatal error
        with _INITIALIZATION_LOCK:
            if _BOT_RUNNING:
                _BOT_RUNNING = False
            else:
                _BOT_SETUP_COMPLETE = False
        
        return 1

if __name__ == "__main__":
    # Example configuration for direct execution
    config = {
        'VERSION': '3.5.1',
        'LOG_DIRECTORY': 'logs',
        'HISTORICAL_DAYS': 100,
        'CHART_INTERVAL': 'day',
        # Add required API credentials and stock list here
        'STOCK_LIST': [],
        'UPSTOX_ACCESS_TOKEN': '',  # Add your token here
        'ENABLE_TELEGRAM_ALERTS': False,
        'TELEGRAM_BOT_TOKEN': '',   # Add your token here
        'TELEGRAM_CHAT_ID': '',     # Add your chat ID here
    }
    
    # Run the bot
    exit_code = main(config)
    sys.exit(exit_code)
