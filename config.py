"""
Enhanced Quantitative Analysis Engine Configuration
Defines all settings and parameters for the trading bot

Version: 3.5.1
Author: rahulreddyallu
Date: 2025-05-01
"""

import os
import datetime

# Bot version
VERSION = '3.5.1'

# Logging configuration
LOG_DIRECTORY = 'logs'
LOG_LEVEL = 'INFO'

# API credentials
# Upstox API Credentials
UPSTOX_API_KEY = "ad55de1b-c7d1-4adc-b559-3830bf1efd72"
UPSTOX_API_SECRET = "969nyjgapm"
UPSTOX_REDIRECT_URI = "https://localhost"
UPSTOX_ACCESS_TOKEN = "eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiI0TEFGUDkiLCJqdGkiOiI2ODEzOWU2N2NiOWRhMDZiZGU2MDFiNzAiLCJpc011bHRpQ2xpZW50IjpmYWxzZSwiaXNQbHVzUGxhbiI6ZmFsc2UsImlhdCI6MTc0NjExNjE5OSwiaXNzIjoidWRhcGktZ2F0ZXdheS1zZXJ2aWNlIiwiZXhwIjoxNzQ2MTM2ODAwfQ.LwX5Qi_mWBq8nNvCfSGm8tGM_Fv49gK78ej_fAzswYU"

# Telegram notification settings
ENABLE_TELEGRAM_ALERTS = True
TELEGRAM_BOT_TOKEN = "7209852741:AAEf-_f6TeZK1-_R55yq365iU_54rk95y-c"
TELEGRAM_CHAT_ID = "936205208"
ENABLE_DAILY_REPORT = True

# Market configuration
HISTORICAL_DAYS = 100  # Number of days of historical data to fetch
CHART_INTERVAL = 'day'  # Options: '1minute', '5minute', '30minute', 'day', 'week', 'month'
MARKET_OPEN_HOUR = 9
MARKET_CLOSE_HOUR = 15
MARKET_DAYS = ['mon', 'tue', 'wed', 'thu', 'fri']

# Execution configuration
ANALYSIS_FREQUENCY = 1  # hours between analyses during market hours
RUN_ON_STARTUP = True
SCHEDULED_MODE = True
RUN_AT_MARKET_OPEN = True
RUN_AT_MARKET_CLOSE = True

# List of stocks to monitor (Upstox instrument keys)
STOCK_LIST = [
    'NSE_EQ|INE009A01021',  # INFOSYS
    'NSE_EQ|INE030A01027',  # HDFC Bank
    'NSE_EQ|INE062A01020',  # TATA Motors
    # Add more stocks as needed
]

# Information about stocks (for better notifications)
STOCK_INFO = {
    'INE009A01021': {
        'name': 'Infosys Ltd',
        'symbol': 'INFY',
        'industry': 'Information Technology',
        'series': 'EQ'
    },
    'INE030A01027': {
        'name': 'HDFC Bank Ltd',
        'symbol': 'HDFCBANK',
        'industry': 'Banking',
        'series': 'EQ'
    },
    'INE062A01020': {
        'name': 'Tata Motors Ltd',
        'symbol': 'TATAMOTORS',
        'industry': 'Automotive',
        'series': 'EQ'
    },
    # Add more stock information as needed
}

# Pattern detection thresholds
PATTERN_THRESHOLDS = {
    # Doji thresholds
    "doji_body_threshold": 0.1,
    
    # Marubozu thresholds
    "marubozu_shadow_threshold": 0.05,
    "marubozu_body_pct": 0.95,
    
    # Spinning top thresholds
    "spinning_top_body_threshold": 0.25,
    "spinning_top_shadow_threshold": 0.35,
    
    # Hammer & shooting star thresholds
    "hammer_lower_shadow_ratio": 2.0,
    "hammer_upper_shadow_threshold": 0.1,
    
    # Paper umbrella thresholds
    "umbrella_lower_shadow_ratio": 2.0,
    "umbrella_upper_shadow_threshold": 0.1,
    
    # Engulfing pattern tolerance
    "engulfing_body_size_factor": 1.1,
    
    # Harami pattern thresholds  
    "harami_body_size_ratio": 0.6,
    
    # Star pattern thresholds
    "star_body_size_threshold": 0.3,
    "star_body_size_factor": 0.6,
}

# Technical indicator parameters
INDICATOR_PARAMS = {
    # RSI parameters
    "rsi_period": 14,
    "rsi_oversold": 30,
    "rsi_overbought": 70,
    
    # MACD parameters
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    
    # Bollinger Bands parameters
    "bb_period": 20,
    "bb_std_dev": 2,
    
    # ATR parameters
    "atr_period": 14,
    "atr_multiplier": 2,
    
    # Stochastic parameters
    "stoch_k_period": 14,
    "stoch_d_period": 3,
    "stoch_slowing": 3,
    "stoch_oversold": 20,
    "stoch_overbought": 80,
    
    # ADX parameters
    "adx_period": 14,
    "adx_threshold": 25,
    
    # Supertrend parameters
    "supertrend_period": 10,
    "supertrend_multiplier": 3,
    
    # Support/Resistance parameters
    "sr_lookback": 100,
    "sr_price_tolerance": 0.02,
    "sr_window_size": 5,
    
    # Fibonacci parameters
    "fib_lookback": 100,
    
    # Moving Average parameters
    "sma_periods": [5, 10, 20, 50, 200],
    "ema_periods": [5, 12, 26, 50],
    
    # Volume parameters
    "volume_ma_periods": [10, 20, 50],
    "high_volume_threshold": 1.5,
    "low_volume_threshold": 0.5,
}

# Signal generation parameters
SIGNAL_PARAMS = {
    # Minimum RRR threshold for valid signals
    "min_rrr": 1.5,
    
    # ATR multipliers for targets and stops
    "target_multiplier": 1.5,
    "stop_multiplier": 1.0,
    
    # Minimum signal strength threshold
    "min_signal_strength": 3,
    
    # Checklist thresholds
    "checklist_high_confidence_threshold": 4,
    "checklist_medium_confidence_threshold": 3,
    "checklist_low_confidence_threshold": 2,
    
    # Pattern strength factors
    "pattern_strength_weights": {
        "bullish_marubozu": 3, "bearish_marubozu": 3,
        "hammer": 3, "hanging_man": 3, "shooting_star": 3,
        "bullish_engulfing": 4, "bearish_engulfing": 4,
        "bullish_harami": 3, "bearish_harami": 3,
        "piercing_pattern": 3, "dark_cloud_cover": 3,
        "morning_star": 4, "evening_star": 4,
        "doji": 1, "spinning_tops": 1
    },
    
    # Support & Resistance significance thresholds
    "sr_near_threshold_pct": 2.0,
    "sr_very_near_threshold_pct": 1.0,
}

# Chart pattern detection parameters
CHART_PATTERN_PARAMS = {
    # Double top/bottom parameters
    "double_pattern_tolerance": 0.03,
    "double_pattern_lookback": 50,
    
    # Triple top/bottom parameters
    "triple_pattern_tolerance": 0.03,
    "triple_pattern_lookback": 100,
    
    # Trading range parameters
    "range_lookback": 60,
    "range_tolerance": 0.05,
    "range_min_touches": 4,
    
    # Flag pattern parameters
    "flag_lookback": 30,
    "flag_pole_threshold": 0.15,
    "flag_consolidation_threshold": 0.05,
    "flag_max_bars": 15,
}

# Function to create a complete configuration dictionary
def get_config():
    """
    Combine all configuration parameters into a single dictionary
    
    Returns:
        Dictionary with all configuration parameters
    """
    config = {
        'VERSION': VERSION,
        'LOG_DIRECTORY': LOG_DIRECTORY,
        'UPSTOX_ACCESS_TOKEN': UPSTOX_ACCESS_TOKEN,
        'ENABLE_TELEGRAM_ALERTS': ENABLE_TELEGRAM_ALERTS,
        'TELEGRAM_BOT_TOKEN': TELEGRAM_BOT_TOKEN,
        'TELEGRAM_CHAT_ID': TELEGRAM_CHAT_ID,
        'ENABLE_DAILY_REPORT': ENABLE_DAILY_REPORT,
        'HISTORICAL_DAYS': HISTORICAL_DAYS,
        'CHART_INTERVAL': CHART_INTERVAL,
        'MARKET_OPEN_HOUR': MARKET_OPEN_HOUR,
        'MARKET_CLOSE_HOUR': MARKET_CLOSE_HOUR,
        'MARKET_DAYS': MARKET_DAYS,
        'ANALYSIS_FREQUENCY': ANALYSIS_FREQUENCY,
        'RUN_ON_STARTUP': RUN_ON_STARTUP,
        'SCHEDULED_MODE': SCHEDULED_MODE,
        'RUN_AT_MARKET_OPEN': RUN_AT_MARKET_OPEN,
        'RUN_AT_MARKET_CLOSE': RUN_AT_MARKET_CLOSE,
        'STOCK_LIST': STOCK_LIST,
        'STOCK_INFO': STOCK_INFO,
    }
    
    # Add pattern thresholds
    for key, value in PATTERN_THRESHOLDS.items():
        config[key] = value
        
    # Add indicator parameters
    for key, value in INDICATOR_PARAMS.items():
        config[key] = value
        
    # Add signal parameters
    for key, value in SIGNAL_PARAMS.items():
        config[key] = value
        
    # Add chart pattern parameters
    for key, value in CHART_PATTERN_PARAMS.items():
        config[key] = value
        
    return config

# For development/testing
if __name__ == "__main__":
    import json
    
    # Print the configuration in a readable format
    print(json.dumps(get_config(), indent=2, default=str))
