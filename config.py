"""
Configuration File for Enhanced Trading Signal Bot
All configurable parameters are defined here
"""

# ===============================================================
# Bot Version
# ===============================================================
VERSION = "3.5.0"

# ===============================================================
# API Credentials
# ===============================================================
# Upstox API access token
UPSTOX_ACCESS_TOKEN = "YOUR_UPSTOX_ACCESS_TOKEN"

# ===============================================================
# Telegram Notification Settings
# ===============================================================
# Enable/disable Telegram alerts
ENABLE_TELEGRAM_ALERTS = True

# Telegram Bot token
TELEGRAM_BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"

# Telegram Chat ID
TELEGRAM_CHAT_ID = "YOUR_TELEGRAM_CHAT_ID"

# Enable daily summary reports
ENABLE_DAILY_REPORT = True

# ===============================================================
# Analysis Parameters
# ===============================================================
# Number of historical days to analyze
HISTORICAL_DAYS = 100

# Chart interval (1minute, 5minute, 30minute, day, week, month)
CHART_INTERVAL = "day"

# Minimum signal strength to trigger an alert (1-5)
MINIMUM_SIGNAL_STRENGTH = 3

# ===============================================================
# Scheduling Settings
# ===============================================================
# Enable scheduled mode
SCHEDULED_MODE = True

# Run on startup (first analysis immediately after starting)
RUN_ON_STARTUP = True

# Analysis frequency in hours during market hours
ANALYSIS_FREQUENCY = 1

# Run at market open
RUN_AT_MARKET_OPEN = True

# Run at market close
RUN_AT_MARKET_CLOSE = True

# Market hours
MARKET_OPEN_HOUR = 9  # 9:00 AM
MARKET_CLOSE_HOUR = 15  # 3:00 PM

# Market days
MARKET_DAYS = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday']

# ===============================================================
# Logging Settings
# ===============================================================
# Log directory
LOG_DIRECTORY = "logs"

# ===============================================================
# Stock List and Information
# ===============================================================
# List of stocks to analyze (instrument keys/symbols)
STOCK_LIST = [
    "NSE_EQ|INE062A01020",  # HDFC Bank
    "NSE_EQ|INE009A01021",  # Infosys
    "NSE_EQ|INE030A01027",  # Reliance Industries
    "NSE_EQ|INE040A01034",  # TCS
    "NSE_EQ|INE467B01029",  # Bharti Airtel
    # Add more stocks as needed
]

# Stock information dictionary (ISIN -> details)
STOCK_INFO = {
    "INE062A01020": {
        "name": "HDFC Bank Ltd",
        "symbol": "HDFCBANK",
        "industry": "Banking",
        "series": "EQ"
    },
    "INE009A01021": {
        "name": "Infosys Ltd",
        "symbol": "INFY",
        "industry": "IT",
        "series": "EQ"
    },
    "INE030A01027": {
        "name": "Reliance Industries Ltd",
        "symbol": "RELIANCE",
        "industry": "Oil & Gas",
        "series": "EQ"
    },
    "INE040A01034": {
        "name": "Tata Consultancy Services Ltd",
        "symbol": "TCS",
        "industry": "IT",
        "series": "EQ"
    },
    "INE467B01029": {
        "name": "Bharti Airtel Ltd",
        "symbol": "BHARTIARTL",
        "industry": "Telecom",
        "series": "EQ"
    },
    # Add more stocks as needed
}

# ===============================================================
# Performance Settings
# ===============================================================
# Threading settings for parallel processing
MAX_WORKER_THREADS = 5

# ===============================================================
# Custom Indicator Parameters
# ===============================================================
# RSI thresholds
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# MACD parameters
MACD_FAST_PERIOD = 12
MACD_SLOW_PERIOD = 26
MACD_SIGNAL_PERIOD = 9

# Bollinger Bands parameters
BB_PERIOD = 20
BB_STD_DEV = 2

# Stochastic parameters
STOCH_K_PERIOD = 14
STOCH_D_PERIOD = 3
STOCH_SLOWING = 3

# ATR parameters
ATR_PERIOD = 14

# SuperTrend parameters
SUPERTREND_PERIOD = 10
SUPERTREND_MULTIPLIER = 3

# Support and Resistance lookback period
SR_LOOKBACK = 100

# Fibonacci retracement lookback
FIBONACCI_LOOKBACK = 100

# Risk-Reward ratio parameters
TARGET_MULTIPLIER = 1.5
STOP_MULTIPLIER = 1.0
MINIMUM_RRR = 1.5
