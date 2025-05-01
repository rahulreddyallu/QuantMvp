# Enhanced Quantitative Analysis Trading Bot

![Version](https://img.shields.io/badge/version-3.5.1-blue)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Overview

The Enhanced Quantitative Analysis Trading Bot is a sophisticated market analysis tool that combines traditional candlestick pattern recognition with advanced technical indicators to generate high-confidence trading signals. The system is designed to provide actionable insights for stock market trading based on rigorous technical analysis.

**Version:** 3.5.1  
**Author:** rahulreddyallu  
**Last Updated:** 2025-05-01

## Features

### Core Functionality

- Comprehensive candlestick pattern recognition (20+ patterns)
- Advanced technical indicator analysis (25+ indicators)
- Signal confirmation through multi-factor validation
- Risk/reward calculation for position sizing
- Automated trading signal generation with confidence metrics
- Real-time notifications via Telegram
- Scheduled execution around market hours
- Comprehensive logging and reporting

### Pattern Recognition

The bot detects both single and multi-candle patterns including:

**Single-Candle Patterns:**
- Doji (indecision marker)
- Marubozu (strong trend indicator)
- Spinning Tops (uncertainty marker)
- Hammer/Hanging Man (reversal signals)
- Shooting Star (bearish reversal indicator)
- Paper Umbrella (potential support indication)

**Multi-Candle Patterns:**
- Engulfing patterns (strong reversal indication)
- Harami patterns (potential reversal signal)
- Piercing/Dark Cloud Cover (reversal patterns)
- Morning/Evening Stars (major reversal signals)
- Double/Triple tops and bottoms (major reversal patterns)
- Bull/Bear flag patterns (continuation patterns)

### Technical Indicators

The bot calculates and interprets numerous technical indicators:

**Price Level Indicators:**
- Dynamic support/resistance identification
- Fibonacci retracement levels

**Volume Indicators:**
- Volume ratio analysis
- On-Balance Volume (OBV)
- Volume Weighted Average Price (VWAP)

**Moving Averages:**
- Simple Moving Averages (SMA) [5, 10, 20, 50, 200]
- Exponential Moving Averages (EMA) [5, 12, 26, 50]
- Moving Average Crossover detection

**Oscillators:**
- Relative Strength Index (RSI)
- Moving Average Convergence Divergence (MACD)
- Stochastic Oscillator
- Stochastic RSI
- Average Directional Index (ADX)
- Aroon Indicator

**Volatility Indicators:**
- Bollinger Bands
- Average True Range (ATR)
- SuperTrend

**Advanced Indicators:**
- Alligator Indicator
- Central Pivot Range (CPR)

## Prerequisites

- Python 3.7 or newer
- Upstox API access token
- Telegram bot token (optional, for notifications)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/rahulreddyallu/QuantMvp.git
   cd QuantMvp

```Install required dependencies:

bash
pip install -r requirements.txt
Or install dependencies manually:

bash
pip install aiogram upstox_client pandas numpy apscheduler
Configure the bot (see Configuration section below)

Run the bot:

bash
python main.py
Configuration

The bot uses a centralized configuration system in config.py. Key configuration parameters include:

API Credentials

Python
# Upstox API token
UPSTOX_ACCESS_TOKEN = 'your_upstox_token_here'

# Telegram notification settings (optional)
ENABLE_TELEGRAM_ALERTS = True
TELEGRAM_BOT_TOKEN = 'your_telegram_bot_token'
TELEGRAM_CHAT_ID = 'your_telegram_chat_id'
Market Configuration

Python
# Historical data settings
HISTORICAL_DAYS = 100  # Number of days of historical data
CHART_INTERVAL = 'day'  # Options: '1minute', '5minute', '30minute', 'day', 'week', 'month'

# Market hours
MARKET_OPEN_HOUR = 9
MARKET_CLOSE_HOUR = 15
MARKET_DAYS = ['mon', 'tue', 'wed', 'thu', 'fri']
Execution Configuration

Python
# How to run the bot
ANALYSIS_FREQUENCY = 1  # Hours between analyses during market hours
RUN_ON_STARTUP = True   # Run analysis when bot starts
SCHEDULED_MODE = True   # Run on a schedule
RUN_AT_MARKET_OPEN = True  # Run at market open
RUN_AT_MARKET_CLOSE = True  # Run at market close
Stock List Configuration

Python
# List of stocks to monitor (Upstox instrument keys)
STOCK_LIST = [
    'NSE_EQ|INE009A01021',  # INFOSYS
    'NSE_EQ|INE030A01027',  # HDFC Bank
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
    # Add more stock information as needed
}
Pattern Detection Thresholds

You can fine-tune the pattern detection sensitivity by adjusting the threshold parameters in config.py.

Technical Indicator Parameters

You can customize indicator calculation by adjusting parameters like RSI periods, MACD settings, etc. in config.py.

Command Line Options

The bot can be run with various command line options:

bash
# Run with custom Upstox token
python main.py --access-token YOUR_UPSTOX_TOKEN

# Run with custom Telegram settings
python main.py --telegram-token YOUR_TELEGRAM_TOKEN --telegram-chat-id YOUR_CHAT_ID

# Run once and exit (no scheduling)
python main.py --run-once

# Disable Telegram notifications
python main.py --disable-telegram

# Specify log directory
python main.py --log-dir ./custom_logs

# Show version and exit
python main.py --version
Operation Flow

Initialization Phase

The bot loads configuration from config.py
A logging system is established for operation tracking
Connections to Upstox API and Telegram API (if enabled) are tested
A startup notification is sent (if Telegram is configured)
Analysis Phase

For each stock in the configured list:

Historical OHLCV data is retrieved from Upstox
Candlestick patterns are identified using precise mathematical criteria
Technical indicators are calculated and interpreted
Pattern signals are validated against indicator confirmation
A consolidated signal is generated with strength and confidence metrics
Signals that meet strength thresholds trigger notifications
Notification System

For significant signals, the bot:

Creates detailed reports with pattern information and indicator confirmation
Calculates risk/reward ratios and position sizing recommendations
Formats and sends alerts via Telegram
Generates a daily summary report of all analyses
Scheduling System

The bot can run:

On demand (one-time analysis)
On a schedule based on market hours
At specific times like market open/close
At regular intervals throughout trading days
Signal Validation System

The bot uses a comprehensive checklist to validate signals:

Pattern Recognition: Identification of significant candlestick patterns
Support/Resistance Alignment: Proximity to key support/resistance levels
Volume Confirmation: Above-average volume supporting the signal
Indicator Confirmation: Technical indicators confirming the signal direction
Risk/Reward Validation: Favorable risk-to-reward ratio for the trade
Based on how many checklist items pass, signals are classified as:

HIGH confidence: 4-5 items pass
MEDIUM confidence: 3 items pass
LOW confidence: 2 items pass (with S/R or RRR valid)
NEUTRAL: Fewer than 2 items pass
Output Examples

Telegram Alert Example

Code
📊 CANDLESTICK PATTERN SIGNAL | INFY | BUY ⭐⭐⭐⭐

Infosys Ltd
Price: ₹1857.25 | Industry: Information Technology

DETECTED PATTERNS:
✅ Bullish Engulfing
✅ Morning Star
✅ Hammer

SIGNAL SUMMARY:
Bullish signal with HIGH confidence (3 buy vs 0 sell patterns)

TREND CONTEXT:
In a clear uptrend

SUPPORTING INDICATORS:
• Rsi: RSI oversold at 29.75
• Macd: MACD line crossed above signal line
• Bollinger: Price below lower Bollinger Band

TRADING CHECKLIST:
• Pattern Recognition: ✅
• S/R Alignment: ✅
• Volume Confirmation: ✅
• Indicator Confirmation: ✅
• Risk:Reward Valid: ❌

Generated: May-01 14:30
Daily Report Example

Code
📈 CANDLESTICK PATTERN ANALYSIS REPORT 📉
Date: 2025-05-01 15:00:00 UTC
Analyzing 25 symbols with 100 days of historical data
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Infosys Ltd (INFY): BUY (Strength: 4/5)
Bullish signal with HIGH confidence (3 buy vs 0 sell patterns)
Detected patterns:
✅ Bullish Engulfing (bullish)
✅ Morning Star (bullish)
✅ Hammer (bullish)
Supporting indicators:
- Rsi: RSI oversold at 29.75
- Macd: MACD line crossed above signal line
- Bollinger: Price below lower Bollinger Band

HDFC Bank Ltd (HDFCBANK): SELL (Strength: 3/5)
Bearish signal with MEDIUM confidence (0 buy vs 2 sell patterns)
Detected patterns:
✅ Bearish Engulfing (bearish)
✅ Evening Star (bearish)
Supporting indicators:
- Rsi: RSI overbought at 72.35
- Macd: MACD line crossed below signal line

Tata Motors Ltd (TATAMOTORS): No significant patterns

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Analysis Summary:
• Analyzed: 25 symbols
• Failed: 0 symbols
• Total signals generated: 7 (4 BUY, 3 SELL)
• Report time: 2025-05-01 15:00:00 UTC
Error Handling

The bot implements a robust error handling system:

Custom Exception Hierarchy

Base TradingBotError exception
Specialized exceptions for different error types
Context-aware error messages
API Retry Mechanism

Automatic retries for transient API failures
Exponential backoff for rate limit handling
Graceful degradation for persistent failures
Data Validation

Empty data detection and handling
Data quality checks before analysis
Minimum data requirements for reliable analysis
Operational Continuity

Individual stock failures don't stop the entire process
Comprehensive logging of all errors for diagnosis
Error notifications for critical failures
Performance Considerations

The analysis of each stock is CPU-intensive due to pattern detection
Memory usage scales with the number of stocks and amount of historical data
Consider running on a server with at least 2 CPU cores and 4GB RAM
Typical analysis of 50 stocks takes approximately 2-3 minutes
Architecture

The bot is organized into three main files:

main.py: Entry point that handles command line options and initialization
config.py: Centralized configuration for all parameters and settings
compute.py: Core implementation of all analysis functions and algorithms
The core consists of several major components:

Data Fetching: Handles API interaction and historical data retrieval
Pattern Recognition: Implements candlestick pattern detection algorithms
Technical Analysis: Calculates and interprets technical indicators
Signal Generation: Combines patterns and indicators into actionable signals
Notification System: Formats and delivers alerts via Telegram
Scheduling System: Manages when analyses are performed
Security Considerations

API Tokens

Never commit API tokens to version control
Store tokens securely or use environment variables
Regularly rotate API tokens for security
Data Privacy

All analysis is performed locally; no data is sent to third parties
Telegram messages may contain trading signals and stock information
Ensure your Telegram chat is private or secure
Contributing

To contribute to this project:

Fork the repository
Create a feature branch: git checkout -b new-feature
Make your changes and test thoroughly
Commit your changes: git commit -m 'Add new feature'
Push to the branch: git push origin new-feature
Submit a pull request
Troubleshooting

Common Issues

Issue: Bot fails to connect to Upstox API
Solution: Verify your access token is correct and not expired

Issue: Telegram notifications not working
Solution:

Verify bot token and chat ID are correct
Ensure you've started a conversation with your bot
Check if bot has permission to send messages
Issue: No signals are generated
Solution:

Check if pattern thresholds are too strict
Ensure your stock list contains valid symbols
Verify that historical data is being retrieved correctly
Log Files

The bot creates detailed log files in the configured log directory. Check these logs for:

API connection issues
Data retrieval problems
Analysis exceptions
Signal generation details
Requirements

Python 3.7+
Required Python packages:
pandas
numpy
aiogram
upstox_client
apscheduler
License

This project is licensed under the MIT License - see the LICENSE file for details.

Disclaimer

This software is for educational and research purposes only. It is not intended to provide investment advice. Trading stocks involves risk, and past performance is not indicative of future results. Always conduct your own research before making investment decisions.

Contact

For questions, issues, or feature requests, please contact:

Author: rahulreddyallu
GitHub: https://github.com/rahulreddyallu
```
