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
