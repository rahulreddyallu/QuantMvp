"""
Technical Indicators Module
Implements popular technical indicators for trading signal validation

Current Date and Time (UTC): 2025-05-01 10:12:36
Current User's Login: rahulreddyallu
"""

import numpy as np
import pandas as pd


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
    
    def calculate_macd(self, fast=12, slow=26, signal=9):
        """
        Calculate Moving Average Convergence Divergence
        
        Args:
            fast: Fast EMA period (default: 12)
            slow: Slow EMA period (default: 26)
            signal: Signal line period (default: 9)
            
        Returns:
            Dictionary with MACD line, signal line, and histogram
        """
        # Calculate MACD components
        fast_ema = self.df['Close'].ewm(span=fast, adjust=False).mean()
        slow_ema = self.df['Close'].ewm(span=slow, adjust=False).mean()
        
        # MACD Line
        self.df['macd_line'] = fast_ema - slow_ema
        
        # Signal Line
        self.df['macd_signal'] = self.df['macd_line'].ewm(span=signal, adjust=False).mean()
        
        # MACD Histogram
        self.df['macd_histogram'] = self.df['macd_line'] - self.df['macd_signal']
        
        # MACD crossovers (bullish/bearish signals)
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
        
        # MACD divergence (price makes new high but MACD doesn't)
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
    
    def calculate_bollinger_bands(self, period=20, std_dev=2):
        """
        Calculate Bollinger Bands
        
        Args:
            period: Period for moving average calculation (default: 20)
            std_dev: Number of standard deviations (default: 2)
            
        Returns:
            Dictionary with upper band, middle band (SMA), and lower band
        """
        # Middle band is SMA
        self.df['bb_middle'] = self.df['Close'].rolling(window=period).mean()
        
        # Calculate standard deviation
        rolling_std = self.df['Close'].rolling(window=period).std()
        
        # Calculate upper and lower bands
        self.df['bb_upper'] = self.df['bb_middle'] + (rolling_std * std_dev)
        self.df['bb_lower'] = self.df['bb_middle'] - (rolling_std * std_dev)
        
        # Calculate % B (position within bands)
        self.df['bb_pct_b'] = (self.df['Close'] - self.df['bb_lower']) / (self.df['bb_upper'] - self.df['bb_lower'])
        
        # Calculate bandwidth (indicator of volatility)
        self.df['bb_bandwidth'] = (self.df['bb_upper'] - self.df['bb_lower']) / self.df['bb_middle']
        
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
    
    def calculate_stochastic(self, k_period=14, d_period=3, slowing=3):
        """
        Calculate Stochastic Oscillator
        
        Args:
            k_period: Period for %K line (default: 14)
            d_period: Period for %D line (default: 3)
            slowing: Slowing period (default: 3)
            
        Returns:
            Dictionary with %K and %D values
        """
        # Calculate %K
        lowest_low = self.df['Low'].rolling(window=k_period).min()
        highest_high = self.df['High'].rolling(window=k_period).max()
        
        # Fast %K
        stoch_k_fast = 100 * ((self.df['Close'] - lowest_low) / (highest_high - lowest_low).replace(0, np.finfo(float).eps))
        
        # Slow %K (with slowing)
        self.df['stoch_k'] = stoch_k_fast.rolling(window=slowing).mean()
        
        # %D is the SMA of %K
        self.df['stoch_d'] = self.df['stoch_k'].rolling(window=d_period).mean()
        
        # Identify overbought/oversold conditions
        self.df['stoch_overbought'] = self.df['stoch_k'] > 80
        self.df['stoch_oversold'] = self.df['stoch_k'] < 20
        
        # Identify crossovers
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
        
        # Calculate Stochastic divergence
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
    
    def calculate_stochastic_rsi(self, rsi_period=14, stoch_period=14, k_period=3, d_period=3):
        """
        Calculate Stochastic RSI
        
        Args:
            rsi_period: Period for RSI calculation (default: 14)
            stoch_period: Period for Stochastic calculation (default: 14)
            k_period: Smoothing for %K (default: 3)
            d_period: Period for %D (default: 3)
            
        Returns:
            Dictionary with StochRSI K and D values
        """
        # Ensure RSI is calculated
        if 'rsi' not in self.df.columns:
            self.calculate_rsi(period=rsi_period)
        
        # Calculate Stochastic RSI
        min_rsi = self.df['rsi'].rolling(window=stoch_period).min()
        max_rsi = self.df['rsi'].rolling(window=stoch_period).max()
        
        # Calculate the raw Stochastic RSI
        stoch_rsi = 100 * ((self.df['rsi'] - min_rsi) / (max_rsi - min_rsi).replace(0, np.finfo(float).eps))
        
        # Smooth with moving averages
        self.df['stoch_rsi_k'] = stoch_rsi.rolling(window=k_period).mean()
        self.df['stoch_rsi_d'] = self.df['stoch_rsi_k'].rolling(window=d_period).mean()
        
        # Identify overbought/oversold conditions
        self.df['stoch_rsi_overbought'] = self.df['stoch_rsi_k'] > 80
        self.df['stoch_rsi_oversold'] = self.df['stoch_rsi_k'] < 20
        
        # Identify crossovers
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
    
    def calculate_adx(self, period=14):
        """
        Calculate Average Directional Index
        
        Args:
            period: Period for ADX calculation (default: 14)
            
        Returns:
            Dictionary with ADX, +DI and -DI values
        """
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
        
        # Calculate DX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.finfo(float).eps)
        
        # Calculate ADX
        self.df['adx'] = pd.Series(dx).rolling(window=period).mean()
        self.df['plus_di'] = plus_di
        self.df['minus_di'] = minus_di
        
        # Identify strong trend
        self.df['adx_strong_trend'] = self.df['adx'] > 25
        self.df['adx_weak_trend'] = self.df['adx'] < 20
        
        # Identify positive/negative crossover
        self.df['adx_bullish_crossover'] = (
            (self.df['plus_di'] > self.df['minus_di']) & 
            (self.df['plus_di'].shift(1) <= self.df['minus_di'].shift(1)) &
            (self.df['adx'] > 25)
        )
        
        self.df['adx_bearish_crossover'] = (
            (self.df['plus_di'] < self.df['minus_di']) & 
            (self.df['plus_di'].shift(1) >= self.df['minus_di'].shift(1)) &
            (self.df['adx'] > 25)
        )
        
        # Strong ADX buy/sell signals
        self.df['adx_strong_buy'] = (
            self.df['adx'] > 25 &
            self.df['plus_di'] > self.df['minus_di'] &
            self.df['adx'] > self.df['adx'].shift(1)
        )
        
        self.df['adx_strong_sell'] = (
            self.df['adx'] > 25 &
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
    
    def calculate_aroon(self, period=25):
        """
        Calculate Aroon Indicator
        
        Args:
            period: Period for Aroon calculation (default: 25)
            
        Returns:
            Dictionary with Aroon Up, Aroon Down, and Aroon Oscillator
        """
        # Initialize arrays for Aroon values
        aroon_up = np.zeros(len(self.df))
        aroon_down = np.zeros(len(self.df))
        
        # Calculate Aroon for each point after the initial period
        for i in range(period, len(self.df)):
            # Get current window
            high_window = self.df['High'].iloc[i-period+1:i+1]
            low_window = self.df['Low'].iloc[i-period+1:i+1]
            
            # Find indexes of highest high and lowest low in the window
            high_idx = high_window.idxmax()
            low_idx = low_window.idxmin()
            
            # Convert to relative position in the window (0 to period-1)
            days_since_high = period - 1 - list(high_window.index).index(high_idx)
            days_since_low = period - 1 - list(low_window.index).index(low_idx)
            
            # Calculate Aroon values
            aroon_up[i] = 100 * (period - days_since_high) / period
            aroon_down[i] = 100 * (period - days_since_low) / period
        
        # Add to DataFrame
        self.df['aroon_up'] = aroon_up
        self.df['aroon_down'] = aroon_down
        self.df['aroon_oscillator'] = self.df['aroon_up'] - self.df['aroon_down']
        
        # Generate signals
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
    
    def calculate_atr(self, period=14):
        """
        Calculate Average True Range
        
        Args:
            period: Period for ATR calculation (default: 14)
            
        Returns:
            Series with ATR values
        """
        # Calculate True Range
        high_low = self.df['High'] - self.df['Low']
        high_close = abs(self.df['High'] - self.df['Close'].shift(1))
        low_close = abs(self.df['Low'] - self.df['Close'].shift(1))
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        self.df['atr'] = tr.rolling(window=period).mean()
        
        # Calculate ATR percentage (ATR relative to price)
        self.df['atr_pct'] = self.df['atr'] / self.df['Close'] * 100
        
        # Calculate potential stop loss levels based on ATR
        self.df['atr_stop_long'] = self.df['Close'] - (self.df['atr'] * 2)
        self.df['atr_stop_short'] = self.df['Close'] + (self.df['atr'] * 2)
        
        # Calculate dynamic price targets based on ATR
        self.df['atr_target_long'] = self.df['Close'] + (self.df['atr'] * 3)
        self.df['atr_target_short'] = self.df['Close'] - (self.df['atr'] * 3)
        
        # Calculate reward:risk ratios based on ATR
        self.df['atr_rrr_long'] = (self.df['atr_target_long'] - self.df['Close']) / (self.df['Close'] - self.df['atr_stop_long'])
        self.df['atr_rrr_short'] = (self.df['Close'] - self.df['atr_target_short']) / (self.df['atr_stop_short'] - self.df['Close'])
        
        return self.df['atr']
    
    #
    # 13. ATR BANDS
    #
    
    def calculate_atr_bands(self, period=14, multiplier=2):
        """
        Calculate ATR Bands
        
        Args:
            period: Period for ATR calculation (default: 14)
            multiplier: Multiplier for bands (default: 2)
            
        Returns:
            Dictionary with upper and lower ATR bands
        """
        # Ensure ATR is calculated
        if 'atr' not in self.df.columns:
            self.calculate_atr(period)
        
        # Calculate moving average
        self.df['atr_ma'] = self.df['Close'].rolling(window=period).mean()
        
        # Calculate ATR bands
        self.df['atr_upper'] = self.df['atr_ma'] + (self.df['atr'] * multiplier)
        self.df['atr_lower'] = self.df['atr_ma'] - (self.df['atr'] * multiplier)
        
        # Generate signals
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
    
    def calculate_supertrend(self, period=10, multiplier=3):
        """
        Calculate SuperTrend indicator
        
        Args:
            period: Period for ATR calculation (default: 10)
            multiplier: Multiplier for bands (default: 3)
            
        Returns:
            Series with SuperTrend values
        """
        # Ensure ATR is calculated
        if 'atr' not in self.df.columns:
            self.calculate_atr(period)
        
        # Calculate basic upper and lower bands
        self.df['basic_upper'] = ((self.df['High'] + self.df['Low']) / 2) + (multiplier * self.df['atr'])
        self.df['basic_lower'] = ((self.df['High'] + self.df['Low']) / 2) - (multiplier * self.df['atr'])
        
        # Initialize SuperTrend columns
        self.df['supertrend'] = 0.0
        self.df['supertrend_direction'] = True  # True for uptrend, False for downtrend
        
        # Calculate SuperTrend iteratively
        for i in range(period, len(self.df)):
            curr = self.df.iloc[i]
            prev = self.df.iloc[i-1]
            
            # Calculate upper and lower band
            curr_upper = curr['basic_upper']
            curr_lower = curr['basic_lower']
            
            # Update bands based on previous values
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
        
        # Create buy/sell signals
        self.df['supertrend_buy'] = (
            (self.df['supertrend_direction'] == True) & 
            (self.df['supertrend_direction'].shift(1) == False)
        )
        
        self.df['supertrend_sell'] = (
            (self.df['supertrend_direction'] == False) & 
            (self.df['supertrend_direction'].shift(1) == True)
        )
        
        return self.df['supertrend']
    
    #
    # 15. ALLIGATOR INDICATOR
    #
    
    def calculate_alligator(self, jaw=13, teeth=8, lips=5):
        """
        Calculate Alligator indicator
        
        Args:
            jaw: Period for Alligator's jaw (default: 13)
            teeth: Period for Alligator's teeth (default: 8)
            lips: Period for Alligator's lips (default: 5)
            
        Returns:
            Dictionary with jaw, teeth, and lips lines
        """
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
        
        # Determine buy/sell signal
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
        self.df['cpr_width_pct'] = self.df['cpr_width'] / self.df['pivot'] * 100
        
        # Price position relative to CPR
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
    
    def detect_double_patterns(self, tolerance=0.03, lookback=50):
        """
        Detect double top and double bottom patterns
        
        Args:
            tolerance: Price tolerance for pattern similarity (default: 3%)
            lookback: Number of periods to look back (default: 50)
            
        Returns:
            Dictionary with double top and double bottom flags
        """
        # Initialize result columns
        self.df['double_top'] = False
        self.df['double_bottom'] = False
        
        # Need at least 'lookback' periods of data
        if len(self.df) < lookback:
            return {'tops': [], 'bottoms': []}
        
        # Find local extrema
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
        
        # Iterate through each row starting from lookback
        for i in range(lookback, len(self.df)):
            # Get window of data
            window = self.df.iloc[i-lookback:i+1]
            
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
                            self.df.loc[self.df.index[i], 'double_top'] = True
            
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
                            self.df.loc[self.df.index[i], 'double_bottom'] = True
        
        return {
            'double_top': self.df['double_top'],
            'double_bottom': self.df['double_bottom']
        }
    
    def detect_triple_patterns(self, tolerance=0.03, lookback=100):
        """
        Detect triple top and triple bottom patterns
        
        Args:
            tolerance: Price tolerance for pattern similarity (default: 3%)
            lookback: Number of periods to look back (default: 100)
            
        Returns:
            Dictionary with triple top and triple bottom flags
        """
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
        
        # Iterate through each row starting from lookback
        for i in range(lookback, len(self.df)):
            # Get window of data
            window = self.df.iloc[i-lookback:i+1]
            
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
                            valley2 < second_max * 0.95 and valley2 < third_max *
