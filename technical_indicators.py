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
                            valley2 < second_max * 0.95 and valley2 < third_max * 0.95):
                            # Triple top confirmed
                            self.df.loc[self.df.index[i], 'triple_top'] = True
            
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
                            self.df.loc[self.df.index[i], 'triple_bottom'] = True
        
        return {
            'triple_top': self.df['triple_top'],
            'triple_bottom': self.df['triple_bottom']
        }
    
    def detect_trading_range(self, lookback=60, tolerance=0.05, min_touches=4):
        """
        Detect if price is trading in a range (sideways market)
        
        Args:
            lookback: Number of periods to look back (default: 60)
            tolerance: Price tolerance for range width (default: 5%)
            min_touches: Minimum number of touches to top/bottom of range (default: 4)
            
        Returns:
            Dictionary with trading range information
        """
        # Initialize result columns
        self.df['in_range'] = False
        
        # Need at least 'lookback' periods of data
        if len(self.df) < lookback:
            return {'in_range': False}
        
        # Iterate through each row starting from lookback
        for i in range(lookback, len(self.df)):
            # Get window of data
            window = self.df.iloc[i-lookback:i+1]
            
            # Find highest high and lowest low in the window
            highest_high = window['High'].max()
            lowest_low = window['Low'].min()
            
            # Calculate range width
            range_width = highest_high - lowest_low
            range_midpoint = (highest_high + lowest_low) / 2
            
            # Check if range is within tolerance (not too wide)
            if range_width / range_midpoint <= tolerance:
                # Count touches to the top and bottom of the range
                upper_touches = sum((window['High'] >= highest_high * 0.98) & (window['High'] <= highest_high * 1.02))
                lower_touches = sum((window['Low'] >= lowest_low * 0.98) & (window['Low'] <= lowest_low * 1.02))
                
                # Check if there are enough touches to confirm the range
                if upper_touches + lower_touches >= min_touches:
                    self.df.loc[self.df.index[i], 'in_range'] = True
                    self.df.loc[self.df.index[i], 'range_top'] = highest_high
                    self.df.loc[self.df.index[i], 'range_bottom'] = lowest_low
                    self.df.loc[self.df.index[i], 'range_midpoint'] = range_midpoint
        
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
    
    def detect_flag_patterns(self, lookback=30, pole_threshold=0.15, flag_threshold=0.05, max_flag_bars=15):
        """
        Detect bull and bear flag patterns
        
        Args:
            lookback: Number of periods to look back (default: 30)
            pole_threshold: Minimum size of the pole as % of price (default: 15%)
            flag_threshold: Maximum size of the flag as % of price (default: 5%)
            max_flag_bars: Maximum length of the flag consolidation (default: 15)
            
        Returns:
            Dictionary with bull and bear flag indicators
        """
        # Initialize result columns
        self.df['bull_flag'] = False
        self.df['bear_flag'] = False
        
        # Need at least 'lookback' periods of data
        if len(self.df) < lookback:
            return {'bull_flag': False, 'bear_flag': False}
        
        # Iterate through each row starting from lookback
        for i in range(lookback, len(self.df)):
            # Ensure we have enough bars to check for a flag
            if i < max_flag_bars + 5:
                continue
                
            # Get window of data for the pole and potential flag
            potential_flag = self.df.iloc[i-max_flag_bars:i+1]
            potential_pole = self.df.iloc[i-max_flag_bars-5:i-max_flag_bars]
            
            # Bull flag: Strong upward move followed by sideways/slight downward consolidation
            if len(potential_pole) > 0:
                # Calculate pole height (sharp upward move)
                pole_low = potential_pole['Low'].min()
                pole_high = potential_pole['High'].max()
                pole_height = pole_high - pole_low
                
                # Calculate flag height (consolidation)
                flag_high = potential_flag['High'].max()
                flag_low = potential_flag['Low'].min()
                flag_height = flag_high - flag_low
                
                # Check for bull flag pattern
                if (pole_height / pole_low >= pole_threshold and  # Significant pole
                    flag_height / flag_low <= flag_threshold and  # Small consolidation
                    flag_high <= pole_high and  # Flag within or below pole high
                    flag_low >= pole_low):  # Flag above pole low
                    self.df.loc[self.df.index[i], 'bull_flag'] = True
                
                # Check for bear flag pattern (downward pole followed by consolidation)
                if (pole_height / pole_high >= pole_threshold and  # Significant downward pole
                    flag_height / flag_high <= flag_threshold and  # Small consolidation
                    flag_low >= pole_low and  # Flag within or above pole low
                    flag_high <= pole_high):  # Flag below pole high
                    self.df.loc[self.df.index[i], 'bear_flag'] = True
        
        return {
            'bull_flag': self.df['bull_flag'],
            'bear_flag': self.df['bear_flag']
        }
    
    #
    # 18. REWARD TO RISK RATIO
    # 
    
    def calculate_reward_risk_ratio(self, target_multiplier=1.5, stop_multiplier=1.0):
        """
        Calculate reward to risk ratio based on ATR for dynamic targets and stops
        
        Args:
            target_multiplier: ATR multiplier for target calculation (default: 1.5)
            stop_multiplier: ATR multiplier for stop calculation (default: 1.0)
            
        Returns:
            Dictionary with RRR values for long and short trades
        """
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
        
        # Calculate reward:risk ratios
        self.df['long_rrr'] = (self.df['long_target'] - self.df['long_entry']) / (self.df['long_entry'] - self.df['long_stop'])
        self.df['short_rrr'] = (self.df['short_entry'] - self.df['short_target']) / (self.df['short_stop'] - self.df['short_entry'])
        
        # Check if RRR meets minimum threshold
        min_rrr = 1.5  # Minimum RRR threshold
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
        
        # Support and Resistance signals
        if 'nearest_support' in self.df.columns and 'nearest_resistance' in self.df.columns:
            if not pd.isna(latest['nearest_support']) and not pd.isna(latest['nearest_resistance']):
                support_distance = latest.get('support_distance_pct', 0)
                resistance_distance = latest.get('resistance_distance_pct', 0)
                
                signals['support_resistance'] = {
                    'signal': 'BUY' if support_distance < 2.0 else 
                             'SELL' if resistance_distance < 2.0 else 'NEUTRAL',
                    'strength': 4 if support_distance < 1.0 or resistance_distance < 1.0 else 
                               3 if support_distance < 2.0 or resistance_distance < 2.0 else 1,
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
        if 'volume_ratio_20' in self.df.columns:
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
            range_position = (latest['Close'] - latest.get('range_bottom', 0)) / range_width if range_width > 0 else 0.5
            
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
                    sr_aligned = (latest['Low'] - support_price) / support_price < 0.02  # Within 2%
            elif pattern_signal['overall_signal'] == 'SELL' and 'nearest_resistance' in self.df.columns:
                resistance_price = latest.get('nearest_resistance', float('inf'))
                if not pd.isna(resistance_price):
                    # For sell signal, high should be near resistance
                    sr_aligned = (resistance_price - latest['High']) / latest['High'] < 0.02  # Within 2%
        
        # 3. Volume confirmation
        volume_confirms = latest.get('high_volume', False)
        
        # 4. Indicator confirmation (from validate_pattern_signal)
        indicator_validation = self.validate_pattern_signal(pattern_signal)
        indicators_confirm = indicator_validation['confirmation']
        
        # 5. Reward to Risk Ratio check
        rrr_confirms = False
        min_rrr = 1.5  # Minimum RRR threshold
        
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
            if passed_count >= 4:  # At least 4 out of 5 items passed
                recommendation = pattern_signal['overall_signal']
                confidence = 'HIGH'
                position_size = 'FULL'
            elif passed_count == 3:  # 3 items passed
                recommendation = pattern_signal['overall_signal']
                confidence = 'MEDIUM'
                position_size = 'REDUCED'
            elif passed_count == 2 and (sr_aligned or rrr_confirms):  # 2 items with either S&R or RRR
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
