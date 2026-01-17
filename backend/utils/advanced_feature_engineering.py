#!/usr/bin/env python3
"""
Advanced Feature Engineering for Quantitative Trading
======================================================

Generates powerful trading signals through:
- Technical indicator combinations
- Multi-timeframe analysis
- Market microstructure features
- Sentiment-price relationship features
- Volatility-adjusted signals
- Momentum and mean reversion indicators
- Feature interaction detection
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import ta  # Technical Analysis library

logger = logging.getLogger(__name__)


@dataclass
class FeatureSet:
    """Organized feature set for ML models"""
    name: str
    features: Dict[str, np.ndarray]
    feature_names: List[str]
    timestamps: np.ndarray
    target: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None


class AdvancedFeatureEngineer:
    """
    Advanced feature engineering for quantitative trading
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_history = {}
        logger.info("Advanced Feature Engineer initialized")
    
    # ==================== MOMENTUM FEATURES ====================
    
    @staticmethod
    def create_momentum_features(df: pd.DataFrame, periods: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """Create momentum indicators"""
        features = pd.DataFrame(index=df.index)
        
        for period in periods:
            # Rate of Change
            roc = df['close'].pct_change(period)
            features[f'roc_{period}'] = roc
            
            # Momentum
            momentum = df['close'].diff(period)
            features[f'momentum_{period}'] = momentum
            
            # RSI (Relative Strength Index)
            rsi = ta.momentum.rsi(df['close'], window=period)
            features[f'rsi_{period}'] = rsi
            
            # Stochastic RSI
            stoch_rsi = ta.momentum.stochrsi(df['close'], window=period)
            features[f'stoch_rsi_{period}'] = stoch_rsi
            
            # MACD
            macd = ta.trend.macd(df['close'], window_slow=period*2, window_fast=period)
            features[f'macd_{period}'] = macd
            
            # Awesome Oscillator
            ao = ta.momentum.awesome_oscillator(df['high'], df['low'], window1=period, window2=period*2)
            features[f'ao_{period}'] = ao
        
        return features
    
    @staticmethod
    def create_trend_features(df: pd.DataFrame, periods: List[int] = [5, 10, 20, 50]) -> pd.DataFrame:
        """Create trend indicators"""
        features = pd.DataFrame(index=df.index)
        
        for period in periods:
            # Moving averages
            sma = df['close'].rolling(window=period).mean()
            ema = df['close'].ewm(span=period, adjust=False).mean()
            
            features[f'sma_{period}'] = sma
            features[f'ema_{period}'] = ema
            
            # Price relative to MA
            features[f'price_to_sma_{period}'] = df['close'] / sma - 1
            features[f'price_to_ema_{period}'] = df['close'] / ema - 1
            
            # ADX (Average Directional Index)
            adx = ta.trend.adx(df['high'], df['low'], df['close'], window=period)
            features[f'adx_{period}'] = adx
            
            # AROON
            aroon_up = ta.trend.aroon_up(df['high'], window=period)
            aroon_down = ta.trend.aroon_down(df['low'], window=period)
            features[f'aroon_up_{period}'] = aroon_up
            features[f'aroon_down_{period}'] = aroon_down
            
            # Ichimoku
            ichimoku = ta.trend.ichimoku_a(df['high'], df['low'], window1=period)
            features[f'ichimoku_{period}'] = ichimoku
        
        return features
    
    @staticmethod
    def create_volatility_features(df: pd.DataFrame, periods: List[int] = [10, 20, 30]) -> pd.DataFrame:
        """Create volatility indicators"""
        features = pd.DataFrame(index=df.index)
        
        for period in periods:
            # Bollinger Bands
            bb = ta.volatility.bollinger_bands(df['close'], window=period)
            features[f'bb_high_{period}'] = bb[0]
            features[f'bb_mid_{period}'] = bb[1]
            features[f'bb_low_{period}'] = bb[2]
            
            # Bollinger Band Width
            bb_width = (bb[0] - bb[2]) / bb[1]
            features[f'bb_width_{period}'] = bb_width
            
            # ATR (Average True Range)
            atr = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=period)
            features[f'atr_{period}'] = atr
            atr_pct = atr / df['close']
            features[f'atr_pct_{period}'] = atr_pct
            
            # Keltner Channels
            kc = ta.volatility.keltner_channel(df['high'], df['low'], df['close'], window=period)
            features[f'kc_high_{period}'] = kc[0]
            features[f'kc_mid_{period}'] = kc[1]
            features[f'kc_low_{period}'] = kc[2]
            
            # Historical Volatility
            log_returns = np.log(df['close'] / df['close'].shift(1))
            hvol = log_returns.rolling(window=period).std()
            features[f'hvol_{period}'] = hvol
        
        return features
    
    @staticmethod
    def create_volume_features(df: pd.DataFrame, periods: List[int] = [10, 20]) -> pd.DataFrame:
        """Create volume-based indicators"""
        features = pd.DataFrame(index=df.index)
        
        for period in periods:
            # Volume averages
            vol_sma = df['volume'].rolling(window=period).mean()
            features[f'vol_sma_{period}'] = vol_sma
            features[f'vol_ratio_{period}'] = df['volume'] / vol_sma
            
            # OBV (On-Balance Volume)
            obv = ta.volume.on_balance_volume(df['close'], df['volume'])
            features[f'obv_{period}'] = obv.rolling(window=period).mean()
            
            # VWAP (Volume Weighted Average Price)
            vwap = ta.volume.volume_weighted_average_price(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                volume=df['volume'],
                window=period
            )
            features[f'vwap_{period}'] = vwap
            
            # Money Flow Index
            mfi = ta.volume.money_flow_index(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                volume=df['volume'],
                window=period
            )
            features[f'mfi_{period}'] = mfi
            
            # Accumulation/Distribution
            ad = ta.volume.acc_dist_index(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                volume=df['volume']
            )
            features[f'ad_{period}'] = ad.rolling(window=period).mean()
        
        return features
    
    # ==================== MICROSTRUCTURE FEATURES ====================
    
    @staticmethod
    def create_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create market microstructure features"""
        features = pd.DataFrame(index=df.index)
        
        # Bid-ask spread proxy (if available)
        if 'bid' in df.columns and 'ask' in df.columns:
            features['spread'] = df['ask'] - df['bid']
            features['spread_pct'] = (df['ask'] - df['bid']) / ((df['ask'] + df['bid']) / 2)
        
        # High-Low range
        features['hl_range'] = df['high'] - df['low']
        features['hl_range_pct'] = (df['high'] - df['low']) / df['close']
        
        # Open-Close range
        features['oc_range'] = abs(df['close'] - df['open'])
        features['oc_range_pct'] = abs(df['close'] - df['open']) / df['open']
        
        # Close position in range
        features['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Gap features
        features['gap'] = df['open'] - df['close'].shift(1)
        features['gap_pct'] = features['gap'] / df['close'].shift(1)
        
        # Price acceleration
        features['price_accel'] = df['close'].diff().diff()
        
        # Volatility of returns
        log_returns = np.log(df['close'] / df['close'].shift(1))
        features['return_std_5'] = log_returns.rolling(window=5).std()
        features['return_std_10'] = log_returns.rolling(window=10).std()
        
        return features
    
    # ==================== SENTIMENT-PRICE FEATURES ====================
    
    @staticmethod
    def create_sentiment_price_features(df: pd.DataFrame, sentiment_series: pd.Series) -> pd.DataFrame:
        """Create features combining sentiment and price"""
        features = pd.DataFrame(index=df.index)
        
        # Sentiment alignment with price direction
        price_direction = np.sign(df['close'].diff())
        features['sentiment_alignment'] = sentiment_series * price_direction
        
        # Sentiment strength vs price move
        price_move = df['close'].pct_change()
        features['sentiment_strength_vs_move'] = abs(sentiment_series) - abs(price_move)
        
        # Rolling correlation
        for window in [5, 10, 20]:
            correlation = sentiment_series.rolling(window=window).corr(price_move)
            features[f'sentiment_price_corr_{window}'] = correlation
        
        # Sentiment divergence from price
        sentiment_ma = sentiment_series.rolling(window=10).mean()
        features['sentiment_divergence'] = sentiment_series - sentiment_ma
        
        # Overbought/oversold based on sentiment
        features['sentiment_extreme'] = (abs(sentiment_series) > 0.7).astype(int)
        
        return features
    
    # ==================== MULTI-TIMEFRAME FEATURES ====================
    
    @staticmethod
    def create_multitimeframe_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create features from multiple timeframe analysis"""
        features = pd.DataFrame(index=df.index)
        
        # Daily trend on different timeframes
        for period in [5, 20, 60]:
            close_change = df['close'].diff(period)
            features[f'trend_{period}d'] = np.sign(close_change)
            features[f'trend_strength_{period}d'] = abs(close_change) / df['close']
        
        # Alignment of trends
        trend_5 = np.sign(df['close'].diff(5))
        trend_20 = np.sign(df['close'].diff(20))
        trend_60 = np.sign(df['close'].diff(60))
        
        features['timeframe_alignment'] = trend_5 * trend_20 * trend_60
        
        return features
    
    # ==================== COMPOSITE SIGNALS ====================
    
    @staticmethod
    def create_composite_signals(df: pd.DataFrame, all_features: pd.DataFrame) -> pd.DataFrame:
        """Create composite trading signals"""
        signals = pd.DataFrame(index=df.index)
        
        # Mean Reversion Signal
        for period in [20, 50]:
            sma = df['close'].rolling(window=period).mean()
            dev = df['close'].std(period)
            z_score = (df['close'] - sma) / (dev + 1e-10)
            signals[f'mean_reversion_{period}'] = -np.sign(z_score) * abs(z_score) / 3
        
        # Momentum-Volatility Signal
        momentum = df['close'].pct_change(10)
        volatility = df['close'].pct_change().rolling(window=10).std()
        signals['momentum_vol_signal'] = momentum / (volatility + 1e-10)
        
        # Trend Strength Signal
        atr = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=20)
        adx = ta.trend.adx(df['high'], df['low'], df['close'], window=20)
        signals['trend_signal'] = (atr / df['close']) * adx / 100
        
        # Volume Confirmation
        vol_rsi = ta.momentum.rsi(df['volume'], window=14)
        signals['volume_signal'] = (vol_rsi - 50) / 50
        
        return signals
    
    # ==================== FEATURE INTERACTIONS ====================
    
    @staticmethod
    def detect_feature_interactions(features_df: pd.DataFrame, k: int = 10) -> List[Tuple[str, str, float]]:
        """
        Detect important feature interactions using correlation
        
        Args:
            features_df: DataFrame with all features
            k: Number of top interactions to return
        
        Returns:
            List of (feature1, feature2, interaction_strength)
        """
        interactions = []
        
        feature_cols = features_df.columns
        correlations = features_df.corr()
        
        # Find strong interactions
        for i, feat1 in enumerate(feature_cols):
            for feat2 in feature_cols[i+1:]:
                corr = abs(correlations.loc[feat1, feat2])
                if 0.3 < corr < 0.95:  # Interesting interaction range
                    interactions.append((feat1, feat2, corr))
        
        # Sort by strength
        interactions.sort(key=lambda x: x[2], reverse=True)
        
        return interactions[:k]
    
    # ==================== FEATURE SELECTION ====================
    
    @staticmethod
    def select_features_by_importance(X: np.ndarray, y: np.ndarray, 
                                     feature_names: List[str], k: int = 50) -> Tuple[List[str], np.ndarray]:
        """
        Select top k features by importance using random forest
        
        Args:
            X: Feature matrix
            y: Target vector
            feature_names: Feature names
            k: Number of features to select
        
        Returns:
            Tuple of (selected_feature_names, feature_importances)
        """
        from sklearn.ensemble import RandomForestRegressor
        
        try:
            rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
            rf.fit(X, y)
            
            importances = rf.feature_importances_
            indices = np.argsort(importances)[::-1][:k]
            
            selected_features = [feature_names[i] for i in indices]
            selected_importances = importances[indices]
            
            logger.info(f"Selected {len(selected_features)} features")
            return selected_features, selected_importances
        
        except Exception as e:
            logger.error(f"Error selecting features: {e}")
            return feature_names[:k], np.ones(k) / k
    
    # ==================== DIMENSIONALITY REDUCTION ====================
    
    @staticmethod
    def apply_pca(features_df: pd.DataFrame, n_components: int = 20) -> Tuple[np.ndarray, PCA]:
        """
        Apply PCA for dimensionality reduction
        
        Args:
            features_df: DataFrame with features
            n_components: Number of principal components
        
        Returns:
            Tuple of (pca_features, pca_model)
        """
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features_df.fillna(0))
        
        pca = PCA(n_components=min(n_components, scaled_features.shape[1]))
        pca_features = pca.fit_transform(scaled_features)
        
        logger.info(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")
        
        return pca_features, pca
    
    # ==================== FEATURE ENGINEERING PIPELINE ====================
    
    def engineer_features(self, df: pd.DataFrame, sentiment_series: Optional[pd.Series] = None,
                         include_pca: bool = True, n_pca_components: int = 20) -> FeatureSet:
        """
        Complete feature engineering pipeline
        
        Args:
            df: OHLCV DataFrame
            sentiment_series: Optional sentiment data
            include_pca: Whether to apply PCA
            n_pca_components: Number of PCA components
        
        Returns:
            FeatureSet with all engineered features
        """
        logger.info("Starting feature engineering pipeline...")
        
        # Create all feature groups
        momentum_features = self.create_momentum_features(df)
        trend_features = self.create_trend_features(df)
        volatility_features = self.create_volatility_features(df)
        volume_features = self.create_volume_features(df)
        microstructure_features = self.create_microstructure_features(df)
        multitimeframe_features = self.create_multitimeframe_features(df)
        composite_signals = self.create_composite_signals(df, None)
        
        # Combine all features
        all_features = pd.concat([
            momentum_features,
            trend_features,
            volatility_features,
            volume_features,
            microstructure_features,
            multitimeframe_features,
            composite_signals
        ], axis=1)
        
        # Add sentiment features if provided
        if sentiment_series is not None:
            sentiment_features = self.create_sentiment_price_features(df, sentiment_series)
            all_features = pd.concat([all_features, sentiment_features], axis=1)
        
        # Fill NaN values
        all_features = all_features.fillna(method='bfill').fillna(0)
        
        feature_names = list(all_features.columns)
        logger.info(f"Created {len(feature_names)} features")
        
        # Apply PCA if requested
        if include_pca:
            pca_features, pca_model = self.apply_pca(all_features, n_pca_components)
            
            # Combine PCA with original features
            pca_feature_names = [f'pca_{i}' for i in range(pca_features.shape[1])]
            all_features[pca_feature_names] = pca_features
            feature_names.extend(pca_feature_names)
        
        # Normalize features
        feature_matrix = all_features[feature_names].values
        self.scaler.fit(feature_matrix)
        normalized_features = self.scaler.transform(feature_matrix)
        
        return FeatureSet(
            name="complete_feature_set",
            features={name: normalized_features[:, i] for i, name in enumerate(feature_names)},
            feature_names=feature_names,
            timestamps=df.index.values,
            metadata={
                'total_features': len(feature_names),
                'sentiment_included': sentiment_series is not None,
                'pca_applied': include_pca
            }
        )


# Singleton instance
_feature_engineer = None

def get_feature_engineer() -> AdvancedFeatureEngineer:
    """Get or create singleton feature engineer"""
    global _feature_engineer
    if _feature_engineer is None:
        _feature_engineer = AdvancedFeatureEngineer()
    return _feature_engineer


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    engineer = get_feature_engineer()
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    sample_data = pd.DataFrame({
        'open': np.random.randn(252).cumsum() + 100,
        'high': np.random.randn(252).cumsum() + 102,
        'low': np.random.randn(252).cumsum() + 98,
        'close': np.random.randn(252).cumsum() + 100,
        'volume': np.random.randint(1000000, 10000000, 252)
    }, index=dates)
    
    # Create sentiment series
    sentiment = pd.Series(np.sin(np.linspace(0, 4*np.pi, 252)) * 0.5, index=dates)
    
    # Engineer features
    feature_set = engineer.engineer_features(sample_data, sentiment)
    print(f"Created {len(feature_set.feature_names)} features")
    print(f"Features shape: {feature_set.features[feature_set.feature_names[0]].shape}")
