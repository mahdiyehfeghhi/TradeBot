from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time
import pickle
from pathlib import Path

from loguru import logger

try:
    import tensorflow as tf
    from tensorflow import keras
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    TENSORFLOW_AVAILABLE = True
except ImportError:
    logger.warning("TensorFlow not available. Install with: pip install tensorflow scikit-learn")
    TENSORFLOW_AVAILABLE = False

from domain.models import Candle, TradeDecision
from domain.ports import Strategy


@dataclass
class PredictionResult:
    symbol: str
    predicted_price: float
    confidence: float
    direction: str  # "up", "down", "sideways"
    strength: float  # 0 to 1
    timeframe: str
    timestamp: int


class TechnicalIndicators:
    """Calculate technical indicators for ML features"""
    
    @staticmethod
    def rsi(prices: np.array, period: int = 14) -> np.array:
        """Calculate RSI"""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = np.convolve(gains, np.ones(period)/period, mode='valid')
        avg_losses = np.convolve(losses, np.ones(period)/period, mode='valid')
        
        rs = avg_gains / (avg_losses + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        # Pad with NaNs to match original length
        return np.concatenate([np.full(len(prices) - len(rsi), np.nan), rsi])
    
    @staticmethod
    def sma(prices: np.array, period: int) -> np.array:
        """Simple Moving Average"""
        sma = np.convolve(prices, np.ones(period)/period, mode='valid')
        return np.concatenate([np.full(len(prices) - len(sma), np.nan), sma])
    
    @staticmethod
    def ema(prices: np.array, period: int) -> np.array:
        """Exponential Moving Average"""
        alpha = 2 / (period + 1)
        ema = np.zeros_like(prices)
        ema[0] = prices[0]
        
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    @staticmethod
    def bollinger_bands(prices: np.array, period: int = 20, std_dev: int = 2) -> Tuple[np.array, np.array, np.array]:
        """Bollinger Bands"""
        sma = TechnicalIndicators.sma(prices, period)
        std = np.array([np.std(prices[max(0, i-period+1):i+1]) for i in range(len(prices))])
        
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        
        return upper, sma, lower
    
    @staticmethod
    def macd(prices: np.array, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.array, np.array, np.array]:
        """MACD"""
        ema_fast = TechnicalIndicators.ema(prices, fast)
        ema_slow = TechnicalIndicators.ema(prices, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram


class DeepLearningPredictor:
    """
    Deep Learning model for cryptocurrency price prediction using LSTM
    """
    
    def __init__(self, symbol: str, sequence_length: int = 60, prediction_horizon: int = 1):
        self.symbol = symbol
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_scaler = MinMaxScaler()
        self.model_path = Path(f"data/models/{symbol}_lstm_model.h5")
        self.scaler_path = Path(f"data/models/{symbol}_scaler.pkl")
        self.is_trained = False
        
        # Ensure model directory exists
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not TENSORFLOW_AVAILABLE:
            logger.error("TensorFlow not available. Cannot use deep learning predictor.")
            return
        
        # Try to load existing model
        self._load_model()
    
    def _create_features(self, candles: List[Candle]) -> np.array:
        """Create feature matrix from candle data"""
        df = pd.DataFrame([{
            'open': c.open,
            'high': c.high,
            'low': c.low,
            'close': c.close,
            'volume': c.volume,
            'timestamp': c.timestamp
        } for c in candles])
        
        if len(df) < 50:
            return np.array([])
        
        # Price features
        df['price_change'] = df['close'].pct_change()
        df['high_low_ratio'] = df['high'] / df['low']
        df['volume_change'] = df['volume'].pct_change()
        
        # Technical indicators
        prices = df['close'].values
        df['rsi'] = TechnicalIndicators.rsi(prices)
        df['sma_10'] = TechnicalIndicators.sma(prices, 10)
        df['sma_20'] = TechnicalIndicators.sma(prices, 20)
        df['ema_12'] = TechnicalIndicators.ema(prices, 12)
        df['ema_26'] = TechnicalIndicators.ema(prices, 26)
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(prices)
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        
        # MACD
        macd, signal, histogram = TechnicalIndicators.macd(prices)
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_histogram'] = histogram
        
        # Time-based features
        df['hour'] = pd.to_datetime(df['timestamp'], unit='s').dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp'], unit='s').dt.dayofweek
        
        # Select feature columns
        feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'price_change', 'high_low_ratio', 'volume_change',
            'rsi', 'sma_10', 'sma_20', 'ema_12', 'ema_26',
            'bb_position', 'macd', 'macd_signal', 'macd_histogram',
            'hour', 'day_of_week'
        ]
        
        # Drop rows with NaN values
        df = df.dropna()
        
        if len(df) < self.sequence_length:
            return np.array([])
        
        return df[feature_columns].values
    
    def _prepare_sequences(self, features: np.array, targets: np.array = None) -> Tuple[np.array, np.array]:
        """Prepare sequences for LSTM training/prediction"""
        if len(features) < self.sequence_length:
            return np.array([]), np.array([])
        
        X, y = [], []
        
        for i in range(self.sequence_length, len(features)):
            X.append(features[i-self.sequence_length:i])
            if targets is not None:
                y.append(targets[i])
        
        return np.array(X), np.array(y)
    
    def _build_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """Build LSTM model architecture"""
        model = keras.Sequential([
            # First LSTM layer with dropout
            keras.layers.LSTM(100, return_sequences=True, input_shape=input_shape),
            keras.layers.Dropout(0.2),
            
            # Second LSTM layer
            keras.layers.LSTM(100, return_sequences=True),
            keras.layers.Dropout(0.2),
            
            # Third LSTM layer
            keras.layers.LSTM(50, return_sequences=False),
            keras.layers.Dropout(0.2),
            
            # Dense layers
            keras.layers.Dense(25, activation='relu'),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(1)
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, candles: List[Candle], epochs: int = 50, validation_split: float = 0.2) -> Dict:
        """Train the LSTM model"""
        if not TENSORFLOW_AVAILABLE:
            return {"error": "TensorFlow not available"}
        
        logger.info(f"Training LSTM model for {self.symbol} with {len(candles)} candles")
        
        # Create features
        features = self._create_features(candles)
        if len(features) == 0:
            return {"error": "Insufficient data for training"}
        
        # Scale features
        features_scaled = self.feature_scaler.fit_transform(features)
        
        # Prepare targets (next period close price)
        close_prices = features[:, 3]  # Close price is the 4th column
        targets = close_prices[self.prediction_horizon:]
        features_scaled = features_scaled[:-self.prediction_horizon]
        
        # Scale targets
        targets_scaled = self.scaler.fit_transform(targets.reshape(-1, 1)).flatten()
        
        # Create sequences
        X, y = self._prepare_sequences(features_scaled, targets_scaled)
        
        if len(X) == 0:
            return {"error": "Insufficient data for sequence creation"}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=validation_split, shuffle=False
        )
        
        # Build model
        self.model = self._build_model((X.shape[1], X.shape[2]))
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        # Save model and scaler
        self._save_model()
        
        # Evaluate model
        train_loss = self.model.evaluate(X_train, y_train, verbose=0)
        test_loss = self.model.evaluate(X_test, y_test, verbose=0)
        
        self.is_trained = True
        
        logger.info(f"Model training completed. Train loss: {train_loss[0]:.6f}, Test loss: {test_loss[0]:.6f}")
        
        return {
            "success": True,
            "train_loss": train_loss[0],
            "test_loss": test_loss[0],
            "epochs_trained": len(history.history['loss']),
            "features_used": features_scaled.shape[1],
            "sequences_created": len(X)
        }
    
    def predict(self, candles: List[Candle]) -> PredictionResult:
        """Make price prediction"""
        if not TENSORFLOW_AVAILABLE or self.model is None:
            return PredictionResult(
                symbol=self.symbol,
                predicted_price=0.0,
                confidence=0.0,
                direction="sideways",
                strength=0.0,
                timeframe="1h",
                timestamp=int(time.time())
            )
        
        # Create features
        features = self._create_features(candles)
        if len(features) == 0:
            return self._create_default_prediction()
        
        # Scale features
        features_scaled = self.feature_scaler.transform(features)
        
        # Get last sequence
        if len(features_scaled) < self.sequence_length:
            return self._create_default_prediction()
        
        last_sequence = features_scaled[-self.sequence_length:].reshape(1, self.sequence_length, -1)
        
        # Make prediction
        prediction_scaled = self.model.predict(last_sequence, verbose=0)
        prediction = self.scaler.inverse_transform(prediction_scaled.reshape(-1, 1))[0][0]
        
        # Calculate confidence and direction
        current_price = candles[-1].close
        price_change_pct = (prediction - current_price) / current_price
        
        # Confidence based on model certainty and historical performance
        base_confidence = 0.7  # Base confidence for trained model
        volatility = np.std([c.close for c in candles[-20:]])
        volatility_factor = min(volatility / current_price, 0.3)  # Reduce confidence in high volatility
        confidence = max(0.1, base_confidence - volatility_factor)
        
        # Determine direction and strength
        if abs(price_change_pct) < 0.01:  # Less than 1% change
            direction = "sideways"
            strength = 0.0
        elif price_change_pct > 0:
            direction = "up"
            strength = min(abs(price_change_pct) * 10, 1.0)  # Scale to 0-1
        else:
            direction = "down"
            strength = min(abs(price_change_pct) * 10, 1.0)
        
        return PredictionResult(
            symbol=self.symbol,
            predicted_price=prediction,
            confidence=confidence,
            direction=direction,
            strength=strength,
            timeframe="1h",
            timestamp=int(time.time())
        )
    
    def _create_default_prediction(self) -> PredictionResult:
        """Create default prediction when model is not available"""
        return PredictionResult(
            symbol=self.symbol,
            predicted_price=0.0,
            confidence=0.0,
            direction="sideways",
            strength=0.0,
            timeframe="1h",
            timestamp=int(time.time())
        )
    
    def _save_model(self):
        """Save model and scaler"""
        if self.model is not None:
            self.model.save(self.model_path)
            
        with open(self.scaler_path, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'feature_scaler': self.feature_scaler
            }, f)
    
    def _load_model(self):
        """Load existing model and scaler"""
        try:
            if self.model_path.exists() and self.scaler_path.exists() and TENSORFLOW_AVAILABLE:
                self.model = keras.models.load_model(self.model_path)
                
                with open(self.scaler_path, 'rb') as f:
                    scalers = pickle.load(f)
                    self.scaler = scalers['scaler']
                    self.feature_scaler = scalers['feature_scaler']
                
                self.is_trained = True
                logger.info(f"Loaded existing LSTM model for {self.symbol}")
        except Exception as e:
            logger.warning(f"Could not load existing model for {self.symbol}: {e}")


class DeepLearningStrategy(Strategy):
    """
    Trading strategy that uses deep learning predictions combined with technical analysis
    """
    
    def __init__(self, symbol: str, confidence_threshold: float = 0.6, 
                 strength_threshold: float = 0.3, auto_train: bool = True):
        self.symbol = symbol
        self.confidence_threshold = confidence_threshold
        self.strength_threshold = strength_threshold
        self.auto_train = auto_train
        self.predictor = DeepLearningPredictor(symbol)
        self.training_data: List[Candle] = []
        self.last_training = 0
        self.training_interval = 86400 * 7  # Retrain weekly
        
    def on_candles(self, candles: List[Candle], price: float) -> TradeDecision:
        candle_list = list(candles)
        
        # Store training data
        self.training_data.extend(candle_list)
        if len(self.training_data) > 1000:  # Keep only recent data
            self.training_data = self.training_data[-1000:]
        
        # Auto-training logic
        current_time = int(time.time())
        if (self.auto_train and 
            current_time - self.last_training > self.training_interval and
            len(self.training_data) > 200):
            
            self._retrain_model()
            self.last_training = current_time
        
        # Make prediction
        if len(candle_list) < 60:
            return TradeDecision(action="hold", reason="insufficient-data", size_quote=0)
        
        prediction = self.predictor.predict(candle_list)
        
        # Generate trading decision based on prediction
        if (prediction.confidence < self.confidence_threshold or 
            prediction.strength < self.strength_threshold):
            return TradeDecision(
                action="hold", 
                reason=f"low-confidence: conf={prediction.confidence:.2f}, strength={prediction.strength:.2f}",
                size_quote=0
            )
        
        # Calculate position size based on prediction strength
        base_size = prediction.strength * prediction.confidence
        
        if prediction.direction == "up":
            return TradeDecision(
                action="buy",
                reason=f"DL-prediction: {prediction.direction} price={prediction.predicted_price:.2f} conf={prediction.confidence:.2f}",
                size_quote=0,
                stop_loss=price * 0.95,  # 5% stop loss
                take_profit=prediction.predicted_price
            )
        elif prediction.direction == "down":
            return TradeDecision(
                action="sell",
                reason=f"DL-prediction: {prediction.direction} price={prediction.predicted_price:.2f} conf={prediction.confidence:.2f}",
                size_quote=0,
                stop_loss=price * 1.05,  # 5% stop loss
                take_profit=prediction.predicted_price
            )
        
        return TradeDecision(
            action="hold",
            reason=f"DL-neutral: {prediction.direction} conf={prediction.confidence:.2f}",
            size_quote=0
        )
    
    def _retrain_model(self):
        """Retrain the deep learning model"""
        logger.info(f"Retraining LSTM model for {self.symbol}")
        try:
            result = self.predictor.train(self.training_data, epochs=30)
            if result.get("success"):
                logger.info(f"Model retrained successfully. Test loss: {result['test_loss']:.6f}")
            else:
                logger.error(f"Model retraining failed: {result.get('error')}")
        except Exception as e:
            logger.error(f"Error during model retraining: {e}")


class EnsembleMLStrategy(Strategy):
    """
    Ensemble strategy combining multiple ML predictions with traditional indicators
    """
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.dl_strategy = DeepLearningStrategy(symbol, auto_train=False)
        self.ml_weight = 0.6
        self.technical_weight = 0.4
        
    def on_candles(self, candles: List[Candle], price: float) -> TradeDecision:
        candle_list = list(candles)
        
        if len(candle_list) < 60:
            return TradeDecision(action="hold", reason="insufficient-data", size_quote=0)
        
        # Get ML prediction
        ml_decision = self.dl_strategy.on_candles(candles, price)
        
        # Get technical analysis signals
        technical_signals = self._get_technical_signals(candle_list, price)
        
        # Combine signals
        combined_signal = self._combine_signals(ml_decision, technical_signals)
        
        return combined_signal
    
    def _get_technical_signals(self, candles: List[Candle], price: float) -> Dict:
        """Get traditional technical analysis signals"""
        prices = np.array([c.close for c in candles])
        
        # RSI
        rsi = TechnicalIndicators.rsi(prices)[-1]
        
        # Moving averages
        sma_20 = TechnicalIndicators.sma(prices, 20)[-1]
        sma_50 = TechnicalIndicators.sma(prices, 50)[-1]
        
        # MACD
        macd, signal, histogram = TechnicalIndicators.macd(prices)
        macd_signal = "buy" if histogram[-1] > 0 else "sell" if histogram[-1] < 0 else "hold"
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(prices)
        bb_position = (price - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1])
        
        # Generate signals
        signals = {
            "rsi_signal": "buy" if rsi < 30 else "sell" if rsi > 70 else "hold",
            "ma_signal": "buy" if price > sma_20 > sma_50 else "sell" if price < sma_20 < sma_50 else "hold",
            "macd_signal": macd_signal,
            "bb_signal": "buy" if bb_position < 0.2 else "sell" if bb_position > 0.8 else "hold",
            "strength": abs(bb_position - 0.5) * 2  # 0 to 1
        }
        
        return signals
    
    def _combine_signals(self, ml_decision: TradeDecision, technical_signals: Dict) -> TradeDecision:
        """Combine ML and technical signals"""
        
        # Convert technical signals to scores
        tech_score = 0
        signal_map = {"buy": 1, "sell": -1, "hold": 0}
        
        for signal_name in ["rsi_signal", "ma_signal", "macd_signal", "bb_signal"]:
            tech_score += signal_map.get(technical_signals[signal_name], 0)
        
        tech_score = tech_score / 4  # Normalize to -1 to 1
        
        # Convert ML decision to score
        ml_score = 0
        if ml_decision.action == "buy":
            ml_score = 1
        elif ml_decision.action == "sell":
            ml_score = -1
        
        # Weighted combination
        combined_score = ml_score * self.ml_weight + tech_score * self.technical_weight
        
        # Generate final decision
        if combined_score > 0.3:
            action = "buy"
            reason = f"ensemble-buy: ML={ml_decision.action} tech_score={tech_score:.2f} combined={combined_score:.2f}"
        elif combined_score < -0.3:
            action = "sell" 
            reason = f"ensemble-sell: ML={ml_decision.action} tech_score={tech_score:.2f} combined={combined_score:.2f}"
        else:
            action = "hold"
            reason = f"ensemble-hold: ML={ml_decision.action} tech_score={tech_score:.2f} combined={combined_score:.2f}"
        
        return TradeDecision(
            action=action,
            reason=reason,
            size_quote=0,
            stop_loss=ml_decision.stop_loss,
            take_profit=ml_decision.take_profit
        )