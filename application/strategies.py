from __future__ import annotations

from typing import Iterable, List, Dict
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from loguru import logger

from domain.models import Candle, TradeDecision, MarketPerformance
from domain.ports import Strategy, MarketDataPort


class RSIMAStrategy(Strategy):
    def __init__(self, rsi_period: int = 14, rsi_buy: int = 35, rsi_sell: int = 65, ma_fast: int = 9, ma_slow: int = 21):
        self.rsi_period = rsi_period
        self.rsi_buy = rsi_buy
        self.rsi_sell = rsi_sell
        self.ma_fast = ma_fast
        self.ma_slow = ma_slow

    @staticmethod
    def _rsi(series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss.replace(0, np.nan))
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def on_candles(self, candles: Iterable[Candle], price: float) -> TradeDecision:
        df = pd.DataFrame([{
            "timestamp": c.timestamp,
            "open": c.open,
            "high": c.high,
            "low": c.low,
            "close": c.close,
            "volume": c.volume,
        } for c in candles])

        if len(df) < max(self.rsi_period, self.ma_slow) + 2:
            return TradeDecision(action="hold", reason="not-enough-data", size_quote=0)

        df["rsi"] = self._rsi(df["close"], self.rsi_period)
        df["ma_fast"] = df["close"].rolling(self.ma_fast).mean()
        df["ma_slow"] = df["close"].rolling(self.ma_slow).mean()
        last = df.iloc[-1]
        try:
            logger.debug(
                "Indicators: price={price} rsi={rsi:.2f} ma_fast={mf:.2f} ma_slow={ms:.2f}",
                price=price,
                rsi=float(last["rsi"]),
                mf=float(last["ma_fast"]),
                ms=float(last["ma_slow"]),
            )
        except Exception:
            pass

        # Define a simple stop as recent swing low/high (2 bars back) for R/R calc
        swing_low = float(df["low"].iloc[-3:-1].min())
        swing_high = float(df["high"].iloc[-3:-1].max())

        # Buy when RSI is oversold regardless of MA cross (more responsive)
        if last["rsi"] <= self.rsi_buy:
            stop = swing_low if swing_low < price else price * 0.98
            return TradeDecision(
                action="buy",
                reason=f"RSI {last['rsi']:.1f} <= {self.rsi_buy}",
                size_quote=0,
                stop_loss=stop,
            )
        # Sell when RSI is overbought
        elif last["rsi"] >= self.rsi_sell:
            stop = swing_high if swing_high > price else price * 1.02
            return TradeDecision(
                action="sell",
                reason=f"RSI {last['rsi']:.1f} >= {self.rsi_sell}",
                size_quote=0,
                stop_loss=stop,
            )
        else:
            return TradeDecision(action="hold", reason="neutral", size_quote=0)


class BreakoutStrategy(Strategy):
    """Simple Donchian breakout strategy with stop at opposite band.

    - Buy when close breaks above highest high of lookback
    - Sell when close breaks below lowest low of lookback
    """

    def __init__(self, lookback: int = 20):
        self.lookback = lookback

    def on_candles(self, candles: Iterable[Candle], price: float) -> TradeDecision:
        df = pd.DataFrame([{ "high": c.high, "low": c.low, "close": c.close } for c in candles])
        if len(df) < self.lookback + 2:
            return TradeDecision(action="hold", reason="not-enough-data", size_quote=0)
        hh = df["high"].rolling(self.lookback).max()
        ll = df["low"].rolling(self.lookback).min()
        last = df.iloc[-1]
        last_hh = float(hh.iloc[-2])  # use previous bar's band to avoid lookahead
        last_ll = float(ll.iloc[-2])
        if last["close"] > last_hh:
            return TradeDecision(action="buy", reason=f"breakout>HH({self.lookback})", size_quote=0, stop_loss=last_ll)
        if last["close"] < last_ll:
            return TradeDecision(action="sell", reason=f"breakdown<LL({self.lookback})", size_quote=0, stop_loss=last_hh)
        return TradeDecision(action="hold", reason="range", size_quote=0)


class MeanReversionStrategy(Strategy):
    """Bollinger-band mean reversion with RSI filter.

    - Buy when price < lower band and RSI < rsi_buy
    - Sell when price > upper band and RSI > rsi_sell
    """

    def __init__(self, bb_period: int = 20, bb_std: float = 2.0, rsi_period: int = 14, rsi_buy: int = 35, rsi_sell: int = 65):
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.rsi_buy = rsi_buy
        self.rsi_sell = rsi_sell

    @staticmethod
    def _rsi(series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss.replace(0, np.nan))
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def on_candles(self, candles: Iterable[Candle], price: float) -> TradeDecision:
        df = pd.DataFrame([{ "close": c.close } for c in candles])
        if len(df) < max(self.bb_period, self.rsi_period) + 5:
            return TradeDecision(action="hold", reason="not-enough-data", size_quote=0)
        ma = df["close"].rolling(self.bb_period).mean()
        std = df["close"].rolling(self.bb_period).std(ddof=0)
        upper = ma + self.bb_std * std
        lower = ma - self.bb_std * std
        rsi = self._rsi(df["close"], self.rsi_period)
        last_close = float(df["close"].iloc[-1])
        last_upper = float(upper.iloc[-1])
        last_lower = float(lower.iloc[-1])
        last_rsi = float(rsi.iloc[-1])
        if last_close < last_lower and last_rsi <= self.rsi_buy:
            stop = min(last_lower, last_close * 0.98)
            return TradeDecision(action="buy", reason=f"BB<{self.bb_std}σ & RSI {last_rsi:.1f}", size_quote=0, stop_loss=stop)
        if last_close > last_upper and last_rsi >= self.rsi_sell:
            stop = max(last_upper, last_close * 1.02)
            return TradeDecision(action="sell", reason=f"BB>{self.bb_std}σ & RSI {last_rsi:.1f}", size_quote=0, stop_loss=stop)
        return TradeDecision(action="hold", reason="inside-bands", size_quote=0)


class EnsembleStrategy(Strategy):
    """Combine multiple strategies via majority vote.

    - Translate actions: buy=+1, sell=-1, hold=0
    - If |sum| >= threshold -> act in that direction, else hold
    - Stop-loss aggregated conservatively: for buy -> min(stop_i), for sell -> max(stop_i)
    """

    def __init__(self, strategies: List[Strategy], threshold: int = 2):
        self.strategies = strategies
        self.threshold = max(1, int(threshold))

    def on_candles(self, candles: Iterable[Candle], price: float) -> TradeDecision:
        votes = 0
        reasons = []
        buy_stops = []
        sell_stops = []
        for s in self.strategies:
            d = s.on_candles(candles, price)
            if d.action == "buy":
                votes += 1
                if d.stop_loss:
                    buy_stops.append(float(d.stop_loss))
            elif d.action == "sell":
                votes -= 1
                if d.stop_loss:
                    sell_stops.append(float(d.stop_loss))
            reasons.append(f"{type(s).__name__}:{d.action}")

        reason = ", ".join(reasons)
        if votes >= self.threshold:
            stop = min(buy_stops) if buy_stops else None
            return TradeDecision(action="buy", reason=f"votes={votes} >= {self.threshold} | {reason}", size_quote=0, stop_loss=stop)
        if -votes >= self.threshold:
            stop = max(sell_stops) if sell_stops else None
            return TradeDecision(action="sell", reason=f"votes={votes} <= -{self.threshold} | {reason}", size_quote=0, stop_loss=stop)
        return TradeDecision(action="hold", reason=f"votes={votes} | {reason}", size_quote=0)


class TopPerformerStrategy(Strategy):
    """
    Strategy that trades symbols with highest 24h performance.
    
    This strategy:
    1. Identifies symbols with highest 24h price change
    2. Uses momentum indicators to time entries/exits
    3. Implements quick profit-taking for fast trading
    """
    
    def __init__(self, min_volume_24h: float = 1000000, min_price_change: float = 5.0, 
                 momentum_period: int = 10, rsi_period: int = 14):
        self.min_volume_24h = min_volume_24h
        self.min_price_change = min_price_change  # minimum % change to consider
        self.momentum_period = momentum_period
        self.rsi_period = rsi_period
        self._market_performances: List[MarketPerformance] = []
        self._last_update = 0
        
    def update_market_performances(self, performances: List[MarketPerformance]):
        """Update the list of market performances"""
        self._market_performances = performances
        self._last_update = int(time.time())
        
    def _get_symbol_performance(self, symbol: str) -> MarketPerformance | None:
        """Get performance data for a specific symbol"""
        for perf in self._market_performances:
            if perf.symbol == symbol:
                return perf
        return None
        
    def _calculate_momentum_score(self, candles: List[Candle], price: float) -> float:
        """Calculate momentum score based on recent price action"""
        if len(candles) < self.momentum_period:
            return 0.0
            
        df = pd.DataFrame([{
            "close": c.close,
            "volume": c.volume,
            "high": c.high,
            "low": c.low
        } for c in candles])
        
        # Calculate price momentum
        price_momentum = (price - df["close"].iloc[-(self.momentum_period+1)]) / df["close"].iloc[-(self.momentum_period+1)] * 100
        
        # Calculate volume momentum (recent vs average)
        recent_volume = df["volume"].tail(3).mean()
        avg_volume = df["volume"].mean()
        volume_momentum = (recent_volume - avg_volume) / avg_volume * 100
        
        # Calculate volatility (for risk assessment)
        volatility = df["close"].pct_change().std() * 100
        
        # Combined momentum score (price momentum weighted by volume, adjusted for volatility)
        momentum_score = price_momentum * (1 + min(volume_momentum / 100, 1.0)) * (1 - min(volatility / 10, 0.5))
        
        return momentum_score
        
    @staticmethod
    def _rsi(series: pd.Series, period: int) -> pd.Series:
        """Calculate RSI"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss.replace(0, np.nan))
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def on_candles(self, candles: Iterable[Candle], price: float) -> TradeDecision:
        candles_list = list(candles)
        
        if len(candles_list) < max(self.momentum_period, self.rsi_period) + 2:
            return TradeDecision(action="hold", reason="insufficient-data", size_quote=0)
            
        # Check if we have recent market performance data for this symbol
        # Note: This assumes the symbol can be extracted from context or passed separately
        # For now, we'll proceed with technical analysis
        
        df = pd.DataFrame([{
            "timestamp": c.timestamp,
            "open": c.open,
            "high": c.high,
            "low": c.low,
            "close": c.close,
            "volume": c.volume,
        } for c in candles_list])
        
        # Calculate indicators
        df["rsi"] = self._rsi(df["close"], self.rsi_period)
        df["price_change"] = df["close"].pct_change() * 100
        
        # Calculate momentum score
        momentum_score = self._calculate_momentum_score(candles_list, price)
        
        # Get recent values
        last = df.iloc[-1]
        recent_rsi = float(last["rsi"])
        recent_change = float(last["price_change"])
        
        # Calculate 24h change (approximate from available data)
        if len(df) >= 24:  # assuming hourly candles
            price_24h_ago = float(df["close"].iloc[-24])
            change_24h = (price - price_24h_ago) / price_24h_ago * 100
        else:
            # Fallback to available data
            price_start = float(df["close"].iloc[0])
            change_24h = (price - price_start) / price_start * 100
            
        # Define stops based on recent volatility
        volatility = df["close"].pct_change().tail(10).std() * price
        stop_distance = max(volatility * 2, price * 0.015)  # min 1.5% stop
        
        # Entry logic for top performers
        if change_24h >= self.min_price_change and momentum_score > 2.0:
            if recent_rsi < 70 and recent_change > 0:  # Not overbought and recent upward movement
                stop_loss = price - stop_distance
                return TradeDecision(
                    action="buy",
                    reason=f"top-performer: 24h:{change_24h:.1f}% momentum:{momentum_score:.1f} RSI:{recent_rsi:.1f}",
                    size_quote=0,
                    stop_loss=stop_loss,
                    take_profit=price + (stop_distance * 2)  # 2:1 reward:risk
                )
                
        # Exit logic for profit taking
        if change_24h >= self.min_price_change * 1.5:  # Extended move
            if recent_rsi > 75 or recent_change < -1.0:  # Overbought or reversal signal
                stop_loss = price + stop_distance  # Stop above current price for short
                return TradeDecision(
                    action="sell",
                    reason=f"profit-taking: 24h:{change_24h:.1f}% RSI:{recent_rsi:.1f} momentum:{momentum_score:.1f}",
                    size_quote=0,
                    stop_loss=stop_loss,
                    take_profit=price - (stop_distance * 1.5)
                )
                
        # Hold if conditions not met
        reason = f"hold: 24h:{change_24h:.1f}% momentum:{momentum_score:.1f} RSI:{recent_rsi:.1f}"
        return TradeDecision(action="hold", reason=reason, size_quote=0)


class AdaptiveStrategy(Strategy):
    """
    Adaptive strategy that switches between different strategies based on market conditions
    and learned performance metrics.
    """
    
    def __init__(self, strategies: List[Strategy], performance_window: int = 100):
        self.strategies = strategies
        self.performance_window = performance_window
        self.strategy_scores = {type(s).__name__: 0.0 for s in strategies}
        self._decision_history = []
        
    def update_strategy_performance(self, strategy_name: str, outcome: float):
        """Update strategy performance score based on outcome"""
        if strategy_name in self.strategy_scores:
            # Simple moving average of outcomes
            self.strategy_scores[strategy_name] = (
                self.strategy_scores[strategy_name] * 0.9 + outcome * 0.1
            )
            
    def _select_best_strategy(self) -> Strategy:
        """Select the best performing strategy"""
        if not self.strategy_scores or all(score == 0 for score in self.strategy_scores.values()):
            # If no performance data, return first strategy
            return self.strategies[0]
            
        best_strategy_name = max(self.strategy_scores.items(), key=lambda x: x[1])[0]
        
        for strategy in self.strategies:
            if type(strategy).__name__ == best_strategy_name:
                return strategy
                
        return self.strategies[0]
        
    def on_candles(self, candles: Iterable[Candle], price: float) -> TradeDecision:
        # Select best performing strategy
        selected_strategy = self._select_best_strategy()
        
        # Get decision from selected strategy
        decision = selected_strategy.on_candles(candles, price)
        
        # Add metadata about which strategy was used
        decision.reason = f"{type(selected_strategy).__name__}: {decision.reason}"
        
        # Store decision for tracking
        self._decision_history.append({
            "strategy": type(selected_strategy).__name__,
            "decision": decision,
            "timestamp": int(time.time())
        })
        
        # Keep only recent history
        if len(self._decision_history) > self.performance_window:
            self._decision_history = self._decision_history[-self.performance_window:]
            
        return decision


class GridTradingStrategy(Strategy):
    """
    Grid trading strategy for ranging markets.
    Places buy/sell orders at fixed intervals to profit from price oscillations.
    """
    
    def __init__(self, grid_size: float = 0.02, grid_levels: int = 5, 
                 range_detection_period: int = 50):
        self.grid_size = grid_size  # 2% grid spacing
        self.grid_levels = grid_levels
        self.range_detection_period = range_detection_period
        self.last_action = "hold"
        self.entry_price = None
        
    def _detect_ranging_market(self, candles: List[Candle]) -> bool:
        """Detect if market is in a ranging condition"""
        if len(candles) < self.range_detection_period:
            return False
            
        prices = [c.close for c in candles[-self.range_detection_period:]]
        price_range = (max(prices) - min(prices)) / min(prices)
        
        # Consider ranging if price movement is less than 10%
        return price_range < 0.10
    
    def on_candles(self, candles: Iterable[Candle], price: float) -> TradeDecision:
        candle_list = list(candles)
        if len(candle_list) < self.range_detection_period:
            return TradeDecision(action="hold", reason="insufficient-data", size_quote=0)
        
        if not self._detect_ranging_market(candle_list):
            return TradeDecision(action="hold", reason="trending-market", size_quote=0)
        
        # Calculate grid levels
        recent_high = max(c.high for c in candle_list[-20:])
        recent_low = min(c.low for c in candle_list[-20:])
        mid_price = (recent_high + recent_low) / 2
        
        if self.entry_price is None:
            self.entry_price = price
        
        price_change = (price - self.entry_price) / self.entry_price
        
        # Grid logic
        if price_change <= -self.grid_size and self.last_action != "buy":
            self.last_action = "buy"
            self.entry_price = price
            return TradeDecision(
                action="buy", 
                reason=f"grid-buy at {price_change:.2%} below last entry",
                size_quote=0,
                stop_loss=price * 0.95
            )
        elif price_change >= self.grid_size and self.last_action != "sell":
            self.last_action = "sell"
            self.entry_price = price
            return TradeDecision(
                action="sell",
                reason=f"grid-sell at {price_change:.2%} above last entry", 
                size_quote=0,
                stop_loss=price * 1.05
            )
        
        return TradeDecision(action="hold", reason="within-grid", size_quote=0)


class DCAStrategy(Strategy):
    """
    Smart Dollar Cost Averaging strategy that adjusts timing based on market conditions.
    """
    
    def __init__(self, dca_interval_candles: int = 24, volatility_threshold: float = 0.05,
                 rsi_period: int = 14, buy_rsi_threshold: int = 50):
        self.dca_interval = dca_interval_candles
        self.volatility_threshold = volatility_threshold
        self.rsi_period = rsi_period
        self.buy_rsi_threshold = buy_rsi_threshold
        self.last_buy_candle = 0
        
    def _calculate_volatility(self, candles: List[Candle], period: int = 20) -> float:
        """Calculate price volatility"""
        if len(candles) < period:
            return 0.0
        
        prices = [c.close for c in candles[-period:]]
        returns = [prices[i]/prices[i-1] - 1 for i in range(1, len(prices))]
        return np.std(returns) if returns else 0.0
    
    def _rsi(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate RSI"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss.replace(0, np.nan))
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def on_candles(self, candles: Iterable[Candle], price: float) -> TradeDecision:
        candle_list = list(candles)
        if len(candle_list) < max(self.rsi_period, 20):
            return TradeDecision(action="hold", reason="insufficient-data", size_quote=0)
        
        # Calculate indicators
        volatility = self._calculate_volatility(candle_list)
        df = pd.DataFrame([{"close": c.close} for c in candle_list])
        rsi = self._rsi(df["close"], self.rsi_period).iloc[-1]
        
        current_candle = len(candle_list)
        candles_since_last_buy = current_candle - self.last_buy_candle
        
        # Smart DCA conditions
        conditions = [
            candles_since_last_buy >= self.dca_interval,  # Time-based
            rsi < self.buy_rsi_threshold,  # Oversold condition
            volatility > self.volatility_threshold  # High volatility (opportunity)
        ]
        
        if all(conditions):
            self.last_buy_candle = current_candle
            return TradeDecision(
                action="buy",
                reason=f"smart-dca: RSI={rsi:.1f}, vol={volatility:.3f}, interval={candles_since_last_buy}",
                size_quote=0,
                stop_loss=price * 0.90
            )
        
        return TradeDecision(
            action="hold", 
            reason=f"waiting-dca: RSI={rsi:.1f}, vol={volatility:.3f}, interval={candles_since_last_buy}",
            size_quote=0
        )


class ArbitrageStrategy(Strategy):
    """
    Statistical arbitrage strategy that identifies price discrepancies.
    """
    
    def __init__(self, correlation_threshold: float = 0.8, 
                 spread_threshold: float = 0.02, lookback_period: int = 100):
        self.correlation_threshold = correlation_threshold
        self.spread_threshold = spread_threshold
        self.lookback_period = lookback_period
        self.price_history = []
        
    def on_candles(self, candles: Iterable[Candle], price: float) -> TradeDecision:
        candle_list = list(candles)
        if len(candle_list) < self.lookback_period:
            return TradeDecision(action="hold", reason="insufficient-data", size_quote=0)
        
        # Store price history for analysis
        self.price_history.append(price)
        if len(self.price_history) > self.lookback_period:
            self.price_history.pop(0)
        
        # Calculate moving average and deviation
        ma = np.mean(self.price_history)
        std = np.std(self.price_history)
        z_score = (price - ma) / std if std > 0 else 0
        
        # Arbitrage signals based on statistical deviation
        if z_score < -2:  # Significantly undervalued
            return TradeDecision(
                action="buy",
                reason=f"statistical-arbitrage: z-score={z_score:.2f} (undervalued)",
                size_quote=0,
                stop_loss=price * 0.95,
                take_profit=ma
            )
        elif z_score > 2:  # Significantly overvalued
            return TradeDecision(
                action="sell",
                reason=f"statistical-arbitrage: z-score={z_score:.2f} (overvalued)",
                size_quote=0,
                stop_loss=price * 1.05,
                take_profit=ma
            )
        
        return TradeDecision(
            action="hold",
            reason=f"z-score={z_score:.2f} within normal range",
            size_quote=0
        )


class MultiTimeframeStrategy(Strategy):
    """
    Multi-timeframe analysis strategy that combines signals from different timeframes.
    """
    
    def __init__(self, short_ma: int = 9, long_ma: int = 21, 
                 trend_period: int = 50, rsi_period: int = 14):
        self.short_ma = short_ma
        self.long_ma = long_ma
        self.trend_period = trend_period
        self.rsi_period = rsi_period
        
    def _analyze_timeframe(self, candles: List[Candle], period_factor: int = 1) -> Dict:
        """Analyze a specific timeframe"""
        # Sample candles for higher timeframe analysis
        sampled_candles = candles[::period_factor] if period_factor > 1 else candles
        
        if len(sampled_candles) < max(self.long_ma, self.trend_period):
            return {"trend": "unknown", "signal": "hold", "strength": 0}
        
        df = pd.DataFrame([{"close": c.close} for c in sampled_candles])
        
        # Calculate indicators
        df["ma_short"] = df["close"].rolling(self.short_ma).mean()
        df["ma_long"] = df["close"].rolling(self.long_ma).mean()
        df["ma_trend"] = df["close"].rolling(self.trend_period).mean()
        
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else last
        
        # Determine trend
        if last["close"] > last["ma_trend"]:
            trend = "up"
        elif last["close"] < last["ma_trend"]:
            trend = "down"
        else:
            trend = "sideways"
        
        # Generate signal
        if last["ma_short"] > last["ma_long"] and prev["ma_short"] <= prev["ma_long"]:
            signal = "buy"
            strength = 2
        elif last["ma_short"] < last["ma_long"] and prev["ma_short"] >= prev["ma_long"]:
            signal = "sell"
            strength = 2
        elif last["ma_short"] > last["ma_long"]:
            signal = "buy"
            strength = 1
        elif last["ma_short"] < last["ma_long"]:
            signal = "sell"
            strength = 1
        else:
            signal = "hold"
            strength = 0
            
        return {"trend": trend, "signal": signal, "strength": strength}
    
    def on_candles(self, candles: Iterable[Candle], price: float) -> TradeDecision:
        candle_list = list(candles)
        if len(candle_list) < self.trend_period * 3:
            return TradeDecision(action="hold", reason="insufficient-data", size_quote=0)
        
        # Analyze multiple timeframes
        tf1 = self._analyze_timeframe(candle_list, 1)    # 1x (base timeframe)
        tf2 = self._analyze_timeframe(candle_list, 3)    # 3x (higher timeframe)
        tf3 = self._analyze_timeframe(candle_list, 9)    # 9x (longer timeframe)
        
        # Combine signals with weights
        signal_score = (
            tf1["strength"] * (1 if tf1["signal"] == "buy" else -1 if tf1["signal"] == "sell" else 0) * 1.0 +
            tf2["strength"] * (1 if tf2["signal"] == "buy" else -1 if tf2["signal"] == "sell" else 0) * 1.5 +
            tf3["strength"] * (1 if tf3["signal"] == "buy" else -1 if tf3["signal"] == "sell" else 0) * 2.0
        )
        
        reason = f"MTF: 1x={tf1['signal']}({tf1['strength']}), 3x={tf2['signal']}({tf2['strength']}), 9x={tf3['signal']}({tf3['strength']})"
        
        # Decision logic
        if signal_score >= 3:
            return TradeDecision(
                action="buy",
                reason=f"{reason}, score={signal_score:.1f}",
                size_quote=0,
                stop_loss=price * 0.95
            )
        elif signal_score <= -3:
            return TradeDecision(
                action="sell",
                reason=f"{reason}, score={signal_score:.1f}",
                size_quote=0,
                stop_loss=price * 1.05
            )
        
        return TradeDecision(
            action="hold",
            reason=f"{reason}, score={signal_score:.1f} (neutral)",
            size_quote=0
        )
