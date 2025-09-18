from __future__ import annotations

from typing import Iterable, List
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
