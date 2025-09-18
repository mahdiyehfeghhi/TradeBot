from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd
from loguru import logger

from domain.models import Candle, TradeDecision
from domain.ports import Strategy


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
