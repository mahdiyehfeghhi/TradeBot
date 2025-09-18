from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from domain.models import Candle
from domain.ports import MarketDataPort


@dataclass
class SymbolScore:
    symbol: str
    volatility: float
    liquidity: float
    score: float


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean().fillna(method="bfill")


async def rank_symbols(
    market: MarketDataPort,
    symbols: List[str],
    *,
    lookback_candles: int = 200,
    top_k: int = 3,
) -> Tuple[List[SymbolScore], List[str]]:
    """Rank symbols by a combined score of volatility (ATR%) and liquidity (volume).

    Returns (scores_sorted, selected_symbols)
    """
    scores: List[SymbolScore] = []
    for sym in symbols:
        try:
            candles: Iterable[Candle] = await market.get_latest_candles(sym, limit=lookback_candles)
            df = pd.DataFrame([{ "high": c.high, "low": c.low, "close": c.close, "volume": c.volume } for c in candles])
            if len(df) < 30:
                logger.warning("Universe: not enough candles for {sym}", sym=sym)
                continue
            atr = _atr(df, period=14).iloc[-1]
            price = float(df["close"].iloc[-1])
            vol_pct = float(atr) / max(price, 1e-9)  # normalized volatility
            liq = float(df["volume"].rolling(20).mean().iloc[-1])
            # Simple score: normalized volatility * log(liquidity)
            score = vol_pct * np.log1p(max(liq, 0.0))
            scores.append(SymbolScore(symbol=sym, volatility=vol_pct, liquidity=liq, score=score))
        except Exception as e:
            # Log the exception details without causing formatting KeyError
            logger.exception("Universe: scoring failed for {sym}: {err}", sym=sym, err=e)

    scores_sorted = sorted(scores, key=lambda x: x.score, reverse=True)
    selected = [s.symbol for s in scores_sorted[: max(1, top_k)]]
    return scores_sorted, selected
