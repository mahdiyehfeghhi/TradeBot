from __future__ import annotations

from typing import Iterable

import pandas as pd

from domain.models import Candle, Ticker
from domain.ports import MarketDataPort


class CSVMarketData(MarketDataPort):
    def __init__(self, csv_path: str, symbol: str):
        self.df = pd.read_csv(csv_path)
        self.symbol = symbol

    async def get_latest_candles(self, symbol: str, limit: int) -> Iterable[Candle]:
        tail = self.df.tail(limit)
        return [
            Candle(
                timestamp=int(row["timestamp"]),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row.get("volume", 0.0)),
            )
            for _, row in tail.iterrows()
        ]

    async def get_ticker(self, symbol: str) -> Ticker:
        price = float(self.df.tail(1)["close"].values[0])
        return Ticker(symbol=symbol, price=price)
