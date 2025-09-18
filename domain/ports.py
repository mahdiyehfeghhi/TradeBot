from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Optional

from .models import Candle, Order, ExecutionReport, AccountBalance, Ticker, TradeDecision


class MarketDataPort(ABC):
    @abstractmethod
    async def get_latest_candles(self, symbol: str, limit: int) -> Iterable[Candle]:
        ...

    @abstractmethod
    async def get_ticker(self, symbol: str) -> Ticker:
        ...


class TradingPort(ABC):
    @abstractmethod
    async def place_order(self, order: Order) -> ExecutionReport:
        ...

    @abstractmethod
    async def get_balance(self, currency: str) -> AccountBalance:
        ...


class Strategy(ABC):
    @abstractmethod
    def on_candles(self, candles: Iterable[Candle], price: float) -> TradeDecision:
        ...


class RiskManager(ABC):
    @abstractmethod
    def position_size_quote(self, equity_quote: float, risk_pct: float, entry: float, stop: Optional[float]) -> float:
        ...
