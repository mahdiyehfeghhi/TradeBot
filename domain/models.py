from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class Candle:
    timestamp: int  # ms or s per feed; normalize at ingestion
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class Ticker:
    symbol: str
    price: float


@dataclass
class Order:
    symbol: str
    side: OrderSide
    quantity: float
    price: Optional[float] = None  # market if None
    client_id: Optional[str] = None


@dataclass
class ExecutionReport:
    order: Order
    executed_qty: float
    avg_price: float
    status: str  # filled/partial/rejected
    id: Optional[str] = None


@dataclass
class AccountBalance:
    currency: str
    total: float
    available: float


@dataclass
class TradeDecision:
    action: str  # "buy", "sell", "hold"
    reason: str
    size_quote: float  # amount in quote currency to allocate (paper sizing)
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
