from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import time


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


@dataclass
class MarketPerformance:
    """Model for tracking 24h market performance of symbols"""
    symbol: str
    price_change_24h: float  # percentage change
    volume_24h: float
    current_price: float
    high_24h: float
    low_24h: float
    timestamp: int = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = int(time.time())


@dataclass 
class TradingEvent:
    """Model for storing trading events and outcomes for learning"""
    timestamp: int
    symbol: str
    action: str  # "buy", "sell", "hold"
    reason: str
    entry_price: float
    exit_price: Optional[float] = None
    quantity: float = 0.0
    pnl: Optional[float] = None  # profit/loss in quote currency
    duration_minutes: Optional[int] = None
    market_conditions: Optional[dict] = None  # RSI, MA, volatility, etc.
    strategy_used: Optional[str] = None
    outcome: Optional[str] = None  # "win", "loss", "breakeven"
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = int(time.time())
            
    def calculate_outcome(self) -> str:
        """Calculate trade outcome based on PnL"""
        if self.pnl is None:
            return "pending"
        elif self.pnl > 0:
            return "win"
        elif self.pnl < 0:
            return "loss"
        else:
            return "breakeven"


@dataclass
class StrategyPerformance:
    """Model for tracking strategy performance metrics"""
    strategy_name: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0  # gross profit / gross loss
    sharpe_ratio: Optional[float] = None
    max_drawdown: float = 0.0
    last_updated: int = None
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = int(time.time())
    
    def update_metrics(self, trades: list[TradingEvent]):
        """Update performance metrics from list of trades"""
        if not trades:
            return
            
        self.total_trades = len(trades)
        winning_trades = [t for t in trades if t.pnl and t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl and t.pnl < 0]
        
        self.winning_trades = len(winning_trades)
        self.losing_trades = len(losing_trades)
        self.total_pnl = sum(t.pnl for t in trades if t.pnl)
        
        if self.total_trades > 0:
            self.win_rate = self.winning_trades / self.total_trades
            
        if winning_trades:
            self.avg_win = sum(t.pnl for t in winning_trades) / len(winning_trades)
            
        if losing_trades:
            self.avg_loss = abs(sum(t.pnl for t in losing_trades) / len(losing_trades))
            
        if self.avg_loss > 0:
            gross_profit = sum(t.pnl for t in winning_trades)
            gross_loss = abs(sum(t.pnl for t in losing_trades))
            self.profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            
        self.last_updated = int(time.time())
