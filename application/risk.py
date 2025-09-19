from __future__ import annotations

import time
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta

from domain.ports import RiskManager
from domain.models import TradingEvent


class SimpleRiskManager(RiskManager):
    def __init__(self, take_profit_rr: float = 1.5):
        self.take_profit_rr = take_profit_rr

    def position_size_quote(self, equity_quote: float, risk_pct: float, entry: float, stop: float | None) -> float:
        # Risk capital in quote currency
        risk_cap = max(equity_quote * (risk_pct / 100.0), 0)
        if stop is None or stop <= 0 or entry <= 0:
            # fallback to a tiny position
            return min(risk_cap, equity_quote)
        risk_per_unit = abs(entry - stop)
        if risk_per_unit <= 0:
            return min(risk_cap, equity_quote)
        # For quote-denominated size, approximate unit risk in quote terms
        # If entry is price (quote/base), risk_per_unit is also in quote per base unit.
        # So size in base = risk_cap / risk_per_unit, size in quote = size_base * entry
        size_base = risk_cap / risk_per_unit
        size_quote = size_base * entry
        return min(size_quote, equity_quote)


class AdvancedRiskManager(RiskManager):
    """
    Advanced risk management system with dynamic position sizing,
    portfolio heat monitoring, and drawdown protection.
    """
    
    def __init__(self, 
                 base_risk_pct: float = 1.0,
                 max_portfolio_heat: float = 5.0,
                 max_drawdown_pct: float = 10.0,
                 volatility_lookback: int = 20,
                 correlation_threshold: float = 0.7,
                 take_profit_rr: float = 2.0):
        self.base_risk_pct = base_risk_pct
        self.max_portfolio_heat = max_portfolio_heat
        self.max_drawdown_pct = max_drawdown_pct
        self.volatility_lookback = volatility_lookback
        self.correlation_threshold = correlation_threshold
        self.take_profit_rr = take_profit_rr
        
        # State tracking
        self.equity_history: List[float] = []
        self.peak_equity: float = 0.0
        self.current_heat: float = 0.0
        self.open_positions: Dict[str, Dict] = {}
        self.trade_history: List[TradingEvent] = []
        
    def update_equity(self, current_equity: float):
        """Update equity tracking for drawdown calculation"""
        self.equity_history.append(current_equity)
        if len(self.equity_history) > 252:  # Keep ~1 year of data
            self.equity_history.pop(0)
            
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
    
    def get_current_drawdown(self) -> float:
        """Calculate current drawdown percentage"""
        if not self.equity_history or self.peak_equity == 0:
            return 0.0
        
        current_equity = self.equity_history[-1]
        return (self.peak_equity - current_equity) / self.peak_equity * 100
    
    def calculate_volatility_adjustment(self, price_history: List[float]) -> float:
        """Calculate volatility-based position size adjustment"""
        if len(price_history) < self.volatility_lookback:
            return 1.0
        
        # Calculate returns
        returns = [price_history[i]/price_history[i-1] - 1 
                  for i in range(1, min(len(price_history), self.volatility_lookback + 1))]
        
        if not returns:
            return 1.0
        
        volatility = np.std(returns)
        avg_volatility = 0.02  # Assume 2% average daily volatility
        
        # Reduce size in high volatility, increase in low volatility
        volatility_factor = avg_volatility / max(volatility, 0.005)
        return max(0.1, min(2.0, volatility_factor))  # Limit between 10% and 200%
    
    def calculate_portfolio_heat(self) -> float:
        """Calculate current portfolio heat (total risk exposure)"""
        total_risk = 0.0
        for symbol, position in self.open_positions.items():
            if position.get('risk_amount'):
                total_risk += position['risk_amount']
        return total_risk
    
    def add_position(self, symbol: str, entry_price: float, stop_loss: float, 
                     position_size: float, risk_amount: float):
        """Track a new open position"""
        self.open_positions[symbol] = {
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'position_size': position_size,
            'risk_amount': risk_amount,
            'timestamp': int(time.time())
        }
        self.current_heat = self.calculate_portfolio_heat()
    
    def remove_position(self, symbol: str):
        """Remove a closed position"""
        if symbol in self.open_positions:
            del self.open_positions[symbol]
            self.current_heat = self.calculate_portfolio_heat()
    
    def position_size_quote(self, equity_quote: float, risk_pct: float, 
                           entry: float, stop: float | None, 
                           price_history: Optional[List[float]] = None,
                           symbol: str = "UNKNOWN") -> float:
        """
        Advanced position sizing with multiple risk factors
        """
        # Update equity tracking
        self.update_equity(equity_quote)
        
        # Base risk calculation
        base_risk_amount = equity_quote * (self.base_risk_pct / 100.0)
        
        # Drawdown protection
        current_drawdown = self.get_current_drawdown()
        if current_drawdown > self.max_drawdown_pct:
            return 0.0  # Stop trading during excessive drawdown
        
        # Reduce size as drawdown increases
        drawdown_factor = max(0.1, 1.0 - (current_drawdown / self.max_drawdown_pct) * 0.5)
        
        # Portfolio heat protection
        if self.current_heat >= self.max_portfolio_heat:
            return 0.0  # Don't add new positions if heat is too high
        
        # Available heat for new position
        available_heat = self.max_portfolio_heat - self.current_heat
        heat_factor = min(1.0, available_heat / self.base_risk_pct)
        
        # Volatility adjustment
        volatility_factor = 1.0
        if price_history:
            volatility_factor = self.calculate_volatility_adjustment(price_history)
        
        # Calculate final risk amount
        adjusted_risk = base_risk_amount * drawdown_factor * heat_factor * volatility_factor
        
        # Position size calculation
        if stop is None or stop <= 0 or entry <= 0:
            return min(adjusted_risk, equity_quote * 0.05)  # Max 5% without stop
        
        risk_per_unit = abs(entry - stop)
        if risk_per_unit <= 0:
            return min(adjusted_risk, equity_quote * 0.05)
        
        # Calculate position size
        size_base = adjusted_risk / risk_per_unit
        size_quote = size_base * entry
        
        # Final size with all constraints
        final_size = min(size_quote, equity_quote * 0.2)  # Max 20% of equity per trade
        
        return final_size
    
    def should_reduce_positions(self) -> bool:
        """Check if positions should be reduced due to risk"""
        return (self.get_current_drawdown() > self.max_drawdown_pct * 0.8 or 
                self.current_heat > self.max_portfolio_heat * 0.9)
    
    def get_risk_metrics(self) -> Dict:
        """Get current risk metrics for monitoring"""
        return {
            "current_drawdown": self.get_current_drawdown(),
            "portfolio_heat": self.current_heat,
            "max_drawdown": self.max_drawdown_pct,
            "max_heat": self.max_portfolio_heat,
            "open_positions": len(self.open_positions),
            "peak_equity": self.peak_equity,
            "current_equity": self.equity_history[-1] if self.equity_history else 0,
            "risk_status": "HIGH" if self.should_reduce_positions() else "NORMAL"
        }


class DynamicRiskManager(AdvancedRiskManager):
    """
    Dynamic risk manager that adapts to market conditions and strategy performance
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.market_regime = "normal"  # normal, volatile, trending, ranging
        self.strategy_performance: Dict[str, float] = {}
        
    def detect_market_regime(self, price_history: List[float]) -> str:
        """Detect current market regime for risk adjustment"""
        if len(price_history) < 20:
            return "normal"
        
        # Calculate metrics
        returns = [price_history[i]/price_history[i-1] - 1 for i in range(1, len(price_history))]
        volatility = np.std(returns[-20:]) if len(returns) >= 20 else 0
        
        # Price range analysis
        recent_prices = price_history[-20:]
        price_range = (max(recent_prices) - min(recent_prices)) / min(recent_prices)
        
        # Trend strength
        ma_short = np.mean(price_history[-5:])
        ma_long = np.mean(price_history[-20:])
        trend_strength = abs(ma_short - ma_long) / ma_long if ma_long > 0 else 0
        
        # Regime classification
        if volatility > 0.05:
            return "volatile"
        elif trend_strength > 0.03:
            return "trending"
        elif price_range < 0.05:
            return "ranging"
        else:
            return "normal"
    
    def update_strategy_performance(self, strategy_name: str, performance_score: float):
        """Update strategy performance for risk adjustment"""
        self.strategy_performance[strategy_name] = performance_score
    
    def position_size_quote(self, equity_quote: float, risk_pct: float, 
                           entry: float, stop: float | None, 
                           price_history: Optional[List[float]] = None,
                           symbol: str = "UNKNOWN",
                           strategy_name: str = "unknown") -> float:
        
        # Detect market regime
        if price_history:
            self.market_regime = self.detect_market_regime(price_history)
        
        # Base size from parent class
        base_size = super().position_size_quote(
            equity_quote, risk_pct, entry, stop, price_history, symbol
        )
        
        # Market regime adjustments
        regime_factor = {
            "volatile": 0.5,   # Reduce size in volatile markets
            "trending": 1.2,   # Increase size in trending markets
            "ranging": 0.8,    # Reduce size in ranging markets
            "normal": 1.0
        }.get(self.market_regime, 1.0)
        
        # Strategy performance adjustment
        strategy_factor = 1.0
        if strategy_name in self.strategy_performance:
            perf = self.strategy_performance[strategy_name]
            # Increase size for good performing strategies, reduce for poor ones
            strategy_factor = max(0.2, min(2.0, 1.0 + perf))
        
        # Apply adjustments
        adjusted_size = base_size * regime_factor * strategy_factor
        
        return min(adjusted_size, equity_quote * 0.15)  # Conservative max 15%
