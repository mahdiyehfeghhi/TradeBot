from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import asyncio
import time

from loguru import logger

from .trader import Trader, TraderContext
from .memory import TradingMemory
from .scanner import MarketScanner
from .strategies import TopPerformerStrategy, AdaptiveStrategy
from domain.ports import MarketDataPort, TradingPort, Strategy, RiskManager
from domain.models import MarketPerformance


@dataclass
class SymbolState:
    symbol: str
    ctx: TraderContext
    trader: Trader
    performance: Optional[MarketPerformance] = None
    last_updated: int = 0


class EnhancedPortfolioEngine:
    """
    Enhanced portfolio engine with multi-symbol trading, top performer selection,
    and intelligent capital allocation.
    """
    
    def __init__(
        self,
        market: MarketDataPort,
        broker: TradingPort,
        strategy_factory,
        risk: RiskManager,
        symbols: List[str],
        ctx_map: Dict[str, TraderContext],
        memory: TradingMemory = None,
        max_active_symbols: int = 3,
    ):
        self.market = market
        self.broker = broker
        self.risk = risk
        self.memory = memory or TradingMemory()
        self.scanner = MarketScanner(market, self.memory)
        self.max_active_symbols = max_active_symbols
        self.symbols = symbols
        self.ctx_map = ctx_map
        
        # Initialize trader states
        self.states: List[SymbolState] = []
        self.strategy_factory = strategy_factory
        self._initialize_traders()
        
        # Performance tracking
        self._last_performance_update = 0
        self._performance_update_interval = 900  # 15 minutes
        
    def _initialize_traders(self):
        """Initialize traders for all symbols"""
        for sym in self.symbols:
            strategy: Strategy = self.strategy_factory()
            trader = Trader(self.market, self.broker, strategy, self.risk, 
                          self.ctx_map[sym], self.memory)
            self.states.append(SymbolState(
                symbol=sym, 
                ctx=self.ctx_map[sym], 
                trader=trader,
                last_updated=int(time.time())
            ))
            
    async def update_market_performances(self) -> List[MarketPerformance]:
        """Update market performance data for all symbols"""
        current_time = int(time.time())
        if current_time - self._last_performance_update < self._performance_update_interval:
            return [state.performance for state in self.states if state.performance]
            
        logger.info("Updating market performances for portfolio")
        performances = await self.scanner.scan_symbols(self.symbols, top_k=len(self.symbols))
        
        # Update state performances
        performance_map = {p.symbol: p for p in performances}
        for state in self.states:
            if state.symbol in performance_map:
                state.performance = performance_map[state.symbol]
                state.last_updated = current_time
                
        self._last_performance_update = current_time
        return performances
        
    def get_top_performers(self, min_change: float = 2.0, top_k: int = None) -> List[SymbolState]:
        """Get top performing symbols that meet criteria"""
        if top_k is None:
            top_k = self.max_active_symbols
            
        eligible_states = []
        for state in self.states:
            if (state.performance and 
                state.performance.price_change_24h >= min_change and
                state.performance.volume_24h > 50000):  # Minimum volume threshold
                eligible_states.append(state)
                
        # Sort by performance and return top K
        eligible_states.sort(key=lambda s: s.performance.price_change_24h, reverse=True)
        return eligible_states[:top_k]
        
    async def allocate_capital_intelligently(self, total_budget: float) -> Dict[str, float]:
        """Intelligently allocate capital based on performance and strategy confidence"""
        await self.update_market_performances()
        top_performers = self.get_top_performers()
        
        if not top_performers:
            # Equal allocation if no clear winners
            allocation_per_symbol = total_budget / min(len(self.states), self.max_active_symbols)
            return {state.symbol: allocation_per_symbol for state in self.states[:self.max_active_symbols]}
            
        allocations = {}
        total_score = 0
        
        # Calculate performance-based scores
        for state in top_performers:
            perf = state.performance
            # Score based on price change, volume, and strategy historical performance
            price_score = min(perf.price_change_24h / 10, 2.0)  # Cap at 2.0
            volume_score = min(perf.volume_24h / 1000000, 1.5)  # Normalize volume
            
            # Get strategy performance from memory
            strategy_name = type(state.trader.strategy).__name__
            strategy_perf = self.memory.get_strategy_performance(strategy_name)
            strategy_score = 1.0
            if strategy_perf and strategy_perf.total_trades >= 5:
                strategy_score = min(strategy_perf.win_rate * 2, 2.0)
                
            combined_score = (price_score + volume_score + strategy_score) / 3
            allocations[state.symbol] = combined_score
            total_score += combined_score
            
        # Normalize allocations to total budget
        if total_score > 0:
            for symbol in allocations:
                allocations[symbol] = (allocations[symbol] / total_score) * total_budget
        else:
            # Fallback to equal allocation
            allocation_per_symbol = total_budget / len(top_performers)
            allocations = {state.symbol: allocation_per_symbol for state in top_performers}
            
        logger.info("Capital allocation: {alloc}", alloc=allocations)
        return allocations
        
    async def run_once(self, total_budget: float = None) -> List[Dict[str, Any]]:
        """Run one iteration of portfolio trading"""
        # Update market performances
        await self.update_market_performances()
        
        # Get intelligent capital allocation
        if total_budget:
            allocations = await self.allocate_capital_intelligently(total_budget)
        else:
            allocations = {}
            
        # Get top performers to focus on
        top_performers = self.get_top_performers()
        active_symbols = {state.symbol for state in top_performers}
        
        results: List[Dict[str, Any]] = []
        
        # Update top performer strategies with current market data
        market_performances = [state.performance for state in self.states if state.performance]
        
        for state in self.states:
            try:
                # Skip low performers unless they're currently held
                if state.symbol not in active_symbols and state.symbol not in allocations:
                    continue
                    
                # Update strategy if it's a top performer strategy
                if isinstance(state.trader.strategy, TopPerformerStrategy):
                    state.trader.strategy.update_market_performances(market_performances)
                    
                # Get budget allocation for this symbol
                budget_for_symbol = allocations.get(state.symbol)
                
                # Run trader iteration
                out = await state.trader.run_once(budget_for_symbol)
                
                # Add performance and allocation info
                out.update({
                    "symbol": state.symbol,
                    "budget_allocated": budget_for_symbol,
                    "performance_24h": state.performance.price_change_24h if state.performance else 0,
                    "is_top_performer": state.symbol in active_symbols
                })
                
                results.append(out)
                
            except Exception as e:
                logger.exception("Portfolio symbol {s} iteration error: {e}", s=state.symbol)
                
        return results
        
    async def get_portfolio_analytics(self) -> Dict[str, Any]:
        """Get comprehensive portfolio analytics"""
        await self.update_market_performances()
        
        # Portfolio performance metrics
        total_pnl = 0
        active_positions = 0
        top_performers = self.get_top_performers()
        
        # Strategy performance analysis
        strategy_performances = {}
        for state in self.states:
            strategy_name = type(state.trader.strategy).__name__
            if strategy_name not in strategy_performances:
                perf = self.memory.get_strategy_performance(strategy_name)
                if perf:
                    strategy_performances[strategy_name] = {
                        "win_rate": perf.win_rate,
                        "total_trades": perf.total_trades,
                        "total_pnl": perf.total_pnl,
                        "profit_factor": perf.profit_factor
                    }
                    
        # Market sentiment analysis
        performances = [state.performance for state in self.states if state.performance]
        market_sentiment = self.scanner.get_market_sentiment(performances)
        
        # Learning insights
        insights = self.memory.get_learning_insights()
        
        return {
            "portfolio_metrics": {
                "total_symbols": len(self.states),
                "active_symbols": len(top_performers),
                "top_performers": [
                    {
                        "symbol": state.symbol,
                        "change_24h": state.performance.price_change_24h,
                        "volume_24h": state.performance.volume_24h
                    } for state in top_performers
                ]
            },
            "strategy_performance": strategy_performances,
            "market_sentiment": market_sentiment,
            "learning_insights": insights,
            "last_updated": int(time.time())
        }


class PortfolioEngine:
    """Legacy portfolio engine for backward compatibility"""
    
    def __init__(
        self,
        market: MarketDataPort,
        broker: TradingPort,
        strategy_factory,
        risk: RiskManager,
        symbols: List[str],
        ctx_map: Dict[str, TraderContext],
    ):
        self.market = market
        self.broker = broker
        self.risk = risk
        self.states: List[SymbolState] = []
        for sym in symbols:
            strategy: Strategy = strategy_factory()
            trader = Trader(market, broker, strategy, risk, ctx_map[sym])
            self.states.append(SymbolState(symbol=sym, ctx=ctx_map[sym], trader=trader))

    async def run_once(self) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for st in self.states:
            try:
                out = await st.trader.run_once()
                results.append({"symbol": st.symbol, **out})
            except Exception as e:
                logger.exception("Portfolio symbol {s} iteration error: {e}", s=st.symbol)
        return results