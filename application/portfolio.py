from __future__ import annotations

import numpy as np
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


class AIPortfolioOptimizer:
    """
    AI-driven portfolio optimization with machine learning for capital allocation,
    risk management, and strategy selection.
    """
    
    def __init__(self, memory: TradingMemory, lookback_days: int = 30):
        self.memory = memory
        self.lookback_days = lookback_days
        self.strategy_performance_cache = {}
        self.market_regime_cache = {}
        
    def analyze_strategy_effectiveness(self, strategies: List[str], 
                                    market_conditions: Dict) -> Dict[str, float]:
        """Analyze which strategies work best in current market conditions"""
        scores = {}
        
        for strategy_name in strategies:
            # Get recent performance for this strategy
            recent_trades = self.memory.get_recent_trades_by_strategy(
                strategy_name, days=self.lookback_days
            )
            
            if not recent_trades:
                scores[strategy_name] = 0.5  # Neutral score for unknown strategies
                continue
            
            # Calculate performance metrics
            wins = sum(1 for trade in recent_trades if trade.pnl and trade.pnl > 0)
            total = len(recent_trades)
            win_rate = wins / total if total > 0 else 0.5
            
            # Total PnL and average trade
            total_pnl = sum(trade.pnl for trade in recent_trades if trade.pnl)
            avg_trade = total_pnl / total if total > 0 else 0
            
            # Market condition matching
            condition_score = self._calculate_condition_match(recent_trades, market_conditions)
            
            # Combined score
            performance_score = (win_rate * 0.4 + 
                               min(max(avg_trade / 1000, -1), 1) * 0.3 +  # Normalize avg trade
                               condition_score * 0.3)
            
            scores[strategy_name] = max(0.1, min(1.0, performance_score))
            
        return scores
    
    def _calculate_condition_match(self, trades: List, current_conditions: Dict) -> float:
        """Calculate how well strategy performs in current market conditions"""
        if not trades:
            return 0.5
        
        # Extract market conditions from trades
        volatility_scores = []
        trend_scores = []
        
        for trade in trades:
            if trade.market_conditions:
                conditions = trade.market_conditions
                
                # Volatility matching
                if 'volatility' in conditions and 'volatility' in current_conditions:
                    vol_diff = abs(conditions['volatility'] - current_conditions['volatility'])
                    volatility_scores.append(max(0, 1 - vol_diff))
                
                # Trend matching
                if 'trend' in conditions and 'trend' in current_conditions:
                    trend_match = 1.0 if conditions['trend'] == current_conditions['trend'] else 0.3
                    trend_scores.append(trend_match)
        
        # Average condition matching scores
        vol_score = np.mean(volatility_scores) if volatility_scores else 0.5
        trend_score = np.mean(trend_scores) if trend_scores else 0.5
        
        return (vol_score + trend_score) / 2
    
    def optimize_capital_allocation(self, symbols: List[str], total_capital: float,
                                  market_performances: List[MarketPerformance],
                                  strategy_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Optimize capital allocation using modern portfolio theory concepts
        adapted for crypto trading.
        """
        if not market_performances:
            # Equal allocation fallback
            allocation_per_symbol = total_capital / len(symbols)
            return {symbol: allocation_per_symbol for symbol in symbols}
        
        # Create performance mapping
        perf_map = {p.symbol: p for p in market_performances}
        
        # Calculate expected returns and risk for each symbol
        symbol_metrics = {}
        for symbol in symbols:
            if symbol not in perf_map:
                continue
            
            perf = perf_map[symbol]
            
            # Expected return (based on momentum and strategy effectiveness)
            momentum_score = min(perf.price_change_24h / 10, 1.0)  # Normalize to [-1, 1]
            strategy_effectiveness = strategy_scores.get(symbol, 0.5)
            expected_return = (momentum_score * 0.6 + strategy_effectiveness * 0.4)
            
            # Risk estimation (based on volatility proxy)
            price_range = (perf.high_24h - perf.low_24h) / perf.current_price
            risk = max(price_range, 0.01)  # Minimum 1% risk
            
            # Sharpe-like ratio
            risk_adjusted_return = expected_return / risk if risk > 0 else 0
            
            symbol_metrics[symbol] = {
                'expected_return': expected_return,
                'risk': risk,
                'sharpe': risk_adjusted_return,
                'volume': perf.volume_24h
            }
        
        # Filter symbols with insufficient volume
        min_volume = 100000  # Minimum volume threshold
        eligible_symbols = [s for s in symbol_metrics 
                           if symbol_metrics[s]['volume'] > min_volume]
        
        if not eligible_symbols:
            # Fallback to equal allocation
            allocation_per_symbol = total_capital / len(symbols)
            return {symbol: allocation_per_symbol for symbol in symbols}
        
        # Risk parity with performance weighting
        allocations = {}
        total_weight = 0
        
        for symbol in eligible_symbols:
            metrics = symbol_metrics[symbol]
            
            # Weight combines Sharpe ratio and inverse risk
            base_weight = max(metrics['sharpe'], 0.1)  # Minimum weight
            risk_adjustment = 1.0 / (1.0 + metrics['risk'])  # Penalize high risk
            
            weight = base_weight * risk_adjustment
            allocations[symbol] = weight
            total_weight += weight
        
        # Normalize to total capital
        if total_weight > 0:
            for symbol in allocations:
                allocations[symbol] = (allocations[symbol] / total_weight) * total_capital
        
        # Apply position limits (max 30% per symbol)
        max_allocation = total_capital * 0.3
        for symbol in allocations:
            allocations[symbol] = min(allocations[symbol], max_allocation)
        
        return allocations
    
    def detect_portfolio_risks(self, current_positions: Dict[str, Dict],
                              market_performances: List[MarketPerformance]) -> Dict[str, Any]:
        """Detect various portfolio risks and suggest actions"""
        risks = {
            "concentration_risk": False,
            "correlation_risk": False,
            "volatility_risk": False,
            "momentum_risk": False,
            "recommendations": []
        }
        
        if not current_positions:
            return risks
        
        total_exposure = sum(pos.get('value', 0) for pos in current_positions.values())
        
        # Concentration risk
        for symbol, position in current_positions.items():
            position_weight = position.get('value', 0) / total_exposure if total_exposure > 0 else 0
            if position_weight > 0.4:  # More than 40% in single position
                risks["concentration_risk"] = True
                risks["recommendations"].append(f"Reduce {symbol} position (currently {position_weight:.1%})")
        
        # Volatility risk
        perf_map = {p.symbol: p for p in market_performances}
        high_vol_symbols = []
        
        for symbol in current_positions:
            if symbol in perf_map:
                perf = perf_map[symbol]
                volatility = (perf.high_24h - perf.low_24h) / perf.current_price
                if volatility > 0.15:  # More than 15% daily range
                    high_vol_symbols.append(symbol)
        
        if len(high_vol_symbols) > len(current_positions) * 0.5:
            risks["volatility_risk"] = True
            risks["recommendations"].append("Consider reducing exposure to high volatility assets")
        
        # Momentum risk (all positions in same direction)
        positive_momentum = sum(1 for symbol in current_positions 
                               if symbol in perf_map and perf_map[symbol].price_change_24h > 0)
        total_positions = len(current_positions)
        
        if positive_momentum / total_positions > 0.8 or positive_momentum / total_positions < 0.2:
            risks["momentum_risk"] = True
            risks["recommendations"].append("Portfolio heavily skewed to one direction, consider diversification")
        
        return risks
    
    def suggest_rebalancing(self, current_allocations: Dict[str, float],
                           optimal_allocations: Dict[str, float],
                           threshold: float = 0.05) -> Dict[str, str]:
        """Suggest portfolio rebalancing actions"""
        suggestions = {}
        
        for symbol in set(list(current_allocations.keys()) + list(optimal_allocations.keys())):
            current = current_allocations.get(symbol, 0)
            optimal = optimal_allocations.get(symbol, 0)
            
            difference = optimal - current
            if abs(difference) > threshold * optimal:  # 5% threshold
                if difference > 0:
                    suggestions[symbol] = f"INCREASE by {difference:,.0f}"
                else:
                    suggestions[symbol] = f"DECREASE by {abs(difference):,.0f}"
        
        return suggestions


class IntelligentPortfolioEngine(EnhancedPortfolioEngine):
    """
    Next-generation portfolio engine with AI optimization, risk management,
    and automated rebalancing.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer = AIPortfolioOptimizer(self.memory)
        self.last_rebalance = 0
        self.rebalance_interval = 3600 * 4  # 4 hours
        self.current_allocations = {}
        
    async def run_intelligent_iteration(self, total_budget: float) -> Dict[str, Any]:
        """Run an intelligent portfolio iteration with AI optimization"""
        
        # Update market data
        await self.update_market_performances()
        market_performances = [state.performance for state in self.states if state.performance]
        
        # Get current market conditions
        market_conditions = self._analyze_market_conditions(market_performances)
        
        # Analyze strategy effectiveness for current conditions
        strategy_names = list(set(type(state.trader.strategy).__name__ for state in self.states))
        strategy_scores = self.optimizer.analyze_strategy_effectiveness(strategy_names, market_conditions)
        
        # Optimize capital allocation
        symbols = [state.symbol for state in self.states]
        optimal_allocations = self.optimizer.optimize_capital_allocation(
            symbols, total_budget, market_performances, strategy_scores
        )
        
        # Detect portfolio risks
        current_positions = self._get_current_positions()
        risks = self.optimizer.detect_portfolio_risks(current_positions, market_performances)
        
        # Check if rebalancing is needed
        rebalancing_suggestions = {}
        current_time = int(time.time())
        if (current_time - self.last_rebalance > self.rebalance_interval or 
            any(risks[key] for key in ["concentration_risk", "volatility_risk"])):
            
            rebalancing_suggestions = self.optimizer.suggest_rebalancing(
                self.current_allocations, optimal_allocations
            )
            self.last_rebalance = current_time
        
        # Execute trading iteration with optimized allocations
        results = await self.run_once(total_budget)
        self.current_allocations = optimal_allocations
        
        # Compile comprehensive report
        return {
            "trading_results": results,
            "market_conditions": market_conditions,
            "strategy_scores": strategy_scores,
            "optimal_allocations": optimal_allocations,
            "portfolio_risks": risks,
            "rebalancing_suggestions": rebalancing_suggestions,
            "ai_insights": {
                "top_strategy": max(strategy_scores.items(), key=lambda x: x[1]) if strategy_scores else None,
                "market_regime": market_conditions.get("regime", "unknown"),
                "risk_level": "HIGH" if any(risks[key] for key in ["concentration_risk", "volatility_risk"]) else "NORMAL"
            }
        }
    
    def _analyze_market_conditions(self, performances: List[MarketPerformance]) -> Dict:
        """Analyze current market conditions"""
        if not performances:
            return {"regime": "unknown", "volatility": 0.02, "trend": "neutral"}
        
        # Calculate market-wide metrics
        price_changes = [p.price_change_24h for p in performances]
        volatilities = [(p.high_24h - p.low_24h) / p.current_price for p in performances]
        
        avg_change = np.mean(price_changes)
        avg_volatility = np.mean(volatilities)
        
        # Determine market regime
        if avg_volatility > 0.1:
            regime = "volatile"
        elif abs(avg_change) > 5:
            regime = "trending"
        elif avg_volatility < 0.03:
            regime = "ranging"
        else:
            regime = "normal"
        
        # Determine trend
        if avg_change > 2:
            trend = "bullish"
        elif avg_change < -2:
            trend = "bearish"
        else:
            trend = "neutral"
        
        return {
            "regime": regime,
            "volatility": avg_volatility,
            "trend": trend,
            "avg_change_24h": avg_change,
            "market_breadth": sum(1 for p in price_changes if p > 0) / len(price_changes)
        }
    
    def _get_current_positions(self) -> Dict[str, Dict]:
        """Get current portfolio positions (mock implementation)"""
        # This would integrate with actual position tracking
        # For now, return empty dict
        return {}