from __future__ import annotations

import asyncio
import time
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from domain.models import TradingEvent, MarketPerformance
from domain.ports import MarketDataPort, TradingPort, Strategy
from application.memory import TradingMemory
from application.monitoring import ComprehensiveMonitor, MonitoringConfig
from application.sentiment import IntegratedSentimentEngine
from application.portfolio import IntelligentPortfolioEngine
from application.risk import DynamicRiskManager
from application.strategy_factory import StrategyFactory
from application.auto_optimizer import AutoOptimizer
from application.trader import TraderContext


@dataclass 
class SuperTradeBotConfig:
    # Trading parameters
    symbols: List[str]
    base_budget: float
    risk_per_trade_pct: float = 1.0
    max_portfolio_heat: float = 5.0
    max_drawdown_pct: float = 10.0
    
    # Strategy selection
    primary_strategy: str = "adaptive"
    fallback_strategy: str = "rsi_ma"
    enable_ml_strategies: bool = True
    enable_news_trading: bool = True
    
    # Auto-optimization
    enable_auto_optimization: bool = True
    optimization_interval_hours: int = 24
    
    # Monitoring and alerts
    enable_monitoring: bool = True
    enable_email_alerts: bool = False
    enable_telegram_alerts: bool = False
    
    # Advanced features
    enable_sentiment_analysis: bool = True
    enable_portfolio_optimization: bool = True
    enable_regime_detection: bool = True
    
    # Performance targets
    target_annual_return: float = 50.0  # 50% annual return target
    max_acceptable_drawdown: float = 15.0  # 15% max drawdown
    min_sharpe_ratio: float = 1.5  # Minimum Sharpe ratio


class SuperTradeBot:
    """
    The ultimate AI-powered cryptocurrency trading bot that combines all advanced features
    for maximum profitability and intelligent automation.
    """
    
    def __init__(self, market: MarketDataPort, broker: TradingPort, config: SuperTradeBotConfig):
        self.market = market
        self.broker = broker
        self.config = config
        
        # Core components
        self.memory = TradingMemory()
        self.risk_manager = DynamicRiskManager(
            base_risk_pct=config.risk_per_trade_pct,
            max_portfolio_heat=config.max_portfolio_heat,
            max_drawdown_pct=config.max_drawdown_pct
        )
        
        # Monitoring system
        if config.enable_monitoring:
            monitoring_config = MonitoringConfig(
                max_drawdown_alert=config.max_acceptable_drawdown * 0.8,
                email_enabled=config.enable_email_alerts,
                telegram_enabled=config.enable_telegram_alerts
            )
            self.monitor = ComprehensiveMonitor(self.memory, monitoring_config)
        else:
            self.monitor = None
        
        # Sentiment analysis
        if config.enable_sentiment_analysis:
            self.sentiment_engine = IntegratedSentimentEngine()
        else:
            self.sentiment_engine = None
        
        # Portfolio engine
        if config.enable_portfolio_optimization:
            self.portfolio_engine = self._create_portfolio_engine()
        else:
            self.portfolio_engine = None
        
        # Auto-optimizer
        if config.enable_auto_optimization:
            data_paths = {symbol: f"data/{symbol}_candles.csv" for symbol in config.symbols}
            self.auto_optimizer = AutoOptimizer(config.symbols, data_paths)
            self.last_optimization = 0
        else:
            self.auto_optimizer = None
        
        # Current strategy selection
        self.current_strategies: Dict[str, Strategy] = {}
        self.strategy_performance: Dict[str, float] = {}
        
        # Performance tracking
        self.start_time = int(time.time())
        self.initial_balance = config.base_budget
        self.current_balance = config.base_budget
        self.peak_balance = config.base_budget
        self.total_trades = 0
        self.winning_trades = 0
        
        # Initialize strategies
        self._initialize_strategies()
        
        logger.info(f"SuperTradeBot initialized with {len(config.symbols)} symbols and ${config.base_budget:,.0f} budget")
    
    def _create_portfolio_engine(self) -> IntelligentPortfolioEngine:
        """Create intelligent portfolio engine"""
        ctx_map = {}
        for symbol in self.config.symbols:
            ctx_map[symbol] = TraderContext(
                symbol=symbol,
                loop_interval_sec=60,
                risk_pct=self.config.risk_per_trade_pct,
                take_profit_rr=2.0,
                min_notional=10000,
                quote_currency="TMN",
                price_precision=0,
                quantity_precision=6
            )
        
        return IntelligentPortfolioEngine(
            market=self.market,
            broker=self.broker,
            strategy_factory=lambda: self._get_best_strategy_for_symbol("default"),
            risk=self.risk_manager,
            symbols=self.config.symbols,
            ctx_map=ctx_map,
            memory=self.memory,
            max_active_symbols=min(len(self.config.symbols), 5)
        )
    
    def _initialize_strategies(self):
        """Initialize trading strategies for each symbol"""
        for symbol in self.config.symbols:
            # Start with primary strategy
            strategy = self._create_strategy(self.config.primary_strategy, symbol)
            self.current_strategies[symbol] = strategy
            self.strategy_performance[symbol] = 0.0
            
            logger.info(f"Initialized {self.config.primary_strategy} strategy for {symbol}")
    
    def _create_strategy(self, strategy_name: str, symbol: str) -> Strategy:
        """Create a strategy instance for a specific symbol"""
        params = {
            "symbol": symbol,
            "market": self.market,
            "memory": self.memory
        }
        
        # Add ML and news trading parameters if enabled
        if strategy_name in ["news_reaction", "market_regime_news"] and self.config.enable_news_trading:
            params.update({"market": self.market, "memory": self.memory})
        
        try:
            return StrategyFactory.create_strategy(strategy_name, params)
        except Exception as e:
            logger.error(f"Failed to create {strategy_name} strategy for {symbol}: {e}")
            # Fallback to simple strategy
            return StrategyFactory.create_strategy(self.config.fallback_strategy, {})
    
    def _get_best_strategy_for_symbol(self, symbol: str) -> Strategy:
        """Get the best performing strategy for a symbol"""
        if symbol in self.current_strategies:
            return self.current_strategies[symbol]
        
        # Create default strategy
        return self._create_strategy(self.config.primary_strategy, symbol)
    
    async def run_trading_iteration(self) -> Dict[str, Any]:
        """Run a single trading iteration with all AI features"""
        iteration_start = time.time()
        results = {
            "timestamp": int(time.time()),
            "symbol_results": {},
            "portfolio_analytics": {},
            "ai_insights": {},
            "performance_metrics": {},
            "alerts": [],
            "recommendations": []
        }
        
        try:
            # 1. Update market sentiment if enabled
            sentiment_data = {}
            if self.sentiment_engine:
                sentiment_data = await self.sentiment_engine.get_comprehensive_sentiment(self.config.symbols)
                results["ai_insights"]["sentiment"] = sentiment_data
            
            # 2. Run portfolio optimization if enabled
            if self.portfolio_engine:
                portfolio_results = await self.portfolio_engine.run_intelligent_iteration(self.config.base_budget)
                results["portfolio_analytics"] = portfolio_results
                
                # Update current balance from portfolio results
                if "trading_results" in portfolio_results:
                    for trade_result in portfolio_results["trading_results"]:
                        if "equity_quote" in trade_result:
                            self.current_balance = trade_result["equity_quote"]
            
            # 3. Individual symbol analysis
            for symbol in self.config.symbols:
                symbol_result = await self._analyze_symbol(symbol, sentiment_data.get(symbol, {}))
                results["symbol_results"][symbol] = symbol_result
            
            # 4. Update performance tracking
            self._update_performance_metrics()
            results["performance_metrics"] = self._get_performance_metrics()
            
            # 5. Monitor and generate alerts
            if self.monitor:
                market_performances = await self._get_market_performances()
                self.monitor.update_trading_metrics(self.current_balance, market_performances)
                dashboard = self.monitor.get_monitoring_dashboard()
                results["alerts"] = dashboard.get("alerts", {}).get("recent", [])
            
            # 6. Strategy optimization check
            if self._should_run_optimization():
                await self._run_strategy_optimization()
            
            # 7. Generate AI recommendations
            recommendations = self._generate_ai_recommendations(results)
            results["recommendations"] = recommendations
            
            # 8. Performance and status summary
            results["ai_insights"]["performance_summary"] = {
                "current_balance": self.current_balance,
                "total_return": (self.current_balance - self.initial_balance) / self.initial_balance * 100,
                "peak_balance": self.peak_balance,
                "current_drawdown": (self.peak_balance - self.current_balance) / self.peak_balance * 100 if self.peak_balance > 0 else 0,
                "total_trades": self.total_trades,
                "win_rate": self.winning_trades / self.total_trades if self.total_trades > 0 else 0,
                "active_strategies": list(self.current_strategies.keys())
            }
            
            iteration_time = time.time() - iteration_start
            logger.info(f"Trading iteration completed in {iteration_time:.2f}s - Balance: ${self.current_balance:,.0f}")
            
        except Exception as e:
            logger.error(f"Error in trading iteration: {e}")
            results["error"] = str(e)
        
        return results
    
    async def _analyze_symbol(self, symbol: str, sentiment_data: Dict) -> Dict[str, Any]:
        """Analyze a single symbol with all available intelligence"""
        try:
            # Get current candles
            candles = await self.market.get_latest_candles(symbol, limit=100)
            ticker = await self.market.get_ticker(symbol)
            current_price = ticker.price
            
            # Get strategy decision
            strategy = self.current_strategies.get(symbol)
            if strategy:
                decision = strategy.on_candles(candles, current_price)
            else:
                decision = None
            
            # Enhance decision with sentiment if available
            if sentiment_data and decision:
                sentiment_signal = sentiment_data.get("overall_signal", "neutral")
                sentiment_confidence = sentiment_data.get("confidence", 0.0)
                
                # Modify decision based on sentiment
                if sentiment_signal == "bullish" and sentiment_confidence > 0.6:
                    if decision.action == "hold":
                        decision.action = "buy"
                        decision.reason += " + bullish-sentiment"
                elif sentiment_signal == "bearish" and sentiment_confidence > 0.6:
                    if decision.action == "hold":
                        decision.action = "sell"
                        decision.reason += " + bearish-sentiment"
            
            # Calculate position size with advanced risk management
            position_size = 0.0
            if decision and decision.action in ["buy", "sell"]:
                price_history = [c.close for c in candles]
                position_size = self.risk_manager.position_size_quote(
                    self.current_balance,
                    self.config.risk_per_trade_pct,
                    current_price,
                    decision.stop_loss,
                    price_history,
                    symbol,
                    type(strategy).__name__ if strategy else "unknown"
                )
            
            return {
                "symbol": symbol,
                "current_price": current_price,
                "decision": {
                    "action": decision.action if decision else "hold",
                    "reason": decision.reason if decision else "no-strategy",
                    "position_size": position_size,
                    "stop_loss": decision.stop_loss if decision else None,
                    "take_profit": decision.take_profit if decision else None
                },
                "sentiment": sentiment_data,
                "strategy_used": type(strategy).__name__ if strategy else "none",
                "risk_metrics": self.risk_manager.get_risk_metrics()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return {"symbol": symbol, "error": str(e)}
    
    async def _get_market_performances(self) -> List[MarketPerformance]:
        """Get market performance data for all symbols"""
        performances = []
        
        for symbol in self.config.symbols:
            try:
                candles = await self.market.get_latest_candles(symbol, limit=2)
                if len(candles) >= 2:
                    current = candles[-1]
                    previous = candles[-2]
                    
                    change_24h = (current.close - previous.close) / previous.close * 100
                    
                    performance = MarketPerformance(
                        symbol=symbol,
                        price_change_24h=change_24h,
                        volume_24h=current.volume,
                        current_price=current.close,
                        high_24h=current.high,
                        low_24h=current.low
                    )
                    performances.append(performance)
            except Exception as e:
                logger.error(f"Error getting performance for {symbol}: {e}")
        
        return performances
    
    def _update_performance_metrics(self):
        """Update performance tracking metrics"""
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance
        
        # Update strategy performance scores
        for symbol, strategy in self.current_strategies.items():
            # Simple performance score based on recent balance change
            # In real implementation, this would track individual trade outcomes
            current_score = (self.current_balance - self.initial_balance) / self.initial_balance
            self.strategy_performance[symbol] = current_score
            
            # Update risk manager with strategy performance
            if hasattr(self.risk_manager, 'update_strategy_performance'):
                self.risk_manager.update_strategy_performance(type(strategy).__name__, current_score)
    
    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        total_return = (self.current_balance - self.initial_balance) / self.initial_balance * 100
        current_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance * 100 if self.peak_balance > 0 else 0
        
        # Calculate time-based metrics
        days_running = (int(time.time()) - self.start_time) / 86400
        annualized_return = total_return * (365 / max(days_running, 1))
        
        return {
            "total_return_pct": total_return,
            "annualized_return_pct": annualized_return,
            "current_drawdown_pct": current_drawdown,
            "max_drawdown_pct": current_drawdown,  # Simplified
            "current_balance": self.current_balance,
            "peak_balance": self.peak_balance,
            "days_running": days_running,
            "total_trades": self.total_trades,
            "win_rate": self.winning_trades / self.total_trades if self.total_trades > 0 else 0,
            "target_achievement": {
                "return_target": total_return >= self.config.target_annual_return * (days_running / 365),
                "drawdown_target": current_drawdown <= self.config.max_acceptable_drawdown,
                "overall_status": "ON_TRACK" if total_return > 0 and current_drawdown < self.config.max_acceptable_drawdown else "NEEDS_ATTENTION"
            }
        }
    
    def _should_run_optimization(self) -> bool:
        """Check if strategy optimization should be run"""
        if not self.auto_optimizer:
            return False
        
        current_time = int(time.time())
        time_since_last = current_time - self.last_optimization
        
        # Run optimization based on time interval or performance triggers
        time_trigger = time_since_last > (self.config.optimization_interval_hours * 3600)
        
        # Performance trigger - run if drawdown is high
        current_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance * 100 if self.peak_balance > 0 else 0
        performance_trigger = current_drawdown > self.config.max_acceptable_drawdown * 0.7
        
        return time_trigger or performance_trigger
    
    async def _run_strategy_optimization(self):
        """Run automatic strategy optimization"""
        logger.info("Running automatic strategy optimization...")
        
        try:
            # Run optimization for key strategies
            strategies_to_optimize = ["rsi_ma", "breakout", "grid", "dca"]
            results = await self.auto_optimizer.optimize_all_strategies(strategies_to_optimize)
            
            # Get best strategy for each symbol
            best_strategies = self.auto_optimizer.get_best_strategy_per_symbol()
            
            # Update current strategies with optimized ones
            for symbol, (strategy_name, result) in best_strategies.items():
                if result.score > self.strategy_performance.get(symbol, 0):
                    logger.info(f"Updating strategy for {symbol} to {strategy_name} (score: {result.score:.4f})")
                    
                    # Create new optimized strategy
                    new_strategy = StrategyFactory.create_strategy(strategy_name, result.parameters)
                    self.current_strategies[symbol] = new_strategy
                    self.strategy_performance[symbol] = result.score
            
            self.last_optimization = int(time.time())
            logger.info("Strategy optimization completed successfully")
            
        except Exception as e:
            logger.error(f"Error during strategy optimization: {e}")
    
    def _generate_ai_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate AI-powered trading recommendations"""
        recommendations = []
        
        # Performance analysis
        metrics = results.get("performance_metrics", {})
        current_return = metrics.get("total_return_pct", 0)
        current_drawdown = metrics.get("current_drawdown_pct", 0)
        
        if current_return < 0:
            recommendations.append("⚠️ Portfolio underperforming - consider strategy diversification")
        elif current_return > self.config.target_annual_return * 0.5:
            recommendations.append("✅ Strong performance - maintain current strategy allocation")
        
        if current_drawdown > self.config.max_acceptable_drawdown * 0.8:
            recommendations.append("🛑 High drawdown detected - reduce position sizes and enable defensive mode")
        
        # Sentiment analysis recommendations
        sentiment_data = results.get("ai_insights", {}).get("sentiment", {})
        if sentiment_data:
            bullish_symbols = [s for s, data in sentiment_data.items() 
                             if data.get("overall_signal") == "bullish" and data.get("confidence", 0) > 0.7]
            bearish_symbols = [s for s, data in sentiment_data.items() 
                             if data.get("overall_signal") == "bearish" and data.get("confidence", 0) > 0.7]
            
            if bullish_symbols:
                recommendations.append(f"📈 Strong bullish sentiment detected for: {', '.join(bullish_symbols)}")
            if bearish_symbols:
                recommendations.append(f"📉 Strong bearish sentiment detected for: {', '.join(bearish_symbols)}")
        
        # Risk management recommendations
        portfolio_analytics = results.get("portfolio_analytics", {})
        if portfolio_analytics.get("portfolio_risks", {}).get("concentration_risk"):
            recommendations.append("⚖️ Portfolio concentration risk detected - diversify holdings")
        
        if portfolio_analytics.get("portfolio_risks", {}).get("volatility_risk"):
            recommendations.append("📊 High volatility risk - consider reducing position sizes")
        
        # Alert-based recommendations
        alerts = results.get("alerts", [])
        critical_alerts = [a for a in alerts if a.get("severity") == "critical"]
        if critical_alerts:
            recommendations.append("🚨 Critical alerts active - immediate attention required")
        
        return recommendations
    
    async def run_continuous_trading(self, max_iterations: Optional[int] = None):
        """Run continuous trading loop with all AI features"""
        logger.info("Starting SuperTradeBot continuous trading mode...")
        
        iteration_count = 0
        
        try:
            while True:
                # Run trading iteration
                results = await self.run_trading_iteration()
                
                # Log key metrics
                performance = results.get("performance_metrics", {})
                balance = performance.get("current_balance", 0)
                total_return = performance.get("total_return_pct", 0)
                
                logger.info(f"Iteration {iteration_count + 1}: Balance=${balance:,.0f} ({total_return:+.2f}%)")
                
                # Check performance targets
                target_status = performance.get("target_achievement", {}).get("overall_status", "UNKNOWN")
                if target_status == "NEEDS_ATTENTION":
                    logger.warning("Performance targets not being met - review and optimization recommended")
                
                # Print recommendations
                recommendations = results.get("recommendations", [])
                for rec in recommendations[:3]:  # Show top 3 recommendations
                    logger.info(f"💡 {rec}")
                
                iteration_count += 1
                
                # Check exit conditions
                if max_iterations and iteration_count >= max_iterations:
                    logger.info(f"Completed {max_iterations} iterations")
                    break
                
                # Sleep between iterations
                await asyncio.sleep(60)  # 1 minute intervals
                
        except KeyboardInterrupt:
            logger.info("Trading stopped by user")
        except Exception as e:
            logger.error(f"Critical error in trading loop: {e}")
        finally:
            logger.info("SuperTradeBot trading session ended")
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive bot status and analytics"""
        metrics = self._get_performance_metrics()
        
        return {
            "bot_info": {
                "version": "SuperTradeBot v2.0",
                "start_time": self.start_time,
                "symbols_trading": self.config.symbols,
                "strategies_active": list(self.current_strategies.keys()),
                "features_enabled": {
                    "ai_sentiment": self.config.enable_sentiment_analysis,
                    "portfolio_optimization": self.config.enable_portfolio_optimization,
                    "auto_optimization": self.config.enable_auto_optimization,
                    "ml_strategies": self.config.enable_ml_strategies,
                    "news_trading": self.config.enable_news_trading,
                    "monitoring": self.config.enable_monitoring
                }
            },
            "performance": metrics,
            "risk_status": self.risk_manager.get_risk_metrics() if self.risk_manager else {},
            "strategy_scores": self.strategy_performance,
            "recommendations": self._generate_ai_recommendations({"performance_metrics": metrics})
        }