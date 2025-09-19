from __future__ import annotations

from typing import Dict, Any, List
from loguru import logger

from domain.ports import Strategy
from application.strategies import (
    RSIMAStrategy, 
    BreakoutStrategy, 
    MeanReversionStrategy, 
    EnsembleStrategy,
    TopPerformerStrategy,
    AdaptiveStrategy,
    GridTradingStrategy,
    DCAStrategy,
    ArbitrageStrategy,
    MultiTimeframeStrategy
)


class StrategyFactory:
    """Factory for creating trading strategies with various configurations"""
    
    @staticmethod
    def create_strategy(strategy_name: str, params: Dict[str, Any] = None) -> Strategy:
        """Create a strategy instance based on name and parameters"""
        if params is None:
            params = {}
            
        logger.info("Creating strategy: {name} with params: {params}", 
                   name=strategy_name, params=params)
        
        if strategy_name == "rsi_ma":
            return RSIMAStrategy(
                rsi_period=params.get("rsi_period", 14),
                rsi_buy=params.get("rsi_buy", 35),
                rsi_sell=params.get("rsi_sell", 65),
                ma_fast=params.get("ma_fast", 9),
                ma_slow=params.get("ma_slow", 21)
            )
            
        elif strategy_name == "breakout":
            return BreakoutStrategy(
                lookback=params.get("breakout_lookback", 20)
            )
            
        elif strategy_name == "mean_reversion":
            return MeanReversionStrategy(
                bb_period=params.get("bb_period", 20),
                bb_std=params.get("bb_std", 2.0),
                rsi_period=params.get("rsi_period", 14),
                rsi_buy=params.get("rsi_buy", 35),
                rsi_sell=params.get("rsi_sell", 65)
            )
            
        elif strategy_name == "top_performer":
            return TopPerformerStrategy(
                min_volume_24h=params.get("min_volume_24h", 1000000),
                min_price_change=params.get("min_price_change", 5.0),
                momentum_period=params.get("momentum_period", 10),
                rsi_period=params.get("rsi_period", 14)
            )
            
        elif strategy_name == "ensemble":
            # Create sub-strategies for ensemble
            strategies = [
                RSIMAStrategy(
                    rsi_period=params.get("rsi_period", 14),
                    rsi_buy=params.get("rsi_buy", 35),
                    rsi_sell=params.get("rsi_sell", 65),
                    ma_fast=params.get("ma_fast", 9),
                    ma_slow=params.get("ma_slow", 21)
                ),
                BreakoutStrategy(
                    lookback=params.get("breakout_lookback", 20)
                ),
                MeanReversionStrategy(
                    bb_period=params.get("bb_period", 20),
                    bb_std=params.get("bb_std", 2.0),
                    rsi_period=params.get("rsi_period", 14),
                    rsi_buy=params.get("rsi_buy", 35),
                    rsi_sell=params.get("rsi_sell", 65)
                )
            ]
            return EnsembleStrategy(
                strategies=strategies,
                threshold=params.get("ensemble_threshold", 2)
            )
            
        elif strategy_name == "adaptive":
            # Create strategies for adaptive selection
            strategies = [
                RSIMAStrategy(),
                BreakoutStrategy(),
                TopPerformerStrategy(),
                MeanReversionStrategy(),
                GridTradingStrategy(),
                DCAStrategy(),
                MultiTimeframeStrategy()
            ]
            return AdaptiveStrategy(
                strategies=strategies,
                performance_window=params.get("performance_window", 100)
            )
            
        elif strategy_name == "grid":
            return GridTradingStrategy(
                grid_size=params.get("grid_size", 0.02),
                grid_levels=params.get("grid_levels", 5),
                range_detection_period=params.get("range_detection_period", 50)
            )
            
        elif strategy_name == "dca":
            return DCAStrategy(
                dca_interval_candles=params.get("dca_interval_candles", 24),
                volatility_threshold=params.get("volatility_threshold", 0.05),
                rsi_period=params.get("rsi_period", 14),
                buy_rsi_threshold=params.get("buy_rsi_threshold", 50)
            )
            
        elif strategy_name == "arbitrage":
            return ArbitrageStrategy(
                correlation_threshold=params.get("correlation_threshold", 0.8),
                spread_threshold=params.get("spread_threshold", 0.02),
                lookback_period=params.get("lookback_period", 100)
            )
            
        elif strategy_name == "multi_timeframe":
            return MultiTimeframeStrategy(
                short_ma=params.get("short_ma", 9),
                long_ma=params.get("long_ma", 21),
                trend_period=params.get("trend_period", 50),
                rsi_period=params.get("rsi_period", 14)
            )
            
        else:
            logger.warning("Unknown strategy name: {name}, falling back to RSI+MA", name=strategy_name)
            return RSIMAStrategy()
            
    @staticmethod
    def get_available_strategies() -> List[Dict[str, Any]]:
        """Get list of available strategies with their descriptions"""
        return [
            {
                "name": "rsi_ma",
                "description": "RSI + Moving Average strategy with trend following",
                "parameters": ["rsi_period", "rsi_buy", "rsi_sell", "ma_fast", "ma_slow"]
            },
            {
                "name": "breakout",
                "description": "Donchian breakout strategy for momentum trading",
                "parameters": ["breakout_lookback"]
            },
            {
                "name": "mean_reversion",
                "description": "Bollinger Bands mean reversion with RSI filter",
                "parameters": ["bb_period", "bb_std", "rsi_period", "rsi_buy", "rsi_sell"]
            },
            {
                "name": "top_performer",
                "description": "Trade top performing symbols with 24h momentum",
                "parameters": ["min_volume_24h", "min_price_change", "momentum_period", "rsi_period"]
            },
            {
                "name": "ensemble",
                "description": "Ensemble of multiple strategies with voting",
                "parameters": ["ensemble_threshold", "rsi_period", "breakout_lookback", "bb_period"]
            },
            {
                "name": "adaptive",
                "description": "Adaptive strategy selection based on performance",
                "parameters": ["performance_window"]
            },
            {
                "name": "grid",
                "description": "Grid trading for ranging markets with dynamic levels",
                "parameters": ["grid_size", "grid_levels", "range_detection_period"]
            },
            {
                "name": "dca",
                "description": "Smart Dollar Cost Averaging with volatility timing",
                "parameters": ["dca_interval_candles", "volatility_threshold", "rsi_period", "buy_rsi_threshold"]
            },
            {
                "name": "arbitrage", 
                "description": "Statistical arbitrage based on price deviations",
                "parameters": ["correlation_threshold", "spread_threshold", "lookback_period"]
            },
            {
                "name": "multi_timeframe",
                "description": "Multi-timeframe analysis with weighted signals",
                "parameters": ["short_ma", "long_ma", "trend_period", "rsi_period"]
            }
        ]
        
    @staticmethod
    def get_strategy_info(strategy_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific strategy"""
        strategies = StrategyFactory.get_available_strategies()
        for strategy in strategies:
            if strategy["name"] == strategy_name:
                return strategy
        return {"name": strategy_name, "description": "Unknown strategy", "parameters": []}