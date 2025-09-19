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
    AdaptiveStrategy
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
                MeanReversionStrategy()
            ]
            return AdaptiveStrategy(
                strategies=strategies,
                performance_window=params.get("performance_window", 100)
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