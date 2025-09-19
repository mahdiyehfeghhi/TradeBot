from __future__ import annotations

import asyncio
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import itertools
import numpy as np
import pandas as pd
from pathlib import Path

from loguru import logger

from domain.models import Candle, TradingEvent, StrategyPerformance
from domain.ports import Strategy
from application.strategy_factory import StrategyFactory
from application.memory import TradingMemory
from backtest.backtester import Backtester


@dataclass
class BacktestResult:
    strategy_name: str
    parameters: Dict[str, Any]
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    total_trades: int
    profit_factor: float
    avg_trade: float
    best_trade: float
    worst_trade: float
    score: float  # Combined performance score


@dataclass
class OptimizationConfig:
    strategy_name: str
    symbol: str
    data_path: str
    optimization_metric: str = "sharpe_ratio"  # "total_return", "sharpe_ratio", "profit_factor"
    max_iterations: int = 100
    population_size: int = 20
    generations: int = 10
    parameter_ranges: Dict[str, Any] = None


class ParameterOptimizer:
    """
    Genetic Algorithm-based parameter optimization for trading strategies
    """
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.memory = TradingMemory()
        self.best_results: List[BacktestResult] = []
        
        # Default parameter ranges for different strategies
        self.default_ranges = {
            "rsi_ma": {
                "rsi_period": (10, 21),
                "rsi_buy": (20, 40),
                "rsi_sell": (60, 80),
                "ma_fast": (5, 15),
                "ma_slow": (15, 30)
            },
            "breakout": {
                "breakout_lookback": (10, 30)
            },
            "mean_reversion": {
                "bb_period": (15, 25),
                "bb_std": (1.5, 2.5),
                "rsi_period": (10, 21),
                "rsi_buy": (20, 40),
                "rsi_sell": (60, 80)
            },
            "grid": {
                "grid_size": (0.01, 0.05),
                "grid_levels": (3, 10),
                "range_detection_period": (30, 70)
            },
            "dca": {
                "dca_interval_candles": (12, 48),
                "volatility_threshold": (0.02, 0.08),
                "rsi_period": (10, 21),
                "buy_rsi_threshold": (30, 60)
            }
        }
        
        # Use provided ranges or defaults
        self.parameter_ranges = config.parameter_ranges or self.default_ranges.get(config.strategy_name, {})
    
    def generate_random_parameters(self) -> Dict[str, Any]:
        """Generate random parameters within specified ranges"""
        params = {}
        
        for param_name, (min_val, max_val) in self.parameter_ranges.items():
            if isinstance(min_val, int) and isinstance(max_val, int):
                params[param_name] = np.random.randint(min_val, max_val + 1)
            else:
                params[param_name] = np.random.uniform(min_val, max_val)
        
        return params
    
    def mutate_parameters(self, params: Dict[str, Any], mutation_rate: float = 0.3) -> Dict[str, Any]:
        """Mutate parameters with given probability"""
        mutated = params.copy()
        
        for param_name, value in params.items():
            if np.random.random() < mutation_rate:
                min_val, max_val = self.parameter_ranges[param_name]
                
                if isinstance(min_val, int) and isinstance(max_val, int):
                    mutated[param_name] = np.random.randint(min_val, max_val + 1)
                else:
                    # Add some noise to current value
                    noise = np.random.normal(0, (max_val - min_val) * 0.1)
                    new_val = value + noise
                    mutated[param_name] = np.clip(new_val, min_val, max_val)
        
        return mutated
    
    def crossover_parameters(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """Create offspring by combining two parameter sets"""
        offspring = {}
        
        for param_name in parent1.keys():
            if np.random.random() < 0.5:
                offspring[param_name] = parent1[param_name]
            else:
                offspring[param_name] = parent2[param_name]
        
        return offspring
    
    async def evaluate_parameters(self, params: Dict[str, Any]) -> BacktestResult:
        """Evaluate a set of parameters using backtesting"""
        try:
            # Create strategy with parameters
            strategy = StrategyFactory.create_strategy(self.config.strategy_name, params)
            
            # Run backtest
            backtester = Backtester()
            result = await backtester.run_backtest(
                strategy=strategy,
                data_path=self.config.data_path,
                symbol=self.config.symbol,
                initial_balance=100000
            )
            
            # Calculate performance metrics
            trades = result.get("trades", [])
            returns = result.get("returns", [])
            equity_curve = result.get("equity_curve", [])
            
            if not trades or not equity_curve:
                return BacktestResult(
                    strategy_name=self.config.strategy_name,
                    parameters=params,
                    total_return=0.0,
                    max_drawdown=100.0,
                    sharpe_ratio=-10.0,
                    win_rate=0.0,
                    total_trades=0,
                    profit_factor=0.0,
                    avg_trade=0.0,
                    best_trade=0.0,
                    worst_trade=0.0,
                    score=-100.0
                )
            
            # Calculate metrics
            total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0] * 100
            
            # Max drawdown
            peak = equity_curve[0]
            max_dd = 0
            for value in equity_curve:
                if value > peak:
                    peak = value
                dd = (peak - value) / peak * 100
                max_dd = max(max_dd, dd)
            
            # Trade statistics
            pnls = [trade.get("pnl", 0) for trade in trades]
            winning_trades = [pnl for pnl in pnls if pnl > 0]
            losing_trades = [pnl for pnl in pnls if pnl < 0]
            
            win_rate = len(winning_trades) / len(trades) if trades else 0
            avg_trade = np.mean(pnls) if pnls else 0
            best_trade = max(pnls) if pnls else 0
            worst_trade = min(pnls) if pnls else 0
            
            # Profit factor
            gross_profit = sum(winning_trades) if winning_trades else 0
            gross_loss = abs(sum(losing_trades)) if losing_trades else 1
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            
            # Sharpe ratio (simplified)
            if returns and len(returns) > 1:
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            else:
                sharpe_ratio = 0
            
            # Combined score
            score = self._calculate_score(total_return, max_dd, sharpe_ratio, win_rate, profit_factor)
            
            return BacktestResult(
                strategy_name=self.config.strategy_name,
                parameters=params,
                total_return=total_return,
                max_drawdown=max_dd,
                sharpe_ratio=sharpe_ratio,
                win_rate=win_rate,
                total_trades=len(trades),
                profit_factor=profit_factor,
                avg_trade=avg_trade,
                best_trade=best_trade,
                worst_trade=worst_trade,
                score=score
            )
            
        except Exception as e:
            logger.error(f"Error evaluating parameters: {e}")
            return BacktestResult(
                strategy_name=self.config.strategy_name,
                parameters=params,
                total_return=-100.0,
                max_drawdown=100.0,
                sharpe_ratio=-10.0,
                win_rate=0.0,
                total_trades=0,
                profit_factor=0.0,
                avg_trade=0.0,
                best_trade=0.0,
                worst_trade=0.0,
                score=-100.0
            )
    
    def _calculate_score(self, total_return: float, max_drawdown: float, 
                        sharpe_ratio: float, win_rate: float, profit_factor: float) -> float:
        """Calculate combined performance score"""
        
        # Normalize metrics to 0-1 scale
        return_score = min(total_return / 100, 1.0) if total_return > 0 else total_return / 100
        drawdown_score = max(0, 1.0 - max_drawdown / 50)  # Penalize >50% drawdown heavily
        sharpe_score = min(max(sharpe_ratio / 3, 0), 1.0)  # Cap at Sharpe 3.0
        win_rate_score = win_rate
        profit_factor_score = min(profit_factor / 3, 1.0) if profit_factor > 1 else 0
        
        # Weighted combination
        weights = {
            "return": 0.25,
            "drawdown": 0.25,
            "sharpe": 0.2,
            "win_rate": 0.15,
            "profit_factor": 0.15
        }
        
        score = (return_score * weights["return"] +
                drawdown_score * weights["drawdown"] +
                sharpe_score * weights["sharpe"] +
                win_rate_score * weights["win_rate"] +
                profit_factor_score * weights["profit_factor"])
        
        return score
    
    async def optimize(self) -> List[BacktestResult]:
        """Run genetic algorithm optimization"""
        logger.info(f"Starting optimization for {self.config.strategy_name} with {self.config.max_iterations} iterations")
        
        # Initialize population
        population = []
        for _ in range(self.config.population_size):
            params = self.generate_random_parameters()
            result = await self.evaluate_parameters(params)
            population.append(result)
            
        logger.info(f"Initial population evaluated. Best score: {max(population, key=lambda x: x.score).score:.4f}")
        
        # Evolution loop
        for generation in range(self.config.generations):
            logger.info(f"Generation {generation + 1}/{self.config.generations}")
            
            # Sort by fitness
            population.sort(key=lambda x: x.score, reverse=True)
            
            # Keep top performers
            elite_size = self.config.population_size // 4
            new_population = population[:elite_size]
            
            # Generate offspring
            while len(new_population) < self.config.population_size:
                # Tournament selection
                parent1 = self._tournament_selection(population, tournament_size=3)
                parent2 = self._tournament_selection(population, tournament_size=3)
                
                # Crossover
                offspring_params = self.crossover_parameters(parent1.parameters, parent2.parameters)
                
                # Mutation
                offspring_params = self.mutate_parameters(offspring_params)
                
                # Evaluate offspring
                offspring_result = await self.evaluate_parameters(offspring_params)
                new_population.append(offspring_result)
            
            population = new_population
            best_score = max(population, key=lambda x: x.score).score
            logger.info(f"Generation {generation + 1} completed. Best score: {best_score:.4f}")
        
        # Final sort and store results
        population.sort(key=lambda x: x.score, reverse=True)
        self.best_results = population[:10]  # Keep top 10
        
        logger.info(f"Optimization completed. Best parameters: {self.best_results[0].parameters}")
        logger.info(f"Best performance: Return={self.best_results[0].total_return:.2f}%, "
                   f"Sharpe={self.best_results[0].sharpe_ratio:.2f}, "
                   f"Win Rate={self.best_results[0].win_rate:.2%}")
        
        return self.best_results
    
    def _tournament_selection(self, population: List[BacktestResult], tournament_size: int = 3) -> BacktestResult:
        """Select parent using tournament selection"""
        tournament = np.random.choice(population, tournament_size, replace=False)
        return max(tournament, key=lambda x: x.score)


class AutoOptimizer:
    """
    Automatically optimize multiple strategies and select the best performer
    """
    
    def __init__(self, symbols: List[str], data_paths: Dict[str, str]):
        self.symbols = symbols
        self.data_paths = data_paths
        self.optimization_results: Dict[str, List[BacktestResult]] = {}
        
    async def optimize_all_strategies(self, strategies_to_optimize: List[str] = None) -> Dict[str, List[BacktestResult]]:
        """Optimize all available strategies"""
        
        if strategies_to_optimize is None:
            # Get all available strategies that can be optimized
            all_strategies = StrategyFactory.get_available_strategies()
            strategies_to_optimize = [s["name"] for s in all_strategies 
                                    if s["name"] in ["rsi_ma", "breakout", "mean_reversion", "grid", "dca"]]
        
        results = {}
        
        for symbol in self.symbols:
            if symbol not in self.data_paths:
                logger.warning(f"No data path for symbol {symbol}, skipping optimization")
                continue
            
            logger.info(f"Optimizing strategies for {symbol}")
            
            for strategy_name in strategies_to_optimize:
                logger.info(f"Optimizing {strategy_name} for {symbol}")
                
                config = OptimizationConfig(
                    strategy_name=strategy_name,
                    symbol=symbol,
                    data_path=self.data_paths[symbol],
                    max_iterations=50,  # Reduced for faster execution
                    population_size=15,
                    generations=8
                )
                
                optimizer = ParameterOptimizer(config)
                strategy_results = await optimizer.optimize()
                
                key = f"{symbol}_{strategy_name}"
                results[key] = strategy_results
                
                # Log best result
                best = strategy_results[0]
                logger.info(f"Best {strategy_name} for {symbol}: Return={best.total_return:.2f}%, "
                           f"Sharpe={best.sharpe_ratio:.2f}, Score={best.score:.4f}")
        
        self.optimization_results = results
        return results
    
    def get_best_strategy_per_symbol(self) -> Dict[str, Tuple[str, BacktestResult]]:
        """Get the best strategy for each symbol"""
        best_strategies = {}
        
        for symbol in self.symbols:
            symbol_results = []
            
            for key, results in self.optimization_results.items():
                if key.startswith(f"{symbol}_"):
                    strategy_name = key.split("_", 1)[1]
                    best_result = results[0]  # Best result for this strategy
                    symbol_results.append((strategy_name, best_result))
            
            if symbol_results:
                # Sort by score and get the best
                symbol_results.sort(key=lambda x: x[1].score, reverse=True)
                best_strategies[symbol] = symbol_results[0]
        
        return best_strategies
    
    def generate_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report"""
        report = {
            "summary": {
                "total_optimizations": len(self.optimization_results),
                "symbols_analyzed": len(self.symbols),
                "best_overall_score": 0.0,
                "best_overall_strategy": "",
                "best_overall_symbol": ""
            },
            "symbol_results": {},
            "strategy_rankings": {},
            "recommendations": []
        }
        
        # Find best overall result
        best_overall_score = -100.0
        best_overall_key = ""
        
        for key, results in self.optimization_results.items():
            best_result = results[0]
            if best_result.score > best_overall_score:
                best_overall_score = best_result.score
                best_overall_key = key
        
        if best_overall_key:
            symbol, strategy = best_overall_key.split("_", 1)
            report["summary"]["best_overall_score"] = best_overall_score
            report["summary"]["best_overall_strategy"] = strategy
            report["summary"]["best_overall_symbol"] = symbol
        
        # Get best strategy per symbol
        best_strategies = self.get_best_strategy_per_symbol()
        
        for symbol, (strategy_name, result) in best_strategies.items():
            report["symbol_results"][symbol] = {
                "best_strategy": strategy_name,
                "score": result.score,
                "total_return": result.total_return,
                "sharpe_ratio": result.sharpe_ratio,
                "max_drawdown": result.max_drawdown,
                "win_rate": result.win_rate,
                "parameters": result.parameters
            }
        
        # Strategy rankings (average score across symbols)
        strategy_scores = {}
        for key, results in self.optimization_results.items():
            symbol, strategy = key.split("_", 1)
            if strategy not in strategy_scores:
                strategy_scores[strategy] = []
            strategy_scores[strategy].append(results[0].score)
        
        for strategy, scores in strategy_scores.items():
            report["strategy_rankings"][strategy] = {
                "avg_score": np.mean(scores),
                "best_score": max(scores),
                "consistency": 1.0 - (np.std(scores) / np.mean(scores)) if np.mean(scores) > 0 else 0
            }
        
        # Generate recommendations
        sorted_strategies = sorted(report["strategy_rankings"].items(), 
                                 key=lambda x: x[1]["avg_score"], reverse=True)
        
        if sorted_strategies:
            best_strategy = sorted_strategies[0]
            report["recommendations"].append(
                f"Top performing strategy overall: {best_strategy[0]} "
                f"(avg score: {best_strategy[1]['avg_score']:.4f})"
            )
        
        # Check for consistently good performers
        consistent_strategies = [name for name, metrics in report["strategy_rankings"].items() 
                               if metrics["consistency"] > 0.8 and metrics["avg_score"] > 0.3]
        
        if consistent_strategies:
            report["recommendations"].append(
                f"Most consistent strategies: {', '.join(consistent_strategies)}"
            )
        
        return report
    
    def save_results(self, filepath: str):
        """Save optimization results to file"""
        import json
        
        # Convert results to serializable format
        serializable_results = {}
        for key, results in self.optimization_results.items():
            serializable_results[key] = [
                {
                    "strategy_name": r.strategy_name,
                    "parameters": r.parameters,
                    "total_return": r.total_return,
                    "max_drawdown": r.max_drawdown,
                    "sharpe_ratio": r.sharpe_ratio,
                    "win_rate": r.win_rate,
                    "total_trades": r.total_trades,
                    "profit_factor": r.profit_factor,
                    "score": r.score
                } for r in results
            ]
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Optimization results saved to {filepath}")


# Example usage function
async def run_optimization_example():
    """Example of how to run the optimization system"""
    
    # Setup
    symbols = ["BTC-TMN", "ETH-TMN"]
    data_paths = {
        "BTC-TMN": "data/sample_candles.csv",
        "ETH-TMN": "data/sample_candles.csv"  # You would have different files
    }
    
    # Create auto optimizer
    auto_optimizer = AutoOptimizer(symbols, data_paths)
    
    # Run optimization for selected strategies
    strategies_to_test = ["rsi_ma", "breakout", "grid"]
    results = await auto_optimizer.optimize_all_strategies(strategies_to_test)
    
    # Generate report
    report = auto_optimizer.generate_optimization_report()
    
    # Save results
    auto_optimizer.save_results("data/optimization_results.json")
    
    # Print summary
    print("\n=== Optimization Summary ===")
    print(f"Best overall strategy: {report['summary']['best_overall_strategy']}")
    print(f"Best overall symbol: {report['summary']['best_overall_symbol']}")
    print(f"Best overall score: {report['summary']['best_overall_score']:.4f}")
    
    print("\n=== Strategy Rankings ===")
    for strategy, metrics in sorted(report["strategy_rankings"].items(), 
                                   key=lambda x: x[1]["avg_score"], reverse=True):
        print(f"{strategy:15} - Avg Score: {metrics['avg_score']:.4f}, Consistency: {metrics['consistency']:.2f}")
    
    print("\n=== Recommendations ===")
    for rec in report["recommendations"]:
        print(f"• {rec}")


if __name__ == "__main__":
    asyncio.run(run_optimization_example())