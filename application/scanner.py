from __future__ import annotations

import asyncio
import time
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta

import pandas as pd
from loguru import logger

from domain.models import MarketPerformance, Candle
from domain.ports import MarketDataPort
from application.memory import TradingMemory


class MarketScanner:
    """
    Advanced market scanner that identifies top-performing symbols
    and provides market intelligence for trading decisions.
    """
    
    def __init__(self, market: MarketDataPort, memory: TradingMemory, 
                 scan_interval_minutes: int = 15):
        self.market = market
        self.memory = memory
        self.scan_interval = scan_interval_minutes * 60  # Convert to seconds
        self._last_scan = 0
        self._cached_performances: List[MarketPerformance] = []
        self._is_scanning = False
        
    async def scan_symbols(self, symbols: List[str], 
                          min_volume_24h: float = 100000,
                          top_k: int = 5) -> List[MarketPerformance]:
        """
        Scan multiple symbols and return top performers by 24h change.
        """
        if self._is_scanning:
            logger.info("Scan already in progress, returning cached results")
            return self._cached_performances[:top_k]
            
        current_time = int(time.time())
        if current_time - self._last_scan < self.scan_interval and self._cached_performances:
            logger.info("Using cached scan results (scan interval not elapsed)")
            return self._cached_performances[:top_k]
            
        logger.info("Starting market scan of {n} symbols", n=len(symbols))
        self._is_scanning = True
        
        try:
            performances = []
            
            for symbol in symbols:
                try:
                    perf = await self._analyze_symbol_performance(symbol)
                    if perf and perf.volume_24h >= min_volume_24h:
                        performances.append(perf)
                        # Store in memory for later analysis
                        self.memory.store_market_performance(perf)
                        
                except Exception as e:
                    logger.warning("Failed to analyze symbol {sym}: {err}", sym=symbol, err=e)
                    continue
                    
                # Small delay to respect rate limits
                await asyncio.sleep(0.1)
                
            # Sort by 24h performance (descending)
            performances.sort(key=lambda x: x.price_change_24h, reverse=True)
            
            self._cached_performances = performances
            self._last_scan = current_time
            
            logger.info("Market scan completed. Top performers: {top}", 
                       top=[(p.symbol, f"{p.price_change_24h:.2f}%") for p in performances[:top_k]])
            
            return performances[:top_k]
            
        finally:
            self._is_scanning = False
            
    async def _analyze_symbol_performance(self, symbol: str) -> Optional[MarketPerformance]:
        """Analyze 24h performance for a single symbol"""
        try:
            # Get 24-48 hours of candles to calculate 24h change
            candles = await self.market.get_latest_candles(symbol, limit=48)
            candles_list = list(candles)
            
            if len(candles_list) < 24:
                logger.warning("Insufficient candle data for {sym}", sym=symbol)
                return None
                
            # Get current price
            ticker = await self.market.get_ticker(symbol)
            current_price = ticker.price
            
            if current_price <= 0:
                return None
                
            # Calculate 24h metrics
            df = pd.DataFrame([{
                "timestamp": c.timestamp,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
            } for c in candles_list])
            
            # Sort by timestamp to ensure chronological order
            df = df.sort_values('timestamp')
            
            # Calculate 24h change (use close price from 24 candles ago)
            price_24h_ago = df["close"].iloc[-24] if len(df) >= 24 else df["close"].iloc[0]
            price_change_24h = ((current_price - price_24h_ago) / price_24h_ago) * 100
            
            # Calculate 24h high/low and volume
            recent_24h = df.tail(24)
            high_24h = max(recent_24h["high"].max(), current_price)
            low_24h = min(recent_24h["low"].min(), current_price)
            volume_24h = recent_24h["volume"].sum()
            
            return MarketPerformance(
                symbol=symbol,
                price_change_24h=price_change_24h,
                volume_24h=volume_24h,
                current_price=current_price,
                high_24h=high_24h,
                low_24h=low_24h,
                timestamp=int(time.time())
            )
            
        except Exception as e:
            logger.error("Error analyzing {sym}: {err}", sym=symbol, err=e)
            return None
            
    def get_market_sentiment(self, performances: List[MarketPerformance]) -> Dict:
        """Analyze overall market sentiment from performance data"""
        if not performances:
            return {"sentiment": "neutral", "strength": 0.0}
            
        positive_count = sum(1 for p in performances if p.price_change_24h > 0)
        negative_count = sum(1 for p in performances if p.price_change_24h < 0)
        total_count = len(performances)
        
        avg_change = sum(p.price_change_24h for p in performances) / total_count
        positive_ratio = positive_count / total_count
        
        # Determine sentiment
        if positive_ratio > 0.6 and avg_change > 2.0:
            sentiment = "bullish"
            strength = min(avg_change / 10, 1.0)  # Normalize to 0-1
        elif positive_ratio < 0.4 and avg_change < -2.0:
            sentiment = "bearish"
            strength = min(abs(avg_change) / 10, 1.0)
        else:
            sentiment = "neutral"
            strength = 0.5
            
        return {
            "sentiment": sentiment,
            "strength": strength,
            "positive_ratio": positive_ratio,
            "avg_change_24h": avg_change,
            "top_gainer": max(performances, key=lambda x: x.price_change_24h),
            "top_loser": min(performances, key=lambda x: x.price_change_24h)
        }
        
    async def get_correlation_matrix(self, symbols: List[str], 
                                   days: int = 7) -> pd.DataFrame:
        """Calculate price correlation matrix between symbols"""
        try:
            price_data = {}
            
            for symbol in symbols:
                candles = await self.market.get_latest_candles(symbol, limit=days * 24)
                candles_list = list(candles)
                
                if len(candles_list) < days * 12:  # At least half the expected data
                    continue
                    
                df = pd.DataFrame([{
                    "timestamp": c.timestamp,
                    "close": c.close,
                } for c in candles_list])
                
                df = df.sort_values('timestamp')
                df['returns'] = df['close'].pct_change()
                price_data[symbol] = df['returns'].dropna()
                
            if len(price_data) < 2:
                return pd.DataFrame()
                
            # Align data by creating a common DataFrame
            combined_data = pd.DataFrame(price_data)
            correlation_matrix = combined_data.corr()
            
            return correlation_matrix
            
        except Exception as e:
            logger.error("Error calculating correlation matrix: {err}", err=e)
            return pd.DataFrame()
            
    def identify_momentum_opportunities(self, performances: List[MarketPerformance],
                                     min_momentum: float = 5.0,
                                     min_volume_ratio: float = 1.5) -> List[MarketPerformance]:
        """
        Identify symbols with strong momentum and volume confirmation.
        """
        opportunities = []
        
        for perf in performances:
            # Check price momentum
            if abs(perf.price_change_24h) < min_momentum:
                continue
                
            # Calculate volume ratio (current vs typical)
            # This is simplified - in a real implementation, you'd compare
            # against historical average volume
            volume_score = 1.0  # Placeholder for volume analysis
            
            # Price position within 24h range
            price_range = perf.high_24h - perf.low_24h
            if price_range <= 0:
                continue
                
            price_position = (perf.current_price - perf.low_24h) / price_range
            
            # For bullish momentum: price should be near highs
            # For bearish momentum: price should be near lows
            if perf.price_change_24h > 0:
                momentum_quality = price_position  # Higher is better for bullish
            else:
                momentum_quality = 1 - price_position  # Lower is better for bearish
                
            # Score the opportunity
            if momentum_quality > 0.7:  # Strong momentum quality
                opportunities.append(perf)
                
        return opportunities
        
    async def get_enhanced_market_data(self, symbol: str) -> Dict:
        """Get comprehensive market data for a symbol"""
        try:
            # Get performance data
            perf = await self._analyze_symbol_performance(symbol)
            if not perf:
                return {}
                
            # Get recent candles for technical analysis
            candles = await self.market.get_latest_candles(symbol, limit=100)
            candles_list = list(candles)
            
            if len(candles_list) < 20:
                return {"performance": perf}
                
            df = pd.DataFrame([{
                "close": c.close,
                "volume": c.volume,
                "high": c.high,
                "low": c.low
            } for c in candles_list])
            
            # Calculate technical indicators
            df['sma_20'] = df['close'].rolling(20).mean()
            df['sma_50'] = df['close'].rolling(50).mean() if len(df) >= 50 else None
            df['volatility'] = df['close'].pct_change().rolling(20).std() * 100
            
            # Volume analysis
            df['volume_sma'] = df['volume'].rolling(20).mean()
            current_volume_ratio = df['volume'].iloc[-1] / df['volume_sma'].iloc[-1] if df['volume_sma'].iloc[-1] > 0 else 1.0
            
            # Support/Resistance levels
            recent_highs = df['high'].rolling(20).max()
            recent_lows = df['low'].rolling(20).min()
            
            return {
                "performance": perf,
                "technical": {
                    "sma_20": df['sma_20'].iloc[-1] if len(df) >= 20 else None,
                    "sma_50": df['sma_50'].iloc[-1] if len(df) >= 50 else None,
                    "volatility": df['volatility'].iloc[-1],
                    "volume_ratio": current_volume_ratio,
                    "resistance": recent_highs.iloc[-1],
                    "support": recent_lows.iloc[-1],
                    "trend": "bullish" if perf.current_price > df['sma_20'].iloc[-1] else "bearish"
                }
            }
            
        except Exception as e:
            logger.error("Error getting enhanced data for {sym}: {err}", sym=symbol, err=e)
            return {}