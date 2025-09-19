from __future__ import annotations

import asyncio
import time
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

from loguru import logger
import httpx
import pandas as pd
import numpy as np

from domain.models import Candle, TradeDecision, MarketPerformance
from domain.ports import Strategy, MarketDataPort
from application.sentiment import IntegratedSentimentEngine, NewsItem
from application.memory import TradingMemory


@dataclass
class NewsEvent:
    timestamp: int
    symbol: str
    event_type: str  # "announcement", "partnership", "regulation", "technical", "market"
    impact_level: str  # "low", "medium", "high", "critical"
    sentiment: float  # -1 to 1
    confidence: float  # 0 to 1
    title: str
    summary: str
    source: str
    keywords: List[str]
    expected_price_impact: float  # Expected % price change
    time_horizon: int  # Expected impact duration in minutes


class NewsEventClassifier:
    """
    Classifies news events and estimates their market impact
    """
    
    def __init__(self):
        # Event type keywords
        self.event_keywords = {
            "regulation": {
                "keywords": ["regulation", "sec", "fed", "government", "law", "legal", "compliance", "ban", "approval"],
                "impact_multiplier": 1.5,
                "time_horizon": 720  # 12 hours
            },
            "partnership": {
                "keywords": ["partnership", "collaboration", "integration", "adoption", "institutional"],
                "impact_multiplier": 1.2,
                "time_horizon": 360  # 6 hours
            },
            "technical": {
                "keywords": ["upgrade", "fork", "update", "launch", "release", "mainnet", "testnet"],
                "impact_multiplier": 1.1,
                "time_horizon": 480  # 8 hours
            },
            "market": {
                "keywords": ["listing", "exchange", "trading", "volume", "price", "surge", "crash"],
                "impact_multiplier": 1.3,
                "time_horizon": 240  # 4 hours
            },
            "announcement": {
                "keywords": ["announcement", "news", "statement", "report", "earnings"],
                "impact_multiplier": 1.0,
                "time_horizon": 180  # 3 hours
            }
        }
        
        # Impact level indicators
        self.impact_indicators = {
            "critical": {
                "keywords": ["hack", "exploit", "crash", "ban", "shutdown", "bankruptcy", "emergency"],
                "min_sentiment": 0.7
            },
            "high": {
                "keywords": ["major", "significant", "breakthrough", "milestone", "record", "massive"],
                "min_sentiment": 0.5
            },
            "medium": {
                "keywords": ["important", "notable", "substantial", "considerable"],
                "min_sentiment": 0.3
            }
        }
    
    def classify_news(self, news_item: NewsItem) -> NewsEvent:
        """Classify a news item into a structured news event"""
        
        text = (news_item.title + " " + news_item.content).lower()
        
        # Determine event type
        event_type = "announcement"  # default
        impact_multiplier = 1.0
        time_horizon = 180
        
        for event_type_name, config in self.event_keywords.items():
            if any(keyword in text for keyword in config["keywords"]):
                event_type = event_type_name
                impact_multiplier = config["impact_multiplier"]
                time_horizon = config["time_horizon"]
                break
        
        # Determine impact level
        impact_level = "low"  # default
        
        for level, config in self.impact_indicators.items():
            if (any(keyword in text for keyword in config["keywords"]) and
                abs(news_item.sentiment_score) >= config["min_sentiment"]):
                impact_level = level
                break
        
        # Estimate price impact
        base_impact = abs(news_item.sentiment_score) * news_item.impact_score
        price_impact = base_impact * impact_multiplier
        
        # Apply impact level multiplier
        level_multipliers = {"low": 1.0, "medium": 1.5, "high": 2.0, "critical": 3.0}
        price_impact *= level_multipliers[impact_level]
        
        # Extract keywords
        keywords = []
        for keyword_list in self.event_keywords.values():
            keywords.extend([kw for kw in keyword_list["keywords"] if kw in text])
        
        # Determine primary symbol
        symbol = news_item.symbols_mentioned[0] if news_item.symbols_mentioned else "UNKNOWN"
        
        return NewsEvent(
            timestamp=news_item.timestamp,
            symbol=symbol,
            event_type=event_type,
            impact_level=impact_level,
            sentiment=news_item.sentiment_score,
            confidence=news_item.impact_score,
            title=news_item.title,
            summary=news_item.content[:200] + "..." if len(news_item.content) > 200 else news_item.content,
            source=news_item.source,
            keywords=list(set(keywords)),
            expected_price_impact=price_impact,
            time_horizon=time_horizon
        )


class NewsBasedTrader:
    """
    Automated news-based trading system that reacts to market-moving events
    """
    
    def __init__(self, market: MarketDataPort, memory: TradingMemory):
        self.market = market
        self.memory = memory
        self.sentiment_engine = IntegratedSentimentEngine()
        self.classifier = NewsEventClassifier()
        self.active_positions: Dict[str, Dict] = {}
        self.processed_news: set = set()
        self.trading_enabled = True
        
        # Trading parameters
        self.min_impact_threshold = 0.3  # Minimum impact to trigger trade
        self.max_position_size = 0.1     # Max 10% of portfolio per news trade
        self.stop_loss_pct = 0.05        # 5% stop loss
        self.take_profit_multiplier = 2.0 # 2:1 reward:risk ratio
        
    async def process_news_events(self, symbols: List[str]) -> List[NewsEvent]:
        """Process recent news and identify trading opportunities"""
        
        # Fetch latest news
        await self.sentiment_engine.news_analyzer.fetch_crypto_news(limit=100)
        recent_news = [
            news for news in self.sentiment_engine.news_analyzer.news_cache
            if news.timestamp > int(time.time()) - 3600  # Last hour
        ]
        
        # Classify news events
        events = []
        for news_item in recent_news:
            # Skip if already processed
            news_id = f"{news_item.title}_{news_item.timestamp}"
            if news_id in self.processed_news:
                continue
            
            # Filter for relevant symbols
            if any(symbol in news_item.symbols_mentioned for symbol in symbols):
                event = self.classifier.classify_news(news_item)
                events.append(event)
                self.processed_news.add(news_id)
        
        # Clean old processed news
        if len(self.processed_news) > 1000:
            # Keep only recent IDs (rough cleanup)
            self.processed_news = set(list(self.processed_news)[-500:])
        
        return events
    
    def evaluate_trading_opportunity(self, event: NewsEvent, current_price: float) -> Optional[TradeDecision]:
        """Evaluate if a news event presents a trading opportunity"""
        
        if not self.trading_enabled:
            return None
        
        # Check minimum impact threshold
        if abs(event.expected_price_impact) < self.min_impact_threshold:
            return None
        
        # Skip if already have position in this symbol
        if event.symbol in self.active_positions:
            return None
        
        # Determine trade direction
        if event.sentiment > 0 and event.expected_price_impact > self.min_impact_threshold:
            action = "buy"
            expected_price = current_price * (1 + event.expected_price_impact)
            stop_loss = current_price * (1 - self.stop_loss_pct)
            take_profit = current_price * (1 + self.stop_loss_pct * self.take_profit_multiplier)
            
        elif event.sentiment < 0 and event.expected_price_impact > self.min_impact_threshold:
            action = "sell"
            expected_price = current_price * (1 - event.expected_price_impact)
            stop_loss = current_price * (1 + self.stop_loss_pct)
            take_profit = current_price * (1 - self.stop_loss_pct * self.take_profit_multiplier)
            
        else:
            return None
        
        # Calculate position size based on confidence and impact
        base_size = event.confidence * min(event.expected_price_impact, 0.1)  # Cap at 10%
        size_factor = min(base_size, self.max_position_size)
        
        reason = (f"news-{action}: {event.event_type} {event.impact_level} "
                 f"impact={event.expected_price_impact:.2%} conf={event.confidence:.2f}")
        
        return TradeDecision(
            action=action,
            reason=reason,
            size_quote=0,  # Will be calculated by risk manager
            stop_loss=stop_loss,
            take_profit=take_profit
        )
    
    def update_active_positions(self, symbol: str, current_price: float):
        """Update and manage active news-based positions"""
        if symbol not in self.active_positions:
            return
        
        position = self.active_positions[symbol]
        entry_time = position["entry_time"]
        time_horizon = position["time_horizon"]
        current_time = int(time.time())
        
        # Check if position should be closed based on time horizon
        if current_time - entry_time > time_horizon * 60:  # time_horizon is in minutes
            logger.info(f"Closing news position for {symbol} - time horizon exceeded")
            del self.active_positions[symbol]
    
    def add_position(self, symbol: str, event: NewsEvent, entry_price: float):
        """Track a new news-based position"""
        self.active_positions[symbol] = {
            "entry_time": int(time.time()),
            "entry_price": entry_price,
            "event_type": event.event_type,
            "impact_level": event.impact_level,
            "time_horizon": event.time_horizon,
            "expected_impact": event.expected_price_impact
        }
        
        logger.info(f"Added news position for {symbol}: {event.event_type} {event.impact_level}")


class NewsReactionStrategy(Strategy):
    """
    Strategy that reacts to news events and market sentiment
    """
    
    def __init__(self, symbol: str, market: MarketDataPort, memory: TradingMemory,
                 reaction_speed: str = "fast"):  # "fast", "medium", "slow"
        self.symbol = symbol
        self.market = market
        self.memory = memory
        self.news_trader = NewsBasedTrader(market, memory)
        self.reaction_speed = reaction_speed
        
        # Reaction speed settings
        self.speed_configs = {
            "fast": {"lookback_minutes": 15, "min_impact": 0.2, "position_hold_time": 60},
            "medium": {"lookback_minutes": 30, "min_impact": 0.3, "position_hold_time": 120},
            "slow": {"lookback_minutes": 60, "min_impact": 0.4, "position_hold_time": 240}
        }
        
        self.config = self.speed_configs[reaction_speed]
        self.last_news_check = 0
        self.news_check_interval = 300  # Check news every 5 minutes
        
    def on_candles(self, candles: List[Candle], price: float) -> TradeDecision:
        current_time = int(time.time())
        
        # Check for new news events periodically
        if current_time - self.last_news_check > self.news_check_interval:
            asyncio.create_task(self._process_news_async(price))
            self.last_news_check = current_time
        
        # Update active positions
        self.news_trader.update_active_positions(self.symbol, price)
        
        # Default technical analysis fallback
        return self._technical_fallback(candles, price)
    
    async def _process_news_async(self, current_price: float):
        """Process news events asynchronously"""
        try:
            events = await self.news_trader.process_news_events([self.symbol])
            
            for event in events:
                if event.symbol == self.symbol:
                    # Check if event is within our reaction time window
                    event_age = int(time.time()) - event.timestamp
                    if event_age <= self.config["lookback_minutes"] * 60:
                        
                        # Evaluate trading opportunity
                        decision = self.news_trader.evaluate_trading_opportunity(event, current_price)
                        if decision and abs(event.expected_price_impact) >= self.config["min_impact"]:
                            
                            # Log the opportunity (in real implementation, would execute trade)
                            logger.info(f"News trading opportunity: {decision.action} {self.symbol} - {decision.reason}")
                            
                            # Track position
                            if decision.action in ["buy", "sell"]:
                                self.news_trader.add_position(self.symbol, event, current_price)
                                
        except Exception as e:
            logger.error(f"Error processing news events: {e}")
    
    def _technical_fallback(self, candles: List[Candle], price: float) -> TradeDecision:
        """Fallback technical analysis when no news signals"""
        candle_list = list(candles)
        
        if len(candle_list) < 20:
            return TradeDecision(action="hold", reason="insufficient-data", size_quote=0)
        
        # Simple momentum check
        recent_prices = [c.close for c in candle_list[-10:]]
        price_momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        
        # Volume analysis
        recent_volumes = [c.volume for c in candle_list[-10:]]
        avg_volume = np.mean(recent_volumes)
        current_volume = candle_list[-1].volume
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # Combined signal
        if price_momentum > 0.02 and volume_ratio > 1.5:  # 2% price increase with high volume
            return TradeDecision(
                action="buy",
                reason=f"momentum-volume: {price_momentum:.2%} vol_ratio={volume_ratio:.1f}",
                size_quote=0,
                stop_loss=price * 0.95
            )
        elif price_momentum < -0.02 and volume_ratio > 1.5:  # 2% price decrease with high volume
            return TradeDecision(
                action="sell",
                reason=f"momentum-volume: {price_momentum:.2%} vol_ratio={volume_ratio:.1f}",
                size_quote=0,
                stop_loss=price * 1.05
            )
        
        return TradeDecision(action="hold", reason="no-clear-signal", size_quote=0)


class SentimentMomentumStrategy(Strategy):
    """
    Strategy that combines sentiment analysis with price momentum
    """
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.sentiment_engine = IntegratedSentimentEngine()
        self.last_sentiment_update = 0
        self.sentiment_cache = {}
        self.sentiment_history = []
        
    def on_candles(self, candles: List[Candle], price: float) -> TradeDecision:
        current_time = int(time.time())
        candle_list = list(candles)
        
        if len(candle_list) < 20:
            return TradeDecision(action="hold", reason="insufficient-data", size_quote=0)
        
        # Update sentiment periodically
        if current_time - self.last_sentiment_update > 900:  # 15 minutes
            asyncio.create_task(self._update_sentiment_async())
            self.last_sentiment_update = current_time
        
        # Get current sentiment
        sentiment_data = self.sentiment_cache.get(self.symbol, {})
        sentiment_score = sentiment_data.get("combined_score", 0.0)
        sentiment_confidence = sentiment_data.get("confidence", 0.0)
        
        # Calculate price momentum
        recent_prices = [c.close for c in candle_list[-10:]]
        short_momentum = (recent_prices[-1] - recent_prices[-5]) / recent_prices[-5]
        long_momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        
        # Calculate volume momentum
        recent_volumes = [c.volume for c in candle_list[-5:]]
        volume_trend = (recent_volumes[-1] - recent_volumes[0]) / recent_volumes[0] if recent_volumes[0] > 0 else 0
        
        # Sentiment momentum (change in sentiment)
        sentiment_momentum = 0.0
        if len(self.sentiment_history) >= 2:
            sentiment_momentum = self.sentiment_history[-1] - self.sentiment_history[-2]
        
        # Combined signal calculation
        momentum_score = (short_momentum * 0.4 + long_momentum * 0.3 + volume_trend * 0.3)
        sentiment_adjusted_score = sentiment_score * sentiment_confidence
        
        # Combine momentum and sentiment
        combined_score = momentum_score * 0.6 + sentiment_adjusted_score * 0.4
        
        # Add sentiment momentum boost
        if sentiment_momentum > 0.1:  # Improving sentiment
            combined_score *= 1.2
        elif sentiment_momentum < -0.1:  # Declining sentiment
            combined_score *= 0.8
        
        # Generate decision
        if combined_score > 0.3 and sentiment_confidence > 0.4:
            return TradeDecision(
                action="buy",
                reason=f"sentiment-momentum: combined={combined_score:.2f} sent={sentiment_score:.2f} mom={momentum_score:.2f}",
                size_quote=0,
                stop_loss=price * 0.96,
                take_profit=price * 1.08
            )
        elif combined_score < -0.3 and sentiment_confidence > 0.4:
            return TradeDecision(
                action="sell",
                reason=f"sentiment-momentum: combined={combined_score:.2f} sent={sentiment_score:.2f} mom={momentum_score:.2f}",
                size_quote=0,
                stop_loss=price * 1.04,
                take_profit=price * 0.92
            )
        
        return TradeDecision(
            action="hold",
            reason=f"neutral: combined={combined_score:.2f} conf={sentiment_confidence:.2f}",
            size_quote=0
        )
    
    async def _update_sentiment_async(self):
        """Update sentiment data asynchronously"""
        try:
            sentiment_data = await self.sentiment_engine.get_comprehensive_sentiment([self.symbol])
            self.sentiment_cache.update(sentiment_data)
            
            # Store sentiment history
            if self.symbol in sentiment_data:
                sentiment_score = sentiment_data[self.symbol].get("combined_score", 0.0)
                self.sentiment_history.append(sentiment_score)
                
                # Keep only recent history
                if len(self.sentiment_history) > 20:
                    self.sentiment_history = self.sentiment_history[-10:]
                    
        except Exception as e:
            logger.error(f"Error updating sentiment: {e}")


class MarketRegimeNewsStrategy(Strategy):
    """
    Advanced strategy that adapts news trading based on market regime
    """
    
    def __init__(self, symbol: str, market: MarketDataPort, memory: TradingMemory):
        self.symbol = symbol
        self.market = market
        self.memory = memory
        self.news_trader = NewsBasedTrader(market, memory)
        self.current_regime = "normal"
        self.regime_history = []
        
        # Regime-specific parameters
        self.regime_configs = {
            "bull_market": {"news_weight": 0.7, "momentum_weight": 0.3, "min_impact": 0.2},
            "bear_market": {"news_weight": 0.8, "momentum_weight": 0.2, "min_impact": 0.3},
            "volatile": {"news_weight": 0.5, "momentum_weight": 0.5, "min_impact": 0.4},
            "normal": {"news_weight": 0.6, "momentum_weight": 0.4, "min_impact": 0.3}
        }
    
    def _detect_market_regime(self, candles: List[Candle]) -> str:
        """Detect current market regime"""
        if len(candles) < 50:
            return "normal"
        
        prices = np.array([c.close for c in candles])
        volumes = np.array([c.volume for c in candles])
        
        # Calculate metrics
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns[-20:])  # 20-period volatility
        trend = (prices[-1] - prices[-30]) / prices[-30]  # 30-period trend
        volume_trend = (np.mean(volumes[-10:]) - np.mean(volumes[-30:-10])) / np.mean(volumes[-30:-10])
        
        # Regime classification
        if volatility > 0.08:  # High volatility
            return "volatile"
        elif trend > 0.15 and volume_trend > 0.2:  # Strong uptrend with volume
            return "bull_market"
        elif trend < -0.15:  # Strong downtrend
            return "bear_market"
        else:
            return "normal"
    
    def on_candles(self, candles: List[Candle], price: float) -> TradeDecision:
        candle_list = list(candles)
        
        # Detect market regime
        self.current_regime = self._detect_market_regime(candle_list)
        self.regime_history.append(self.current_regime)
        
        # Keep regime history manageable
        if len(self.regime_history) > 100:
            self.regime_history = self.regime_history[-50:]
        
        # Get regime-specific config
        config = self.regime_configs[self.current_regime]
        
        # Update news trader parameters based on regime
        self.news_trader.min_impact_threshold = config["min_impact"]
        
        # Process news with regime awareness
        current_time = int(time.time())
        
        # In volatile markets, react faster to news
        if self.current_regime == "volatile":
            self.news_trader.stop_loss_pct = 0.03  # Tighter stops
            self.news_trader.take_profit_multiplier = 1.5  # Lower targets
        else:
            self.news_trader.stop_loss_pct = 0.05  # Normal stops
            self.news_trader.take_profit_multiplier = 2.0  # Normal targets
        
        # Generate base technical signal
        base_signal = self._generate_base_signal(candle_list, price)
        
        # Adjust signal based on regime
        return self._adjust_signal_for_regime(base_signal, config)
    
    def _generate_base_signal(self, candles: List[Candle], price: float) -> TradeDecision:
        """Generate base trading signal"""
        if len(candles) < 20:
            return TradeDecision(action="hold", reason="insufficient-data", size_quote=0)
        
        # Simple momentum and volume analysis
        prices = [c.close for c in candles[-10:]]
        volumes = [c.volume for c in candles[-10:]]
        
        price_change = (prices[-1] - prices[0]) / prices[0]
        volume_ratio = volumes[-1] / np.mean(volumes[:-1]) if len(volumes) > 1 else 1
        
        if price_change > 0.02 and volume_ratio > 1.5:
            return TradeDecision(
                action="buy",
                reason=f"base-momentum: price_change={price_change:.2%} vol_ratio={volume_ratio:.1f}",
                size_quote=0,
                stop_loss=price * 0.95
            )
        elif price_change < -0.02 and volume_ratio > 1.5:
            return TradeDecision(
                action="sell",
                reason=f"base-momentum: price_change={price_change:.2%} vol_ratio={volume_ratio:.1f}",
                size_quote=0,
                stop_loss=price * 1.05
            )
        
        return TradeDecision(action="hold", reason="no-momentum", size_quote=0)
    
    def _adjust_signal_for_regime(self, signal: TradeDecision, config: Dict) -> TradeDecision:
        """Adjust trading signal based on market regime"""
        
        # Modify reason to include regime
        original_reason = signal.reason
        signal.reason = f"regime-{self.current_regime}: {original_reason}"
        
        # In bear markets, be more conservative
        if self.current_regime == "bear_market":
            if signal.action == "buy":
                # Require stronger signals in bear market
                if "momentum" not in original_reason:
                    signal.action = "hold"
                    signal.reason = f"bear-market-caution: {original_reason}"
        
        # In volatile markets, use smaller positions
        elif self.current_regime == "volatile":
            # Signals remain the same but risk management handles position sizing
            pass
        
        # In bull markets, be more aggressive
        elif self.current_regime == "bull_market":
            if signal.action == "hold" and "momentum" in original_reason:
                # Convert some holds to weak buys in bull markets
                signal.action = "buy"
                signal.reason = f"bull-market-bias: {original_reason}"
        
        return signal