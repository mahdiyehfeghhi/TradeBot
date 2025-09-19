from __future__ import annotations

import asyncio
import time
import re
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

import httpx
import pandas as pd
import numpy as np
from loguru import logger


@dataclass
class NewsItem:
    title: str
    content: str
    source: str
    url: str
    timestamp: int
    sentiment_score: float = 0.0  # -1 to 1 (negative to positive)
    symbols_mentioned: List[str] = None
    impact_score: float = 0.0  # 0 to 1 (low to high impact)
    
    def __post_init__(self):
        if self.symbols_mentioned is None:
            self.symbols_mentioned = []


@dataclass
class MarketSentiment:
    symbol: str
    overall_sentiment: float  # -1 to 1
    news_count: int
    positive_news: int
    negative_news: int
    key_events: List[str]
    sentiment_trend: str  # "improving", "declining", "stable"
    confidence: float  # 0 to 1


class NewsSentimentAnalyzer:
    """
    Analyzes news and social media sentiment for trading decisions
    """
    
    def __init__(self):
        self.crypto_keywords = {
            'bitcoin': ['BTC', 'bitcoin', 'btc'],
            'ethereum': ['ETH', 'ethereum', 'eth'],
            'binance': ['BNB', 'binance'],
            'cardano': ['ADA', 'cardano'],
            'solana': ['SOL', 'solana'],
            'polkadot': ['DOT', 'polkadot'],
            'chainlink': ['LINK', 'chainlink'],
            'dogecoin': ['DOGE', 'dogecoin', 'doge']
        }
        
        # Sentiment keywords (simple approach - could be enhanced with ML)
        self.positive_words = {
            'bullish', 'positive', 'growth', 'surge', 'rally', 'boom', 'breakout',
            'adoption', 'partnership', 'upgrade', 'innovation', 'breakthrough',
            'institutional', 'investment', 'buy', 'accumulate', 'hodl', 'moon',
            'pump', 'gains', 'profit', 'winner', 'success', 'launch', 'listing'
        }
        
        self.negative_words = {
            'bearish', 'negative', 'decline', 'crash', 'dump', 'sell', 'bear',
            'regulation', 'ban', 'hack', 'scam', 'fraud', 'bubble', 'overvalued',
            'correction', 'dip', 'loss', 'risk', 'concern', 'warning', 'fear',
            'panic', 'liquidation', 'bankruptcy', 'delisting', 'shutdown'
        }
        
        self.news_cache: List[NewsItem] = []
        self.sentiment_cache: Dict[str, MarketSentiment] = {}
        
    def extract_symbols(self, text: str) -> List[str]:
        """Extract cryptocurrency symbols from text"""
        text_lower = text.lower()
        symbols = []
        
        for symbol, keywords in self.crypto_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    symbols.append(symbol.upper())
                    break
        
        # Also look for patterns like $BTC, #BTC, BTC/USD
        crypto_pattern = r'[\$#]?([A-Z]{3,5})(?:[/-][A-Z]{3,4})?'
        matches = re.findall(crypto_pattern, text.upper())
        
        for match in matches:
            if match in ['BTC', 'ETH', 'ADA', 'SOL', 'DOT', 'LINK', 'DOGE', 'BNB']:
                symbols.append(match)
        
        return list(set(symbols))  # Remove duplicates
    
    def calculate_sentiment_score(self, text: str) -> float:
        """Calculate sentiment score from text (-1 to 1)"""
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        
        total_sentiment_words = positive_count + negative_count
        if total_sentiment_words == 0:
            return 0.0
        
        # Calculate net sentiment
        net_sentiment = (positive_count - negative_count) / total_sentiment_words
        
        # Apply intensity based on word frequency
        intensity = min(total_sentiment_words / len(words), 0.5) if words else 0
        
        return net_sentiment * (0.5 + intensity)
    
    def calculate_impact_score(self, news_item: NewsItem) -> float:
        """Calculate potential market impact of news (0 to 1)"""
        score = 0.0
        
        # Source credibility (higher score for known sources)
        credible_sources = {
            'coindesk', 'cointelegraph', 'bloomberg', 'reuters', 'binance',
            'coinbase', 'kraken', 'official', 'sec', 'fed', 'treasury'
        }
        
        source_lower = news_item.source.lower()
        if any(source in source_lower for source in credible_sources):
            score += 0.3
        
        # Title importance indicators
        title_lower = news_item.title.lower()
        important_keywords = {
            'regulation': 0.4, 'sec': 0.4, 'fed': 0.3, 'etf': 0.4,
            'institutional': 0.3, 'adoption': 0.3, 'partnership': 0.2,
            'hack': 0.5, 'ban': 0.5, 'crash': 0.4, 'surge': 0.3
        }
        
        for keyword, weight in important_keywords.items():
            if keyword in title_lower:
                score += weight
        
        # Number of symbols mentioned (broader impact)
        if len(news_item.symbols_mentioned) > 1:
            score += 0.1 * len(news_item.symbols_mentioned)
        
        # Sentiment strength
        score += abs(news_item.sentiment_score) * 0.2
        
        return min(score, 1.0)
    
    async def fetch_crypto_news(self, limit: int = 50) -> List[NewsItem]:
        """Fetch crypto news from various sources"""
        news_items = []
        
        try:
            # Example using a free crypto news API (you'd need to replace with actual API)
            # This is a mock implementation - in reality you'd use services like:
            # - CoinAPI, CryptoCompare, NewsAPI, etc.
            
            # Mock news data for demonstration
            mock_news = [
                {
                    "title": "Bitcoin breaks $50,000 as institutional adoption grows",
                    "content": "Bitcoin has surged past $50,000 following increased institutional investment...",
                    "source": "CoinDesk",
                    "url": "https://example.com/news1",
                    "published": int(time.time()) - 3600
                },
                {
                    "title": "Ethereum upgrade shows promising scalability improvements",
                    "content": "The latest Ethereum network upgrade demonstrates significant improvements...",
                    "source": "CoinTelegraph", 
                    "url": "https://example.com/news2",
                    "published": int(time.time()) - 7200
                },
                {
                    "title": "Regulatory concerns mount over cryptocurrency trading",
                    "content": "Government officials express concerns about cryptocurrency market volatility...",
                    "source": "Reuters",
                    "url": "https://example.com/news3", 
                    "published": int(time.time()) - 10800
                }
            ]
            
            for article in mock_news:
                symbols = self.extract_symbols(article["title"] + " " + article["content"])
                sentiment = self.calculate_sentiment_score(article["title"] + " " + article["content"])
                
                news_item = NewsItem(
                    title=article["title"],
                    content=article["content"],
                    source=article["source"],
                    url=article["url"],
                    timestamp=article["published"],
                    sentiment_score=sentiment,
                    symbols_mentioned=symbols
                )
                
                news_item.impact_score = self.calculate_impact_score(news_item)
                news_items.append(news_item)
            
            # Cache the news
            self.news_cache.extend(news_items)
            # Keep only recent news
            cutoff_time = int(time.time()) - 86400 * 7  # 7 days
            self.news_cache = [n for n in self.news_cache if n.timestamp > cutoff_time]
            
            logger.info(f"Fetched {len(news_items)} news items")
            
        except Exception as e:
            logger.error(f"Error fetching crypto news: {e}")
        
        return news_items
    
    def analyze_symbol_sentiment(self, symbol: str, hours_back: int = 24) -> MarketSentiment:
        """Analyze sentiment for a specific symbol"""
        cutoff_time = int(time.time()) - (hours_back * 3600)
        
        # Get relevant news
        relevant_news = [
            news for news in self.news_cache
            if news.timestamp > cutoff_time and 
            (symbol in news.symbols_mentioned or symbol.lower() in news.title.lower())
        ]
        
        if not relevant_news:
            return MarketSentiment(
                symbol=symbol,
                overall_sentiment=0.0,
                news_count=0,
                positive_news=0,
                negative_news=0,
                key_events=[],
                sentiment_trend="stable",
                confidence=0.0
            )
        
        # Calculate metrics
        sentiments = [news.sentiment_score for news in relevant_news]
        impacts = [news.impact_score for news in relevant_news]
        
        # Weight sentiment by impact
        weighted_sentiment = sum(s * i for s, i in zip(sentiments, impacts)) / sum(impacts) if impacts else 0
        
        positive_count = sum(1 for s in sentiments if s > 0.1)
        negative_count = sum(1 for s in sentiments if s < -0.1)
        
        # Key events (high impact news)
        key_events = [
            news.title for news in relevant_news
            if news.impact_score > 0.5
        ][:5]  # Top 5
        
        # Sentiment trend analysis
        if len(relevant_news) >= 4:
            recent_sentiment = np.mean([n.sentiment_score for n in relevant_news[:len(relevant_news)//2]])
            older_sentiment = np.mean([n.sentiment_score for n in relevant_news[len(relevant_news)//2:]])
            
            if recent_sentiment > older_sentiment + 0.1:
                trend = "improving"
            elif recent_sentiment < older_sentiment - 0.1:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "stable"
        
        # Confidence based on news volume and source quality
        confidence = min(len(relevant_news) / 10, 1.0)  # Max confidence with 10+ news items
        avg_impact = np.mean(impacts) if impacts else 0
        confidence *= (0.5 + avg_impact * 0.5)  # Boost confidence with high-impact news
        
        sentiment = MarketSentiment(
            symbol=symbol,
            overall_sentiment=weighted_sentiment,
            news_count=len(relevant_news),
            positive_news=positive_count,
            negative_news=negative_count,
            key_events=key_events,
            sentiment_trend=trend,
            confidence=confidence
        )
        
        # Cache the result
        self.sentiment_cache[symbol] = sentiment
        
        return sentiment
    
    def get_market_sentiment_signals(self, symbols: List[str]) -> Dict[str, Dict]:
        """Get trading signals based on sentiment analysis"""
        signals = {}
        
        for symbol in symbols:
            sentiment = self.analyze_symbol_sentiment(symbol)
            
            # Generate signal based on sentiment
            signal_strength = abs(sentiment.overall_sentiment) * sentiment.confidence
            
            if sentiment.overall_sentiment > 0.3 and sentiment.confidence > 0.5:
                signal = "bullish"
                strength = signal_strength
            elif sentiment.overall_sentiment < -0.3 and sentiment.confidence > 0.5:
                signal = "bearish" 
                strength = signal_strength
            else:
                signal = "neutral"
                strength = 0.0
            
            signals[symbol] = {
                "signal": signal,
                "strength": strength,
                "sentiment_score": sentiment.overall_sentiment,
                "confidence": sentiment.confidence,
                "news_count": sentiment.news_count,
                "trend": sentiment.sentiment_trend,
                "key_events": sentiment.key_events[:3]  # Top 3 events
            }
        
        return signals


class SocialSentimentTracker:
    """
    Track sentiment from social media platforms (Twitter, Reddit, etc.)
    """
    
    def __init__(self):
        self.platform_weights = {
            'twitter': 0.4,
            'reddit': 0.3,
            'telegram': 0.2,
            'discord': 0.1
        }
        
        self.social_data: List[Dict] = []
        
    async def fetch_twitter_sentiment(self, symbols: List[str]) -> Dict[str, float]:
        """Fetch Twitter sentiment (mock implementation)"""
        # In real implementation, you'd use Twitter API v2
        sentiment_scores = {}
        
        for symbol in symbols:
            # Mock sentiment based on symbol
            base_sentiment = hash(symbol) / 2**31  # Pseudo-random between -1 and 1
            # Add some time-based variation
            time_factor = (time.time() % 86400) / 86400  # 0 to 1 based on time of day
            final_sentiment = base_sentiment * 0.8 + (time_factor - 0.5) * 0.4
            
            sentiment_scores[symbol] = max(-1, min(1, final_sentiment))
        
        return sentiment_scores
    
    async def fetch_reddit_sentiment(self, symbols: List[str]) -> Dict[str, float]:
        """Fetch Reddit sentiment (mock implementation)"""
        # In real implementation, you'd use Reddit API (PRAW)
        sentiment_scores = {}
        
        for symbol in symbols:
            # Mock implementation
            base_sentiment = (hash(symbol + "reddit") / 2**31) * 0.8
            sentiment_scores[symbol] = max(-1, min(1, base_sentiment))
        
        return sentiment_scores
    
    async def get_combined_social_sentiment(self, symbols: List[str]) -> Dict[str, Dict]:
        """Get combined social media sentiment"""
        twitter_sentiment = await self.fetch_twitter_sentiment(symbols)
        reddit_sentiment = await self.fetch_reddit_sentiment(symbols)
        
        combined_sentiment = {}
        
        for symbol in symbols:
            twitter_score = twitter_sentiment.get(symbol, 0.0)
            reddit_score = reddit_sentiment.get(symbol, 0.0)
            
            # Weighted average
            combined_score = (
                twitter_score * self.platform_weights['twitter'] +
                reddit_score * self.platform_weights['reddit']
            ) / (self.platform_weights['twitter'] + self.platform_weights['reddit'])
            
            # Determine sentiment strength and direction
            if combined_score > 0.2:
                sentiment_label = "bullish"
            elif combined_score < -0.2:
                sentiment_label = "bearish"
            else:
                sentiment_label = "neutral"
            
            combined_sentiment[symbol] = {
                "overall_score": combined_score,
                "sentiment": sentiment_label,
                "twitter_score": twitter_score,
                "reddit_score": reddit_score,
                "strength": abs(combined_score)
            }
        
        return combined_sentiment


class IntegratedSentimentEngine:
    """
    Combines news and social sentiment for comprehensive market sentiment analysis
    """
    
    def __init__(self):
        self.news_analyzer = NewsSentimentAnalyzer()
        self.social_tracker = SocialSentimentTracker()
        self.sentiment_history: Dict[str, List[Dict]] = {}
        
    async def get_comprehensive_sentiment(self, symbols: List[str]) -> Dict[str, Dict]:
        """Get comprehensive sentiment analysis combining news and social media"""
        
        # Fetch news sentiment
        await self.news_analyzer.fetch_crypto_news()
        news_signals = self.news_analyzer.get_market_sentiment_signals(symbols)
        
        # Fetch social sentiment
        social_sentiment = await self.social_tracker.get_combined_social_sentiment(symbols)
        
        # Combine both sources
        comprehensive_sentiment = {}
        
        for symbol in symbols:
            news_data = news_signals.get(symbol, {})
            social_data = social_sentiment.get(symbol, {})
            
            # Weight news vs social (news typically more reliable for crypto)
            news_weight = 0.7
            social_weight = 0.3
            
            news_score = news_data.get("sentiment_score", 0.0)
            social_score = social_data.get("overall_score", 0.0)
            
            combined_score = news_score * news_weight + social_score * social_weight
            
            # Determine overall signal
            confidence = news_data.get("confidence", 0.0) * 0.7 + social_data.get("strength", 0.0) * 0.3
            
            if combined_score > 0.2 and confidence > 0.4:
                overall_signal = "bullish"
            elif combined_score < -0.2 and confidence > 0.4:
                overall_signal = "bearish"
            else:
                overall_signal = "neutral"
            
            comprehensive_sentiment[symbol] = {
                "overall_signal": overall_signal,
                "combined_score": combined_score,
                "confidence": confidence,
                "news_sentiment": news_data,
                "social_sentiment": social_data,
                "recommendation": self._generate_trading_recommendation(
                    combined_score, confidence, news_data, social_data
                )
            }
            
            # Store in history
            if symbol not in self.sentiment_history:
                self.sentiment_history[symbol] = []
            
            self.sentiment_history[symbol].append({
                "timestamp": int(time.time()),
                "score": combined_score,
                "confidence": confidence,
                "signal": overall_signal
            })
            
            # Keep only recent history
            if len(self.sentiment_history[symbol]) > 100:
                self.sentiment_history[symbol] = self.sentiment_history[symbol][-50:]
        
        return comprehensive_sentiment
    
    def _generate_trading_recommendation(self, score: float, confidence: float, 
                                       news_data: Dict, social_data: Dict) -> Dict:
        """Generate specific trading recommendations based on sentiment"""
        
        action = "hold"
        strength = 0.0
        reasoning = []
        
        # Strong bullish signals
        if score > 0.4 and confidence > 0.6:
            action = "buy"
            strength = min(score * confidence, 1.0)
            reasoning.append(f"Strong positive sentiment (score: {score:.2f})")
            
            if news_data.get("trend") == "improving":
                reasoning.append("News sentiment improving")
                strength *= 1.1
            
            if news_data.get("news_count", 0) > 5:
                reasoning.append("High news volume supports signal")
                
        # Strong bearish signals
        elif score < -0.4 and confidence > 0.6:
            action = "sell"
            strength = min(abs(score) * confidence, 1.0)
            reasoning.append(f"Strong negative sentiment (score: {score:.2f})")
            
            if news_data.get("trend") == "declining":
                reasoning.append("News sentiment declining")
                strength *= 1.1
        
        # Moderate signals
        elif abs(score) > 0.2 and confidence > 0.4:
            if score > 0:
                action = "weak_buy"
                reasoning.append("Moderate positive sentiment")
            else:
                action = "weak_sell"
                reasoning.append("Moderate negative sentiment")
            
            strength = abs(score) * confidence * 0.5
        
        # Add specific news insights
        key_events = news_data.get("key_events", [])
        if key_events:
            reasoning.append(f"Key events: {', '.join(key_events[:2])}")
        
        return {
            "action": action,
            "strength": strength,
            "reasoning": reasoning,
            "risk_level": "high" if confidence < 0.3 else "medium" if confidence < 0.6 else "low"
        }