from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import pandas as pd
from loguru import logger

from domain.models import TradingEvent, StrategyPerformance, MarketPerformance
from domain.ports import MarketDataPort


class TradingMemory:
    """
    Learning and memory system for the trading bot.
    Stores all trading events, analyzes patterns, and provides insights.
    """
    
    def __init__(self, db_path: str = "data/trading_memory.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self._init_database()
        
    def _init_database(self):
        """Initialize the SQLite database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS trading_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    quantity REAL DEFAULT 0.0,
                    pnl REAL,
                    duration_minutes INTEGER,
                    market_conditions TEXT,  -- JSON
                    strategy_used TEXT,
                    outcome TEXT
                );
                
                CREATE TABLE IF NOT EXISTS strategy_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_name TEXT UNIQUE NOT NULL,
                    total_trades INTEGER DEFAULT 0,
                    winning_trades INTEGER DEFAULT 0,
                    losing_trades INTEGER DEFAULT 0,
                    total_pnl REAL DEFAULT 0.0,
                    win_rate REAL DEFAULT 0.0,
                    avg_win REAL DEFAULT 0.0,
                    avg_loss REAL DEFAULT 0.0,
                    profit_factor REAL DEFAULT 0.0,
                    sharpe_ratio REAL,
                    max_drawdown REAL DEFAULT 0.0,
                    last_updated INTEGER
                );
                
                CREATE TABLE IF NOT EXISTS market_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    price_change_24h REAL NOT NULL,
                    volume_24h REAL NOT NULL,
                    current_price REAL NOT NULL,
                    high_24h REAL NOT NULL,
                    low_24h REAL NOT NULL
                );
                
                CREATE INDEX IF NOT EXISTS idx_trading_events_symbol ON trading_events(symbol);
                CREATE INDEX IF NOT EXISTS idx_trading_events_timestamp ON trading_events(timestamp);
                CREATE INDEX IF NOT EXISTS idx_trading_events_strategy ON trading_events(strategy_used);
                CREATE INDEX IF NOT EXISTS idx_market_performance_symbol ON market_performance(symbol);
                CREATE INDEX IF NOT EXISTS idx_market_performance_timestamp ON market_performance(timestamp);
            """)
            
    def store_trading_event(self, event: TradingEvent) -> int:
        """Store a trading event in the database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO trading_events 
                (timestamp, symbol, action, reason, entry_price, exit_price, quantity, 
                 pnl, duration_minutes, market_conditions, strategy_used, outcome)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event.timestamp, event.symbol, event.action, event.reason, event.entry_price,
                event.exit_price, event.quantity, event.pnl, event.duration_minutes,
                json.dumps(event.market_conditions) if event.market_conditions else None,
                event.strategy_used, event.outcome or event.calculate_outcome()
            ))
            return cursor.lastrowid
            
    def update_trading_event(self, event_id: int, **updates):
        """Update a trading event (e.g., when trade is closed)"""
        if not updates:
            return
            
        set_clause = ", ".join(f"{key} = ?" for key in updates.keys())
        values = list(updates.values()) + [event_id]
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(f"""
                UPDATE trading_events 
                SET {set_clause}
                WHERE id = ?
            """, values)
            
    def store_market_performance(self, performance: MarketPerformance) -> int:
        """Store market performance data"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO market_performance 
                (timestamp, symbol, price_change_24h, volume_24h, current_price, high_24h, low_24h)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                performance.timestamp, performance.symbol, performance.price_change_24h,
                performance.volume_24h, performance.current_price, performance.high_24h, performance.low_24h
            ))
            return cursor.lastrowid
            
    def get_recent_events(self, symbol: Optional[str] = None, hours: int = 24) -> List[TradingEvent]:
        """Get recent trading events"""
        since_timestamp = int((datetime.now() - timedelta(hours=hours)).timestamp())
        
        query = """
            SELECT timestamp, symbol, action, reason, entry_price, exit_price, quantity,
                   pnl, duration_minutes, market_conditions, strategy_used, outcome
            FROM trading_events 
            WHERE timestamp >= ?
        """
        params = [since_timestamp]
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
            
        query += " ORDER BY timestamp DESC"
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            events = []
            for row in cursor.fetchall():
                market_conditions = json.loads(row[9]) if row[9] else None
                events.append(TradingEvent(
                    timestamp=row[0], symbol=row[1], action=row[2], reason=row[3],
                    entry_price=row[4], exit_price=row[5], quantity=row[6], pnl=row[7],
                    duration_minutes=row[8], market_conditions=market_conditions,
                    strategy_used=row[10], outcome=row[11]
                ))
            return events
            
    def get_strategy_performance(self, strategy_name: str) -> Optional[StrategyPerformance]:
        """Get performance metrics for a strategy"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT strategy_name, total_trades, winning_trades, losing_trades, total_pnl,
                       win_rate, avg_win, avg_loss, profit_factor, sharpe_ratio, max_drawdown, last_updated
                FROM strategy_performance 
                WHERE strategy_name = ?
            """, (strategy_name,))
            
            row = cursor.fetchone()
            if row:
                return StrategyPerformance(
                    strategy_name=row[0], total_trades=row[1], winning_trades=row[2],
                    losing_trades=row[3], total_pnl=row[4], win_rate=row[5], avg_win=row[6],
                    avg_loss=row[7], profit_factor=row[8], sharpe_ratio=row[9],
                    max_drawdown=row[10], last_updated=row[11]
                )
        return None
        
    def update_strategy_performance(self, strategy_name: str) -> StrategyPerformance:
        """Update and return strategy performance metrics"""
        # Get all trades for this strategy
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT timestamp, symbol, action, reason, entry_price, exit_price, quantity,
                       pnl, duration_minutes, market_conditions, strategy_used, outcome
                FROM trading_events 
                WHERE strategy_used = ? AND pnl IS NOT NULL
                ORDER BY timestamp
            """, (strategy_name,))
            
            trades = []
            for row in cursor.fetchall():
                market_conditions = json.loads(row[9]) if row[9] else None
                trades.append(TradingEvent(
                    timestamp=row[0], symbol=row[1], action=row[2], reason=row[3],
                    entry_price=row[4], exit_price=row[5], quantity=row[6], pnl=row[7],
                    duration_minutes=row[8], market_conditions=market_conditions,
                    strategy_used=row[10], outcome=row[11]
                ))
                
        # Calculate performance metrics
        perf = StrategyPerformance(strategy_name=strategy_name)
        perf.update_metrics(trades)
        
        # Store/update in database
        conn.execute("""
            INSERT OR REPLACE INTO strategy_performance 
            (strategy_name, total_trades, winning_trades, losing_trades, total_pnl,
             win_rate, avg_win, avg_loss, profit_factor, sharpe_ratio, max_drawdown, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            perf.strategy_name, perf.total_trades, perf.winning_trades, perf.losing_trades,
            perf.total_pnl, perf.win_rate, perf.avg_win, perf.avg_loss, perf.profit_factor,
            perf.sharpe_ratio, perf.max_drawdown, perf.last_updated
        ))
        
        return perf
        
    def get_top_performing_strategies(self, limit: int = 5) -> List[StrategyPerformance]:
        """Get top performing strategies by profit factor"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT strategy_name, total_trades, winning_trades, losing_trades, total_pnl,
                       win_rate, avg_win, avg_loss, profit_factor, sharpe_ratio, max_drawdown, last_updated
                FROM strategy_performance 
                WHERE total_trades >= 5
                ORDER BY profit_factor DESC, total_pnl DESC
                LIMIT ?
            """, (limit,))
            
            strategies = []
            for row in cursor.fetchall():
                strategies.append(StrategyPerformance(
                    strategy_name=row[0], total_trades=row[1], winning_trades=row[2],
                    losing_trades=row[3], total_pnl=row[4], win_rate=row[5], avg_win=row[6],
                    avg_loss=row[7], profit_factor=row[8], sharpe_ratio=row[9],
                    max_drawdown=row[10], last_updated=row[11]
                ))
            return strategies
            
    def analyze_market_patterns(self, symbol: str, hours: int = 168) -> Dict:
        """Analyze patterns in market data and trading outcomes"""
        since_timestamp = int((datetime.now() - timedelta(hours=hours)).timestamp())
        
        with sqlite3.connect(self.db_path) as conn:
            # Get market performance data
            df_market = pd.read_sql_query("""
                SELECT timestamp, price_change_24h, volume_24h, current_price
                FROM market_performance 
                WHERE symbol = ? AND timestamp >= ?
                ORDER BY timestamp
            """, conn, params=(symbol, since_timestamp))
            
            # Get trading events
            df_trades = pd.read_sql_query("""
                SELECT timestamp, action, pnl, market_conditions, outcome
                FROM trading_events 
                WHERE symbol = ? AND timestamp >= ?
                ORDER BY timestamp
            """, conn, params=(symbol, since_timestamp))
            
        analysis = {
            "symbol": symbol,
            "analysis_period_hours": hours,
            "market_stats": {},
            "trading_stats": {},
            "patterns": {}
        }
        
        if not df_market.empty:
            analysis["market_stats"] = {
                "avg_price_change_24h": df_market["price_change_24h"].mean(),
                "volatility": df_market["price_change_24h"].std(),
                "avg_volume": df_market["volume_24h"].mean(),
                "price_trend": "bullish" if df_market["price_change_24h"].mean() > 0 else "bearish"
            }
            
        if not df_trades.empty:
            winning_trades = df_trades[df_trades["outcome"] == "win"]
            analysis["trading_stats"] = {
                "total_trades": len(df_trades),
                "win_rate": len(winning_trades) / len(df_trades) if len(df_trades) > 0 else 0,
                "avg_pnl": df_trades["pnl"].mean() if "pnl" in df_trades else 0,
                "best_trade": df_trades["pnl"].max() if "pnl" in df_trades else 0,
                "worst_trade": df_trades["pnl"].min() if "pnl" in df_trades else 0
            }
            
        return analysis
        
    def get_learning_insights(self, symbol: Optional[str] = None) -> Dict:
        """Generate learning insights from historical data"""
        query_conditions = ""
        params = []
        
        if symbol:
            query_conditions = "WHERE symbol = ?"
            params.append(symbol)
            
        with sqlite3.connect(self.db_path) as conn:
            # Get strategy effectiveness
            strategy_effectiveness = pd.read_sql_query(f"""
                SELECT strategy_used, COUNT(*) as total_trades, 
                       AVG(CASE WHEN outcome = 'win' THEN 1.0 ELSE 0.0 END) as win_rate,
                       AVG(pnl) as avg_pnl
                FROM trading_events 
                {query_conditions}
                GROUP BY strategy_used
                HAVING total_trades >= 3
                ORDER BY win_rate DESC, avg_pnl DESC
            """, conn, params=params)
            
            # Get time-based patterns
            time_patterns = pd.read_sql_query(f"""
                SELECT strftime('%H', datetime(timestamp, 'unixepoch')) as hour,
                       COUNT(*) as trades,
                       AVG(CASE WHEN outcome = 'win' THEN 1.0 ELSE 0.0 END) as win_rate
                FROM trading_events 
                {query_conditions}
                GROUP BY hour
                HAVING trades >= 2
                ORDER BY win_rate DESC
            """, conn, params=params)
            
        insights = {
            "best_strategies": strategy_effectiveness.to_dict('records') if not strategy_effectiveness.empty else [],
            "best_trading_hours": time_patterns.to_dict('records') if not time_patterns.empty else [],
            "recommendations": []
        }
        
        # Generate recommendations
        if not strategy_effectiveness.empty:
            best_strategy = strategy_effectiveness.iloc[0]
            insights["recommendations"].append(
                f"Use {best_strategy['strategy_used']} strategy (win rate: {best_strategy['win_rate']:.2%})"
            )
            
        if not time_patterns.empty:
            best_hour = time_patterns.iloc[0]
            insights["recommendations"].append(
                f"Trade during hour {best_hour['hour']} for best results (win rate: {best_hour['win_rate']:.2%})"
            )
            
        return insights