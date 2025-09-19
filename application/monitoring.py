from __future__ import annotations

import asyncio
import time
import smtplib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import numpy as np
from loguru import logger

from domain.models import TradingEvent, StrategyPerformance, MarketPerformance
from application.memory import TradingMemory


@dataclass
class Alert:
    id: str
    type: str  # "risk", "performance", "market", "system"
    severity: str  # "low", "medium", "high", "critical"
    title: str
    message: str
    timestamp: int
    symbol: Optional[str] = None
    strategy: Optional[str] = None
    data: Optional[Dict] = None


@dataclass
class MonitoringConfig:
    max_drawdown_alert: float = 8.0  # Alert at 8% drawdown
    min_win_rate_alert: float = 0.3   # Alert if win rate drops below 30%
    max_daily_loss_alert: float = 5.0  # Alert at 5% daily loss
    volatility_spike_threshold: float = 0.15  # Alert on >15% volatility
    volume_drop_threshold: float = 0.5  # Alert if volume drops >50%
    
    # Email settings (optional)
    email_enabled: bool = False
    smtp_server: str = ""
    smtp_port: int = 587
    email_user: str = ""
    email_password: str = ""
    alert_emails: List[str] = None
    
    # Telegram settings (optional)
    telegram_enabled: bool = False
    telegram_token: str = ""
    telegram_chat_id: str = ""


class AlertManager:
    """
    Manages alerts, notifications, and escalation procedures
    """
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.active_alerts: List[Alert] = []
        self.alert_history: List[Alert] = []
        self.alert_callbacks: List[Callable] = []
        self.alert_counter = 0
        
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add a callback function to handle alerts"""
        self.alert_callbacks.append(callback)
    
    def create_alert(self, alert_type: str, severity: str, title: str, 
                    message: str, symbol: str = None, strategy: str = None, 
                    data: Dict = None) -> Alert:
        """Create a new alert"""
        self.alert_counter += 1
        alert = Alert(
            id=f"alert_{self.alert_counter}_{int(time.time())}",
            type=alert_type,
            severity=severity,
            title=title,
            message=message,
            timestamp=int(time.time()),
            symbol=symbol,
            strategy=strategy,
            data=data or {}
        )
        
        self.active_alerts.append(alert)
        self.alert_history.append(alert)
        
        # Keep only recent history
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-500:]
        
        # Trigger callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
        
        # Send notifications
        asyncio.create_task(self._send_notifications(alert))
        
        logger.warning(f"Alert created: {alert.severity} - {alert.title}: {alert.message}")
        return alert
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Mark an alert as resolved"""
        for i, alert in enumerate(self.active_alerts):
            if alert.id == alert_id:
                self.active_alerts.pop(i)
                logger.info(f"Alert resolved: {alert.title}")
                return True
        return False
    
    def get_active_alerts(self, severity: str = None) -> List[Alert]:
        """Get active alerts, optionally filtered by severity"""
        if severity:
            return [a for a in self.active_alerts if a.severity == severity]
        return self.active_alerts.copy()
    
    async def _send_notifications(self, alert: Alert):
        """Send notifications for the alert"""
        try:
            if self.config.email_enabled and alert.severity in ["high", "critical"]:
                await self._send_email_alert(alert)
            
            if self.config.telegram_enabled and alert.severity in ["medium", "high", "critical"]:
                await self._send_telegram_alert(alert)
                
        except Exception as e:
            logger.error(f"Failed to send alert notifications: {e}")
    
    async def _send_email_alert(self, alert: Alert):
        """Send email alert"""
        if not self.config.alert_emails:
            return
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config.email_user
            msg['To'] = ', '.join(self.config.alert_emails)
            msg['Subject'] = f"TradeBot Alert: {alert.severity.upper()} - {alert.title}"
            
            body = f"""
Alert Details:
- Type: {alert.type}
- Severity: {alert.severity}
- Title: {alert.title}
- Message: {alert.message}
- Symbol: {alert.symbol or 'N/A'}
- Strategy: {alert.strategy or 'N/A'}
- Time: {datetime.fromtimestamp(alert.timestamp)}

Data: {alert.data}
"""
            
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port) as server:
                server.starttls()
                server.login(self.config.email_user, self.config.email_password)
                server.send_message(msg)
                
            logger.info(f"Email alert sent for: {alert.title}")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    async def _send_telegram_alert(self, alert: Alert):
        """Send Telegram alert"""
        try:
            import httpx
            
            message = f"""
🚨 TradeBot Alert: {alert.severity.upper()}
📊 {alert.title}
💬 {alert.message}
🏷️ Symbol: {alert.symbol or 'N/A'}
📈 Strategy: {alert.strategy or 'N/A'}
⏰ {datetime.fromtimestamp(alert.timestamp).strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            url = f"https://api.telegram.org/bot{self.config.telegram_token}/sendMessage"
            data = {
                "chat_id": self.config.telegram_chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(url, data=data)
                if response.status_code == 200:
                    logger.info(f"Telegram alert sent for: {alert.title}")
                else:
                    logger.error(f"Failed to send Telegram alert: {response.text}")
                    
        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")


class PerformanceMonitor:
    """
    Monitors trading performance and detects anomalies
    """
    
    def __init__(self, memory: TradingMemory, alert_manager: AlertManager):
        self.memory = memory
        self.alert_manager = alert_manager
        self.performance_history: List[Dict] = []
        self.last_equity = 0.0
        self.peak_equity = 0.0
        self.daily_start_equity = 0.0
        self.daily_start_time = 0
        
    def update_equity(self, current_equity: float):
        """Update equity tracking and check for alerts"""
        current_time = int(time.time())
        
        # Initialize daily tracking
        if current_time - self.daily_start_time > 86400:  # New day
            self.daily_start_equity = current_equity
            self.daily_start_time = current_time
        
        # Update peak equity
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        # Calculate metrics
        drawdown = (self.peak_equity - current_equity) / self.peak_equity * 100 if self.peak_equity > 0 else 0
        daily_change = (current_equity - self.daily_start_equity) / self.daily_start_equity * 100 if self.daily_start_equity > 0 else 0
        
        # Store performance data
        self.performance_history.append({
            "timestamp": current_time,
            "equity": current_equity,
            "drawdown": drawdown,
            "daily_change": daily_change
        })
        
        # Keep only recent data
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-500:]
        
        # Check for alerts
        self._check_performance_alerts(current_equity, drawdown, daily_change)
        
        self.last_equity = current_equity
    
    def _check_performance_alerts(self, equity: float, drawdown: float, daily_change: float):
        """Check for performance-related alerts"""
        config = self.alert_manager.config
        
        # Drawdown alert
        if drawdown > config.max_drawdown_alert:
            self.alert_manager.create_alert(
                "risk", "high", "Excessive Drawdown",
                f"Current drawdown: {drawdown:.1f}% (limit: {config.max_drawdown_alert:.1f}%)",
                data={"drawdown": drawdown, "equity": equity}
            )
        
        # Daily loss alert
        if daily_change < -config.max_daily_loss_alert:
            self.alert_manager.create_alert(
                "risk", "medium", "Daily Loss Limit",
                f"Daily loss: {daily_change:.1f}% (limit: {config.max_daily_loss_alert:.1f}%)",
                data={"daily_change": daily_change, "equity": equity}
            )
    
    def analyze_strategy_performance(self, strategy_name: str) -> Dict:
        """Analyze individual strategy performance"""
        recent_trades = self.memory.get_recent_trades_by_strategy(strategy_name, days=30)
        
        if not recent_trades:
            return {"status": "insufficient_data"}
        
        # Calculate metrics
        wins = sum(1 for trade in recent_trades if trade.pnl and trade.pnl > 0)
        total = len(recent_trades)
        win_rate = wins / total if total > 0 else 0
        
        total_pnl = sum(trade.pnl for trade in recent_trades if trade.pnl)
        avg_trade = total_pnl / total if total > 0 else 0
        
        # Check for alerts
        config = self.alert_manager.config
        if win_rate < config.min_win_rate_alert and total >= 10:
            self.alert_manager.create_alert(
                "performance", "medium", f"Low Win Rate: {strategy_name}",
                f"Win rate: {win_rate:.1%} (threshold: {config.min_win_rate_alert:.1%})",
                strategy=strategy_name,
                data={"win_rate": win_rate, "total_trades": total}
            )
        
        return {
            "status": "analyzed",
            "win_rate": win_rate,
            "total_trades": total,
            "total_pnl": total_pnl,
            "avg_trade": avg_trade
        }


class MarketMonitor:
    """
    Monitors market conditions and detects anomalies
    """
    
    def __init__(self, alert_manager: AlertManager):
        self.alert_manager = alert_manager
        self.market_history: Dict[str, List[MarketPerformance]] = {}
        
    def update_market_data(self, performances: List[MarketPerformance]):
        """Update market data and check for alerts"""
        current_time = int(time.time())
        
        for perf in performances:
            symbol = perf.symbol
            
            if symbol not in self.market_history:
                self.market_history[symbol] = []
            
            self.market_history[symbol].append(perf)
            
            # Keep only recent data
            if len(self.market_history[symbol]) > 100:
                self.market_history[symbol] = self.market_history[symbol][-50:]
            
            # Check for market alerts
            self._check_market_alerts(perf)
    
    def _check_market_alerts(self, perf: MarketPerformance):
        """Check for market-related alerts"""
        config = self.alert_manager.config
        symbol = perf.symbol
        
        # Volatility spike alert
        volatility = (perf.high_24h - perf.low_24h) / perf.current_price
        if volatility > config.volatility_spike_threshold:
            self.alert_manager.create_alert(
                "market", "medium", f"High Volatility: {symbol}",
                f"24h volatility: {volatility:.1%} (threshold: {config.volatility_spike_threshold:.1%})",
                symbol=symbol,
                data={"volatility": volatility, "price": perf.current_price}
            )
        
        # Volume drop alert
        if len(self.market_history[symbol]) > 1:
            prev_volume = self.market_history[symbol][-2].volume_24h
            volume_change = (perf.volume_24h - prev_volume) / prev_volume if prev_volume > 0 else 0
            
            if volume_change < -config.volume_drop_threshold:
                self.alert_manager.create_alert(
                    "market", "low", f"Volume Drop: {symbol}",
                    f"Volume dropped {abs(volume_change):.1%}",
                    symbol=symbol,
                    data={"volume_change": volume_change, "current_volume": perf.volume_24h}
                )


class SystemHealthMonitor:
    """
    Monitors system health and performance
    """
    
    def __init__(self, alert_manager: AlertManager):
        self.alert_manager = alert_manager
        self.last_heartbeat = int(time.time())
        self.error_count = 0
        self.api_call_times: List[float] = []
        
    def heartbeat(self):
        """Update system heartbeat"""
        self.last_heartbeat = int(time.time())
    
    def record_error(self, error_type: str, error_message: str):
        """Record a system error"""
        self.error_count += 1
        
        # Alert on multiple errors
        if self.error_count > 5:
            self.alert_manager.create_alert(
                "system", "high", "Multiple System Errors",
                f"Error count: {self.error_count} - Latest: {error_message}",
                data={"error_type": error_type, "error_count": self.error_count}
            )
            self.error_count = 0  # Reset after alert
    
    def record_api_call_time(self, duration: float):
        """Record API call duration"""
        self.api_call_times.append(duration)
        
        # Keep only recent measurements
        if len(self.api_call_times) > 100:
            self.api_call_times = self.api_call_times[-50:]
        
        # Alert on slow API calls
        if duration > 30.0:  # 30 seconds
            self.alert_manager.create_alert(
                "system", "medium", "Slow API Response",
                f"API call took {duration:.1f} seconds",
                data={"duration": duration, "avg_duration": np.mean(self.api_call_times)}
            )
    
    def check_system_health(self) -> Dict:
        """Check overall system health"""
        current_time = int(time.time())
        
        # Check heartbeat
        heartbeat_age = current_time - self.last_heartbeat
        if heartbeat_age > 300:  # 5 minutes
            self.alert_manager.create_alert(
                "system", "critical", "System Unresponsive",
                f"No heartbeat for {heartbeat_age} seconds",
                data={"heartbeat_age": heartbeat_age}
            )
        
        # Calculate API performance
        avg_api_time = np.mean(self.api_call_times) if self.api_call_times else 0
        
        return {
            "heartbeat_age": heartbeat_age,
            "error_count": self.error_count,
            "avg_api_time": avg_api_time,
            "api_calls_tracked": len(self.api_call_times),
            "status": "healthy" if heartbeat_age < 60 else "warning" if heartbeat_age < 300 else "critical"
        }


class ComprehensiveMonitor:
    """
    Comprehensive monitoring system that combines all monitors
    """
    
    def __init__(self, memory: TradingMemory, config: MonitoringConfig = None):
        self.config = config or MonitoringConfig()
        self.alert_manager = AlertManager(self.config)
        self.performance_monitor = PerformanceMonitor(memory, self.alert_manager)
        self.market_monitor = MarketMonitor(self.alert_manager)
        self.system_monitor = SystemHealthMonitor(self.alert_manager)
        
        # Set up alert callback for logging
        self.alert_manager.add_alert_callback(self._log_alert)
    
    def _log_alert(self, alert: Alert):
        """Log alerts to the system"""
        logger.warning(f"Alert: {alert.severity} - {alert.title}: {alert.message}")
    
    def update_trading_metrics(self, equity: float, market_performances: List[MarketPerformance]):
        """Update all trading metrics"""
        self.performance_monitor.update_equity(equity)
        self.market_monitor.update_market_data(market_performances)
        self.system_monitor.heartbeat()
    
    def get_monitoring_dashboard(self) -> Dict:
        """Get comprehensive monitoring dashboard data"""
        return {
            "alerts": {
                "active": len(self.alert_manager.get_active_alerts()),
                "critical": len(self.alert_manager.get_active_alerts("critical")),
                "high": len(self.alert_manager.get_active_alerts("high")),
                "recent": [
                    {
                        "type": a.type,
                        "severity": a.severity,
                        "title": a.title,
                        "message": a.message,
                        "timestamp": a.timestamp
                    } for a in self.alert_manager.active_alerts[-10:]
                ]
            },
            "performance": {
                "current_equity": self.performance_monitor.last_equity,
                "peak_equity": self.performance_monitor.peak_equity,
                "drawdown": ((self.performance_monitor.peak_equity - self.performance_monitor.last_equity) 
                           / self.performance_monitor.peak_equity * 100) if self.performance_monitor.peak_equity > 0 else 0
            },
            "system_health": self.system_monitor.check_system_health(),
            "config": {
                "max_drawdown_alert": self.config.max_drawdown_alert,
                "min_win_rate_alert": self.config.min_win_rate_alert,
                "max_daily_loss_alert": self.config.max_daily_loss_alert
            }
        }