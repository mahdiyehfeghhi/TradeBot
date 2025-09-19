from __future__ import annotations

import asyncio
from dataclasses import dataclass
from loguru import logger
import time

from domain.models import Order, OrderSide, TradingEvent
from domain.ports import MarketDataPort, TradingPort, Strategy, RiskManager
from application.memory import TradingMemory


@dataclass
class TraderContext:
    symbol: str
    loop_interval_sec: int
    risk_pct: float
    take_profit_rr: float
    min_notional: float
    quote_currency: str
    price_precision: int
    quantity_precision: int


class Trader:
    def __init__(self, market: MarketDataPort, broker: TradingPort, strategy: Strategy, 
                 risk: RiskManager, ctx: TraderContext, memory: TradingMemory = None):
        self.market = market
        self.broker = broker
        self.strategy = strategy
        self.risk = risk
        self.ctx = ctx
        self.memory = memory or TradingMemory()
        self._open_positions = {}  # Track open positions for learning

    async def run_once(self, budget_quote: float | None = None):
        """Perform a single trading iteration and return telemetry for UI.
        Note: budget_quote is ignored for equity calc; we always compute from balances
        to dynamically reflect current capital (paper and live)."""
        price = 0.0
        report = None
        equity_quote = 0.0
        meta: dict = {}
        try:
            logger.debug("Fetching latest candles for {sym}", sym=self.ctx.symbol)
            candles = await self.market.get_latest_candles(self.ctx.symbol, limit=100)
            logger.debug("Fetched {n} candles", n=len(list(candles)) if hasattr(candles, "__len__") else "?" )
            ticker = await self.market.get_ticker(self.ctx.symbol)
            price = ticker.price
            logger.info("Ticker price for {sym}: {price}", sym=self.ctx.symbol, price=price)
            
            # Check for position exits before making new decisions
            self._check_position_exits(price)

            decision = self.strategy.on_candles(candles, price)
            logger.info("Decision: action={a} reason={r} stop={s}", a=decision.action, r=getattr(decision, "reason", None), s=getattr(decision, "stop_loss", None))
            
            # Store market conditions for learning
            candles_list = list(candles)
            market_conditions = self._extract_market_conditions(candles_list, price)
            
            meta.update({
                "action": decision.action,
                "reason": getattr(decision, "reason", None),
                "stop": getattr(decision, "stop_loss", None),
                "market_conditions": market_conditions,
            })

            # Determine equity in quote currency combining base+quote
            quote_bal = await self.broker.get_balance(self.ctx.quote_currency)
            base_cur = self.ctx.symbol.split("-")[0]
            base_bal = await self.broker.get_balance(base_cur)  # e.g., BTC from BTC-IRT
            logger.debug("Balances: {base}={b_avail} avail, {quote}={q_avail} avail", base=base_cur, b_avail=base_bal.available, quote=self.ctx.quote_currency, q_avail=quote_bal.available)
            equity_quote = max(0.0, (quote_bal.available or 0.0)) + max(0.0, (base_bal.available or 0.0)) * max(price, 0.0)
            logger.info("Equity (in {q}): {eq}", q=self.ctx.quote_currency, eq=equity_quote)
            meta.update({
                "price": price,
                "equity": equity_quote,
                "base_currency": base_cur,
                "base_available": base_bal.available,
                "quote_currency": self.ctx.quote_currency,
                "quote_available": quote_bal.available,
            })

            if decision.action != "hold" and price > 0:
                size_quote_raw = self.risk.position_size_quote(equity_quote, self.ctx.risk_pct, price, decision.stop_loss)
                logger.debug("Raw position size (quote): {sz}", sz=size_quote_raw)
                # Determine side-specific available capacity in quote terms
                if decision.action == "buy":
                    avail_quote_for_side = max(0.0, quote_bal.available or 0.0)
                    # If portfolio budget cap provided, apply it for buy side
                    if budget_quote is not None:
                        avail_quote_for_side = min(avail_quote_for_side, max(0.0, budget_quote))
                else:  # sell
                    avail_quote_for_side = max(0.0, (base_bal.available or 0.0) * price)
                # Respect min notional but never exceed side-specific available
                size_quote = min(max(size_quote_raw, self.ctx.min_notional), avail_quote_for_side)
                logger.info("Sizing: raw={raw} min_notional={minn} avail_side={avail} -> final={final}", raw=size_quote_raw, minn=self.ctx.min_notional, avail=avail_quote_for_side, final=size_quote)
                meta.update({
                    "size_quote_raw": size_quote_raw,
                    "avail_quote_for_side": avail_quote_for_side,
                    "size_quote": size_quote,
                })

                # Early exit if side-specific available is below exchange min_notional
                if decision.action == "buy" and avail_quote_for_side < self.ctx.min_notional:
                    remaining = max(0.0, self.ctx.min_notional - avail_quote_for_side)
                    logger.warning(
                        "Skipping order: available quote {avail} < min_notional {minn} (remaining {rem})",
                        avail=avail_quote_for_side,
                        minn=self.ctx.min_notional,
                        rem=remaining,
                    )
                    meta["skip_reason"] = "insufficient_quote_min_notional"
                    meta["remaining_to_min_quote"] = remaining
                    return {"price": price, "equity_quote": equity_quote, "report": None, "meta": meta}
                if decision.action == "sell" and (base_bal.available or 0.0) * price < self.ctx.min_notional:
                    notional = max(0.0, (base_bal.available or 0.0) * price)
                    remaining = max(0.0, self.ctx.min_notional - notional)
                    logger.warning(
                        "Skipping order: available base*price {notional} < min_notional {minn} (remaining {rem})",
                        notional=notional,
                        minn=self.ctx.min_notional,
                        rem=remaining,
                    )
                    meta["skip_reason"] = "insufficient_base_min_notional"
                    meta["remaining_to_min_base_quote"] = remaining
                    return {"price": price, "equity_quote": equity_quote, "report": None, "meta": meta}

                if size_quote <= 0:
                    logger.warning("Skipping order: computed size_quote <= 0 after clamping")
                    meta["skip_reason"] = "size_quote<=0"
                    return {"price": price, "equity_quote": equity_quote, "report": None, "meta": meta}

                qty_base = size_quote / price
                # round to precision
                qty_base = round(qty_base, self.ctx.quantity_precision)
                # Ensure notional still meets min_notional after rounding (for buys)
                if decision.action == "buy" and qty_base * price < self.ctx.min_notional:
                    # bump by one precision step if possible given available quote
                    step = 10 ** (-self.ctx.quantity_precision)
                    bumped = round(qty_base + step, self.ctx.quantity_precision)
                    if (bumped * price) <= (max(0.0, quote_bal.available or 0.0)):
                        qty_base = bumped
                if qty_base <= 0:
                    logger.warning("Skipping order: qty_base rounded to 0 (precision={p})", p=self.ctx.quantity_precision)
                    meta["skip_reason"] = "qty_rounded_zero"
                    return {"price": price, "equity_quote": equity_quote, "report": None, "meta": meta}
                side = OrderSide.BUY if decision.action == "buy" else OrderSide.SELL
                # For paper broker, pass current price for immediate fill; live adapters can ignore price for market orders
                order = Order(symbol=self.ctx.symbol, side=side, quantity=qty_base, price=price)
                logger.info("Placing order: {side} {qty} {base} @ ~{price} {quote} (notional ~{notional})", side=side.value, qty=qty_base, base=base_cur, price=price, quote=self.ctx.quote_currency, notional=round(qty_base*price, 2))
                report = await self.broker.place_order(order)
                logger.info("Order executed: {r}", r=report)
                
                # Store trading event for learning
                if report and report.status in ["filled", "partial"]:
                    trading_event = TradingEvent(
                        timestamp=int(time.time()),
                        symbol=self.ctx.symbol,
                        action=decision.action,
                        reason=decision.reason,
                        entry_price=report.avg_price,
                        quantity=report.executed_qty,
                        market_conditions=market_conditions,
                        strategy_used=type(self.strategy).__name__
                    )
                    event_id = self.memory.store_trading_event(trading_event)
                    
                    # Track open position for later PnL calculation
                    self._open_positions[self.ctx.symbol] = {
                        "event_id": event_id,
                        "entry_price": report.avg_price,
                        "quantity": report.executed_qty,
                        "side": side.value,
                        "timestamp": int(time.time()),
                        "stop_loss": decision.stop_loss,
                        "take_profit": decision.take_profit
                    }
                    
                meta.update({
                    "placed": True,
                    "side": side.value,
                    "qty_base": qty_base,
                    "event_stored": report and report.status in ["filled", "partial"]
                })
            else:
                if price <= 0:
                    logger.warning("No trading: price is non-positive ({p})", p=price)
                    meta["skip_reason"] = "price_non_positive"
                else:
                    logger.debug("No trading: strategy action is HOLD")
                    meta["skip_reason"] = "hold"
        except Exception as e:
            logger.exception("Trader iteration error: {e}")

        return {"price": price, "equity_quote": equity_quote, "report": report, "meta": meta}
    
    def _extract_market_conditions(self, candles: list, current_price: float) -> dict:
        """Extract market conditions for learning purposes"""
        if len(candles) < 20:
            return {}
            
        import pandas as pd
        import numpy as np
        
        df = pd.DataFrame([{
            "close": c.close,
            "volume": c.volume,
            "high": c.high,
            "low": c.low
        } for c in candles])
        
        try:
            # Calculate basic indicators
            sma_20 = df["close"].rolling(20).mean().iloc[-1] if len(df) >= 20 else current_price
            volatility = df["close"].pct_change().rolling(10).std().iloc[-1] * 100
            
            # RSI calculation
            delta = df["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / (loss.replace(0, np.nan))
            rsi = (100 - (100 / (1 + rs))).iloc[-1] if len(df) >= 14 else 50
            
            # Volume analysis
            avg_volume = df["volume"].rolling(20).mean().iloc[-1]
            current_volume = df["volume"].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Price position
            high_20 = df["high"].rolling(20).max().iloc[-1]
            low_20 = df["low"].rolling(20).min().iloc[-1]
            price_position = (current_price - low_20) / (high_20 - low_20) if high_20 > low_20 else 0.5
            
            return {
                "sma_20": float(sma_20),
                "volatility": float(volatility),
                "rsi": float(rsi),
                "volume_ratio": float(volume_ratio),
                "price_position": float(price_position),
                "trend": "bullish" if current_price > sma_20 else "bearish"
            }
        except Exception as e:
            logger.warning("Failed to extract market conditions: {err}", err=e)
            return {}
    
    def _check_position_exits(self, current_price: float):
        """Check if any open positions should be closed and update learning data"""
        if self.ctx.symbol not in self._open_positions:
            return
            
        position = self._open_positions[self.ctx.symbol]
        entry_price = position["entry_price"]
        quantity = position["quantity"]
        side = position["side"]
        event_id = position["event_id"]
        
        # Calculate current PnL
        if side == "buy":
            pnl = (current_price - entry_price) * quantity
        else:  # sell
            pnl = (entry_price - current_price) * quantity
            
        # Check stop loss and take profit
        should_exit = False
        exit_reason = ""
        
        if position.get("stop_loss") and side == "buy" and current_price <= position["stop_loss"]:
            should_exit = True
            exit_reason = "stop_loss"
        elif position.get("stop_loss") and side == "sell" and current_price >= position["stop_loss"]:
            should_exit = True
            exit_reason = "stop_loss"
        elif position.get("take_profit") and side == "buy" and current_price >= position["take_profit"]:
            should_exit = True
            exit_reason = "take_profit"
        elif position.get("take_profit") and side == "sell" and current_price <= position["take_profit"]:
            should_exit = True
            exit_reason = "take_profit"
            
        # Update learning data (even if not exiting, for ongoing tracking)
        duration_minutes = (int(time.time()) - position["timestamp"]) // 60
        outcome = "win" if pnl > 0 else ("loss" if pnl < 0 else "breakeven")
        
        try:
            self.memory.update_trading_event(event_id, 
                                           exit_price=current_price,
                                           pnl=pnl,
                                           duration_minutes=duration_minutes,
                                           outcome=outcome)
        except Exception as e:
            logger.warning("Failed to update trading event: {err}", err=e)
            
        # If position should be closed, remove from tracking
        if should_exit:
            logger.info("Position exit triggered: {reason} PnL: {pnl}", reason=exit_reason, pnl=pnl)
            del self._open_positions[self.ctx.symbol]

    async def run(self, budget_quote: float | None = None):
        logger.info("Starting trader loop for {symbol}", symbol=self.ctx.symbol)
        while True:
            await self.run_once(budget_quote)
            await asyncio.sleep(self.ctx.loop_interval_sec)
