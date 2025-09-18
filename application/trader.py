from __future__ import annotations

import asyncio
from dataclasses import dataclass
from loguru import logger

from domain.models import Order, OrderSide
from domain.ports import MarketDataPort, TradingPort, Strategy, RiskManager


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
    def __init__(self, market: MarketDataPort, broker: TradingPort, strategy: Strategy, risk: RiskManager, ctx: TraderContext):
        self.market = market
        self.broker = broker
        self.strategy = strategy
        self.risk = risk
        self.ctx = ctx

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

            decision = self.strategy.on_candles(candles, price)
            logger.info("Decision: action={a} reason={r} stop={s}", a=decision.action, r=getattr(decision, "reason", None), s=getattr(decision, "stop_loss", None))
            meta.update({
                "action": decision.action,
                "reason": getattr(decision, "reason", None),
                "stop": getattr(decision, "stop_loss", None),
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
                meta.update({
                    "placed": True,
                    "side": side.value,
                    "qty_base": qty_base,
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

    async def run(self, budget_quote: float | None = None):
        logger.info("Starting trader loop for {symbol}", symbol=self.ctx.symbol)
        while True:
            await self.run_once(budget_quote)
            await asyncio.sleep(self.ctx.loop_interval_sec)
