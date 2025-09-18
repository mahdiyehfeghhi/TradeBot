from __future__ import annotations

import math
from dataclasses import dataclass

from domain.models import AccountBalance, ExecutionReport, Order
from domain.ports import TradingPort


@dataclass
class PaperState:
    base_currency: str
    quote_currency: str
    base_balance: float = 0.0
    quote_balance: float = 1_000_000.0


class PaperBroker(TradingPort):
    def __init__(self, state: PaperState):
        self.state = state

    async def place_order(self, order: Order) -> ExecutionReport:
        # Market order fill at current price passed via order.price=None -> we need caller's last price
        # For simplicity, assume we have to pass avg_price via quantity*last_price on caller side.
        # Here we require order.price to be set to last trade price for paper execution if provided.
        price = float(order.price) if order.price is not None else math.nan
        # If price NaN, reject
        if math.isnan(price) or price <= 0:
            return ExecutionReport(order=order, executed_qty=0.0, avg_price=0.0, status="rejected")

        if order.side.value == "buy":
            cost = order.quantity * price
            if cost > self.state.quote_balance:
                return ExecutionReport(order=order, executed_qty=0.0, avg_price=0.0, status="rejected")
            self.state.quote_balance -= cost
            self.state.base_balance += order.quantity
        else:
            if order.quantity > self.state.base_balance:
                return ExecutionReport(order=order, executed_qty=0.0, avg_price=0.0, status="rejected")
            self.state.base_balance -= order.quantity
            self.state.quote_balance += order.quantity * price

        return ExecutionReport(order=order, executed_qty=order.quantity, avg_price=price, status="filled")

    async def get_balance(self, currency: str) -> AccountBalance:
        if currency == self.state.base_currency:
            return AccountBalance(currency=currency, total=self.state.base_balance, available=self.state.base_balance)
        else:
            return AccountBalance(currency=currency, total=self.state.quote_balance, available=self.state.quote_balance)
