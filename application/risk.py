from __future__ import annotations

from domain.ports import RiskManager


class SimpleRiskManager(RiskManager):
    def __init__(self, take_profit_rr: float = 1.5):
        self.take_profit_rr = take_profit_rr

    def position_size_quote(self, equity_quote: float, risk_pct: float, entry: float, stop: float | None) -> float:
        # Risk capital in quote currency
        risk_cap = max(equity_quote * (risk_pct / 100.0), 0)
        if stop is None or stop <= 0 or entry <= 0:
            # fallback to a tiny position
            return min(risk_cap, equity_quote)
        risk_per_unit = abs(entry - stop)
        if risk_per_unit <= 0:
            return min(risk_cap, equity_quote)
        # For quote-denominated size, approximate unit risk in quote terms
        # If entry is price (quote/base), risk_per_unit is also in quote per base unit.
        # So size in base = risk_cap / risk_per_unit, size in quote = size_base * entry
        size_base = risk_cap / risk_per_unit
        size_quote = size_base * entry
        return min(size_quote, equity_quote)
