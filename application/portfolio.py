from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from loguru import logger

from .trader import Trader, TraderContext
from domain.ports import MarketDataPort, TradingPort, Strategy, RiskManager


@dataclass
class SymbolState:
    symbol: str
    ctx: TraderContext
    trader: Trader


class PortfolioEngine:
    def __init__(
        self,
        market: MarketDataPort,
        broker: TradingPort,
        strategy_factory,
        risk: RiskManager,
        symbols: List[str],
        ctx_map: Dict[str, TraderContext],
    ):
        self.market = market
        self.broker = broker
        self.risk = risk
        self.states: List[SymbolState] = []
        for sym in symbols:
            strategy: Strategy = strategy_factory()
            trader = Trader(market, broker, strategy, risk, ctx_map[sym])
            self.states.append(SymbolState(symbol=sym, ctx=ctx_map[sym], trader=trader))

    async def run_once(self) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for st in self.states:
            try:
                out = await st.trader.run_once()
                results.append({"symbol": st.symbol, **out})
            except Exception as e:
                logger.exception("Portfolio symbol {s} iteration error: {e}", s=st.symbol)
        return results