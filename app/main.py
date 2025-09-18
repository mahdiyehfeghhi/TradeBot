from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from app.config import Config, Settings
from app.logger import setup_logging
from application.risk import SimpleRiskManager
from application.strategies import RSIMAStrategy
from application.trader import Trader, TraderContext
from infrastructure.csv_market import CSVMarketData
from infrastructure.paper_broker import PaperBroker, PaperState
from infrastructure.wallex.rest_client import WallexRestClient


def build_args():
    p = argparse.ArgumentParser(description="TradeBot Wallex")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--mode", choices=["paper", "live"], default=None)
    p.add_argument("--symbol", default=None)
    p.add_argument("--budget", type=float, default=None, help="Paper budget in quote currency")
    p.add_argument("--loop-interval", type=int, default=None)
    p.add_argument("--csv", default=None, help="Optional CSV for paper/backtest market data")
    return p.parse_args()


async def main():
    setup_logging()
    args = build_args()
    cfg = Config.load(Path(args.config))
    env = Settings()

    # CLI overrides
    if args.mode:
        cfg.app.mode = args.mode
    if args.symbol:
        cfg.app.symbol = args.symbol
    if args.loop_interval is not None:
        cfg.app.loop_interval_sec = args.loop_interval

    symbol_cfg = cfg.symbols[cfg.app.symbol]

    # Strategy
    strat_cfg = cfg.strategy.params
    strategy = RSIMAStrategy(
        rsi_period=strat_cfg.rsi_period,
        rsi_buy=strat_cfg.rsi_buy,
        rsi_sell=strat_cfg.rsi_sell,
        ma_fast=strat_cfg.ma_fast,
        ma_slow=strat_cfg.ma_slow,
    )
    risk = SimpleRiskManager(take_profit_rr=cfg.risk.take_profit_rr)

    # Ports
    if cfg.app.mode == "paper":
        if args.csv:
            market = CSVMarketData(args.csv, cfg.app.symbol)
        else:
            # Fallback: use REST client as market-only
            market = WallexRestClient(cfg.wallex.base_url, env.wallex_api_key, env.wallex_api_secret, env.app_user_agent, cfg.wallex.endpoints, cfg.wallex.symbol_transform)
        broker = PaperBroker(PaperState(cfg.app.base_currency, cfg.app.quote_currency, base_balance=0.0, quote_balance=cfg.app.budget))
    else:
        market = WallexRestClient(
            cfg.wallex.base_url,
            env.wallex_api_key,
            env.wallex_api_secret,
            env.app_user_agent,
            cfg.wallex.endpoints,
            cfg.wallex.symbol_transform,
        )
        # TODO: implement real trading port for Wallex; currently reuse market client and raise for place_order
        from infrastructure.wallex.trading_port import WallexTradingPort  # type: ignore
        broker = WallexTradingPort(cfg.wallex, env)

    ctx = TraderContext(
        symbol=cfg.app.symbol,
        loop_interval_sec=cfg.app.loop_interval_sec,
        risk_pct=cfg.risk.risk_per_trade_pct,
        take_profit_rr=cfg.risk.take_profit_rr,
        min_notional=symbol_cfg.min_notional,
        quote_currency=cfg.app.quote_currency,
        price_precision=symbol_cfg.price_precision,
        quantity_precision=symbol_cfg.quantity_precision,
    )

    trader = Trader(market, broker, strategy, risk, ctx)
    try:
        await trader.run(budget_quote=args.budget if cfg.app.mode == "paper" else None)
    finally:
        if isinstance(market, WallexRestClient):
            await market.close()


if __name__ == "__main__":
    asyncio.run(main())
