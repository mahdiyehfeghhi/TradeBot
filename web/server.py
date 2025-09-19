from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
import webbrowser

from app.config import Config, Settings
from app.logger import setup_logging
from application.risk import SimpleRiskManager
from application.strategies import RSIMAStrategy, BreakoutStrategy, MeanReversionStrategy, EnsembleStrategy
from application.trader import Trader, TraderContext
from application.portfolio import PortfolioEngine
from application.universe import rank_symbols
from infrastructure.csv_market import CSVMarketData
from infrastructure.paper_broker import PaperBroker, PaperState
from infrastructure.wallex.rest_client import WallexRestClient
import httpx
import importlib


@dataclass
class MetricsBus:
    is_running: bool = False
    last_price: float = 0.0
    equity_series: List[Dict[str, Any]] = field(default_factory=list)
    price_series: List[Dict[str, Any]] = field(default_factory=list)
    trades: List[Dict[str, Any]] = field(default_factory=list)
    events: List[str] = field(default_factory=list)
    # Optional per-symbol series for portfolio mode
    per_symbol: Dict[str, Dict[str, List[Dict[str, Any]]]] = field(default_factory=dict)

    def on_price(self, price: float):
        self.last_price = price
        self.price_series.append({"t": datetime.utcnow().isoformat(), "price": price})
        if len(self.price_series) > 2000:
            self.price_series = self.price_series[-2000:]

    def on_equity(self, equity_quote: float):
        self.equity_series.append({"t": datetime.utcnow().isoformat(), "equity": equity_quote})
        if len(self.equity_series) > 2000:
            self.equity_series = self.equity_series[-2000:]

    def on_trade(self, report: Dict[str, Any]):
        self.trades.append(report)
        if len(self.trades) > 1000:
            self.trades = self.trades[-1000:]
    def on_event(self, msg: str):
        self.events.append(msg)
        if len(self.events) > 500:
            self.events = self.events[-500:]


class EngineController:
    def __init__(self):
        self._task: Optional[asyncio.Task] = None
        self.metrics = MetricsBus()
        self._market = None
        self._broker = None
        self._portfolio: Optional[PortfolioEngine] = None

    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self, cfg_path: str = "config.yaml", csv_path: Optional[str] = None, *, mode: Optional[str] = None, symbol: Optional[str] = None, budget: Optional[float] = None, loop_interval: Optional[int] = None):
        if self.is_running():
            return

        cfg = Config.load(Path(cfg_path))
        env = Settings()

        # Apply overrides
        if mode in ("paper", "live"):
            cfg.app.mode = mode
        if symbol:
            cfg.app.symbol = symbol
        if loop_interval is not None and loop_interval > 0:
            cfg.app.loop_interval_sec = int(loop_interval)
        if budget is not None and budget > 0:
            cfg.app.budget = float(budget)

        strat = cfg.strategy.params
        def strategy_factory():
            # Build an ensemble of three strategies
            s1 = RSIMAStrategy(
                rsi_period=strat.rsi_period,
                rsi_buy=strat.rsi_buy,
                rsi_sell=strat.rsi_sell,
                ma_fast=strat.ma_fast,
                ma_slow=strat.ma_slow,
            )
            s2 = BreakoutStrategy(lookback=strat.breakout_lookback)
            s3 = MeanReversionStrategy(
                bb_period=strat.bb_period,
                bb_std=strat.bb_std,
                rsi_period=strat.rsi_period,
                rsi_buy=strat.rsi_buy,
                rsi_sell=strat.rsi_sell,
            )
            return EnsembleStrategy([s1, s2, s3], threshold=strat.ensemble_threshold)
        risk = SimpleRiskManager(take_profit_rr=cfg.risk.take_profit_rr)

        if cfg.app.mode == "paper":
            if csv_path is None:
                default_csv = Path(__file__).parents[1] / "data" / "sample_candles.csv"
                csv_path = str(default_csv) if default_csv.exists() else None
            market = CSVMarketData(csv_path, cfg.app.symbol) if csv_path else WallexRestClient(
                cfg.wallex.base_url,
                env.wallex_api_key,
                env.wallex_api_secret,
                env.app_user_agent,
                cfg.wallex.endpoints,
                cfg.wallex.symbol_transform,
                cfg.wallex.candle_resolution,
            )
            broker = PaperBroker(PaperState(cfg.app.base_currency, cfg.app.quote_currency, base_balance=0.0, quote_balance=cfg.app.budget))
        else:
            market = WallexRestClient(
                cfg.wallex.base_url,
                env.wallex_api_key,
                env.wallex_api_secret,
                env.app_user_agent,
                cfg.wallex.endpoints,
                cfg.wallex.symbol_transform,
                cfg.wallex.candle_resolution,
            )
            from infrastructure.wallex.trading_port import WallexTradingPort
            broker = WallexTradingPort(cfg.wallex, env)

        # Build contexts
        ctx_map: Dict[str, TraderContext] = {}
        if cfg.portfolio and cfg.portfolio.enabled:
            # Determine symbols: manual or auto selection
            if cfg.portfolio.selection == "auto":
                candidate = cfg.portfolio.universe or cfg.portfolio.symbols or [cfg.app.symbol]
                # use market client for ranking (real data)
                market_for_universe = WallexRestClient(
                    cfg.wallex.base_url,
                    env.wallex_api_key,
                    env.wallex_api_secret,
                    env.app_user_agent,
                    cfg.wallex.endpoints,
                    cfg.wallex.symbol_transform,
                    cfg.wallex.candle_resolution,
                )
                try:
                    scores, selected = await rank_symbols(
                        market_for_universe,
                        candidate,
                        lookback_candles=cfg.portfolio.lookback_candles,
                        top_k=cfg.portfolio.top_k,
                    )
                    # Filter by configured symbols to ensure min_notional/precision are available
                    selected_cfg = [s for s in selected if s in cfg.symbols]
                    if not selected_cfg:
                        # try fallback to any candidates that are configured
                        selected_cfg = [s for s in candidate if s in cfg.symbols]
                    if not selected_cfg:
                        # As a last resort, fall back to the primary app symbol
                        selected_cfg = [cfg.app.symbol]
                    symbols = selected_cfg or [cfg.app.symbol]
                    # Log and emit to events
                    for s in scores:
                        self.metrics.on_event(f"[Universe] {s.symbol} score={s.score:.6f} vol%={s.volatility:.4f} liq={s.liquidity:.2f}")
                    self.metrics.on_event(f"[Universe] Selected: {', '.join(symbols)} (from {', '.join(candidate)})")
                finally:
                    try:
                        await market_for_universe.close()
                    except Exception:
                        pass
            else:
                symbols = cfg.portfolio.symbols or [cfg.app.symbol]
        else:
            symbols = [cfg.app.symbol]
        for sym in symbols:
            sym_cfg = cfg.symbols[sym]
            ctx_map[sym] = TraderContext(
                symbol=sym,
                loop_interval_sec=cfg.app.loop_interval_sec,
                risk_pct=cfg.risk.risk_per_trade_pct,
                take_profit_rr=cfg.risk.take_profit_rr,
                min_notional=sym_cfg.min_notional,
                quote_currency=cfg.app.quote_currency,
                price_precision=sym_cfg.price_precision,
                quantity_precision=sym_cfg.quantity_precision,
            )

        # Initialize engine(s)
        if cfg.portfolio and cfg.portfolio.enabled and len(symbols) > 1:
            self._portfolio = PortfolioEngine(market, broker, strategy_factory, risk, symbols, ctx_map)
            trader = None
        else:
            self._portfolio = None
            trader = Trader(market, broker, strategy_factory(), risk, ctx_map[symbols[0]])
        # keep refs for cleanup
        self._market = market
        self._broker = broker
        self.metrics.is_running = True

        async def run_wrapper():
            budget_quote = cfg.app.budget if cfg.app.mode == "paper" else None
            try:
                while True:
                    if self._portfolio is not None:
                        outs = await self._portfolio.run_once()
                        for out in outs:
                            sym = out.get("symbol")
                            price = out.get("price", 0.0)
                            equity_quote = out.get("equity_quote", 0.0)
                            report = out.get("report")
                            meta = out.get("meta") or {}
                            # per-symbol series
                            ps = self.metrics.per_symbol.setdefault(sym, {"price": [], "equity": []})
                            if price:
                                ps["price"].append({"t": datetime.utcnow().isoformat(), "price": price})
                                if len(ps["price"]) > 2000:
                                    ps["price"] = ps["price"][-2000:]
                            if equity_quote:
                                ps["equity"].append({"t": datetime.utcnow().isoformat(), "equity": equity_quote})
                                if len(ps["equity"]) > 2000:
                                    ps["equity"] = ps["equity"][-2000:]
                            # events/trades
                            try:
                                action = meta.get("action")
                                reason = meta.get("reason")
                                skip = meta.get("skip_reason")
                                msg = None
                                if report is not None:
                                    side = getattr(report.order.side, "value", str(report.order.side))
                                    msg = f"[{sym}] سفارش {side} با مقدار {report.executed_qty} در قیمت {report.avg_price} ثبت شد وضعیت: {report.status}"
                                elif skip == "hold":
                                    msg = f"[{sym}] هیچ معامله‌ای انجام نشد: سیگنال ندارد ({reason or 'نامشخص'})"
                                elif skip == "size_quote<=0":
                                    msg = f"[{sym}] هیچ معامله‌ای انجام نشد: موجودی کافی برای حداقل مبلغ سفارش وجود ندارد"
                                elif skip == "qty_rounded_zero":
                                    msg = f"[{sym}] هیچ معامله‌ای انجام نشد: مقدار سفارش پس از گرد کردن صفر شد"
                                elif skip == "price_non_positive":
                                    msg = f"[{sym}] هیچ معامله‌ای انجام نشد: قیمت معتبر دریافت نشد"
                                elif skip == "insufficient_quote_min_notional":
                                    rem = meta.get("remaining_to_min_quote")
                                    if rem is not None:
                                        msg = f"[{sym}] هیچ معامله‌ای انجام نشد: موجودی تومانی کمتر از حداقل است (کمبود: {rem:,.0f} تومان)"
                                    else:
                                        msg = f"[{sym}] هیچ معامله‌ای انجام نشد: موجودی تومانی کمتر از حداقل مبلغ سفارش است"
                                elif skip == "insufficient_base_min_notional":
                                    rem = meta.get("remaining_to_min_base_quote")
                                    if rem is not None:
                                        msg = f"[{sym}] هیچ معامله‌ای انجام نشد: ارزش دارایی پایه کمتر از حداقل است (کمبود معادل: {rem:,.0f} تومان)"
                                    else:
                                        msg = f"[{sym}] هیچ معامله‌ای انجام نشد: ارزش دارایی پایه کمتر از حداقل مبلغ سفارش است"
                                if msg:
                                    self.metrics.on_event(msg)
                            except Exception:
                                pass
                            if report is not None:
                                self.metrics.on_trade({
                                    "time": datetime.utcnow().isoformat(),
                                    "symbol": sym,
                                    "side": getattr(report.order.side, "value", str(report.order.side)),
                                    "qty": report.executed_qty,
                                    "price": report.avg_price,
                                    "status": report.status,
                                })
                        await asyncio.sleep(cfg.app.loop_interval_sec)
                    else:
                        out = await trader.run_once(budget_quote=budget_quote)
                        price = out.get("price", 0.0)
                        equity_quote = out.get("equity_quote", 0.0)
                        report = out.get("report")
                        meta = out.get("meta") or {}
                        if price:
                            self.metrics.on_price(price)
                        if equity_quote:
                            self.metrics.on_equity(equity_quote)
                        # Build Persian message for UI
                        try:
                            action = meta.get("action")
                            reason = meta.get("reason")
                            skip = meta.get("skip_reason")
                            msg = None
                            if report is not None:
                                side = getattr(report.order.side, "value", str(report.order.side))
                                msg = f"سفارش {side} با مقدار {report.executed_qty} در قیمت {report.avg_price} ثبت شد وضعیت: {report.status}"
                            elif skip == "hold":
                                msg = f"هیچ معامله‌ای انجام نشد: سیگنال ندارد ({reason or 'نامشخص'})"
                            elif skip == "size_quote<=0":
                                msg = "هیچ معامله‌ای انجام نشد: موجودی کافی برای حداقل مبلغ سفارش وجود ندارد"
                            elif skip == "qty_rounded_zero":
                                msg = "هیچ معامله‌ای انجام نشد: مقدار سفارش پس از گرد کردن صفر شد"
                            elif skip == "price_non_positive":
                                msg = "هیچ معامله‌ای انجام نشد: قیمت معتبر دریافت نشد"
                            elif skip == "insufficient_quote_min_notional":
                                rem = meta.get("remaining_to_min_quote")
                                if rem is not None:
                                    msg = f"هیچ معامله‌ای انجام نشد: موجودی تومانی کمتر از حداقل است (کمبود: {rem:,.0f} تومان)"
                                else:
                                    msg = "هیچ معامله‌ای انجام نشد: موجودی تومانی کمتر از حداقل مبلغ سفارش است"
                            elif skip == "insufficient_base_min_notional":
                                rem = meta.get("remaining_to_min_base_quote")
                                if rem is not None:
                                    msg = f"هیچ معامله‌ای انجام نشد: ارزش دارایی پایه کمتر از حداقل است (کمبود معادل: {rem:,.0f} تومان)"
                                else:
                                    msg = "هیچ معامله‌ای انجام نشد: ارزش دارایی پایه کمتر از حداقل مبلغ سفارش است"
                            if msg:
                                self.metrics.on_event(msg)
                        except Exception:
                            pass
                        if report is not None:
                            self.metrics.on_trade({
                                "time": datetime.utcnow().isoformat(),
                                "symbol": ctx_map[symbols[0]].symbol,
                                "side": getattr(report.order.side, "value", str(report.order.side)),
                                "qty": report.executed_qty,
                                "price": report.avg_price,
                                "status": report.status,
                            })
                        await asyncio.sleep(ctx_map[symbols[0]].loop_interval_sec)
            except asyncio.CancelledError:
                pass
            finally:
                self.metrics.is_running = False
                # Close adapters if they have close methods
                try:
                    if hasattr(self._market, "close") and callable(self._market.close):
                        await self._market.close()  # type: ignore[attr-defined]
                except Exception:
                    pass
                try:
                    # WallexTradingPort exposes _close()
                    if hasattr(self._broker, "_close") and callable(self._broker._close):  # type: ignore[attr-defined]
                        await self._broker._close()  # type: ignore[attr-defined]
                except Exception:
                    pass

        self._task = asyncio.create_task(run_wrapper())

    async def stop(self):
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except Exception:
                pass
        self._task = None
        self.metrics.is_running = False
        # best-effort cleanup if loop terminated abruptly
        try:
            if hasattr(self._market, "close") and callable(self._market.close):
                await self._market.close()  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            if hasattr(self._broker, "_close") and callable(self._broker._close):  # type: ignore[attr-defined]
                await self._broker._close()  # type: ignore[attr-defined]
        except Exception:
            pass


setup_logging("DEBUG")
controller = EngineController()
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = Path(__file__).with_name("static")
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).with_name("static").joinpath("index.html")
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.on_event("startup")
async def _open_browser_once():
    """Open the default browser to the dashboard shortly after server starts.
    This runs best when starting via VS Code task or terminal.
    """
    try:
        # Small delay to ensure Uvicorn is ready
        await asyncio.sleep(0.7)
        webbrowser.open_new_tab("http://127.0.0.1:8001/")
    except Exception:
        # Non-fatal if browser can't be opened (e.g., headless or permissions)
        pass


@app.post("/api/start")
async def api_start(cfg: str = "config.yaml", csv: Optional[str] = None, mode: Optional[str] = None, symbol: Optional[str] = None, budget: Optional[float] = None, loop_interval: Optional[int] = None):
    logger.info("API start requested: cfg={cfg} mode={mode} symbol={symbol} budget={budget} loop={loop}", cfg=cfg, mode=mode, symbol=symbol, budget=budget, loop=loop_interval)
    try:
        if controller.is_running():
            return {"ok": True, "already_running": True}
        # Normalize symbol if provided (handle TMN/IRT/IRR aliases and formatting)
        norm_symbol = symbol
        if symbol:
            try:
                s = symbol.strip().upper().replace("_", "-")
                if s.endswith("-IRT") or s.endswith("-IRR"):
                    s = s.rsplit("-", 1)[0] + "-TMN"
                norm_symbol = s
            except Exception:
                norm_symbol = symbol
        # Validate symbol against config if provided
        if norm_symbol:
            from app.config import Config as _Cfg
            cfg_obj = _Cfg.load(Path(cfg))
            if norm_symbol not in cfg_obj.symbols:
                allowed = ", ".join(sorted(cfg_obj.symbols.keys()))
                err = f"نماد '{norm_symbol}' در تنظیمات تعریف نشده است. نمادهای مجاز: {allowed}"
                controller.metrics.on_event(err)
                return {"ok": False, "error": err}

        await controller.start(cfg, csv, mode=mode, symbol=norm_symbol, budget=budget, loop_interval=loop_interval)
        return {"ok": True, "already_running": False}
    except Exception as e:
        # Log and surface the error to UI events for visibility
        logger.exception("Engine start failed")
        try:
            controller.metrics.on_event(f"خطا در شروع ربات: {e}")
        except Exception:
            pass
        return {"ok": False, "error": str(e)}


@app.post("/api/stop")
async def api_stop():
    logger.info("API stop requested")
    await controller.stop()
    return {"ok": True}


@app.get("/api/status")
async def api_status():
    return {"running": controller.is_running(), "last_price": controller.metrics.last_price}


@app.get("/api/metrics")
async def api_metrics():
    return {
        "price": controller.metrics.price_series,
        "equity": controller.metrics.equity_series,
        "trades": controller.metrics.trades,
        "events": controller.metrics.events[-50:],
        "per_symbol": controller.metrics.per_symbol,
    }


@app.get("/api/logs/recent")
async def api_logs_recent(lines: int = 200):
    """Return the last N lines of the rotating log file for quick inspection."""
    log_path = Path("logs").joinpath("tradebot.log")
    if not log_path.exists():
        return {"ok": False, "error": "Log file not found yet."}
    try:
        # Read last N lines safely
        with log_path.open("r", encoding="utf-8", errors="replace") as f:
            content = f.readlines()
        tail = content[-max(1, min(lines, 2000)) :]
        return {"ok": True, "lines": tail}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/api/check-live-balance")
async def api_check_live_balance():
    """Validate Wallex API credentials by fetching balances without placing trades."""
    cfg = Config.load(Path("config.yaml"))
    env = Settings()
    try:
        # Reload module to ensure latest code is used without needing full server restart
        try:
            from infrastructure.wallex import trading_port as _trading_port_mod  # type: ignore
            importlib.reload(_trading_port_mod)
        except Exception:
            pass
        from infrastructure.wallex.trading_port import WallexTradingPort
        port = WallexTradingPort(cfg.wallex, env)
        quote = await port.get_balance(cfg.app.quote_currency)
        base = await port.get_balance(cfg.app.base_currency)
        try:
            await port._close()  # best-effort cleanup
        except Exception:
            pass
        return {
            "ok": True,
            "base": {"currency": base.currency, "total": base.total, "available": base.available},
            "quote": {"currency": quote.currency, "total": quote.total, "available": quote.available},
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/api/wallex/raw-balances")
async def api_wallex_raw_balances():
    """Return raw balances payload from Wallex to troubleshoot currency keys (TMN/IRT/IRR)."""
    cfg = Config.load(Path("config.yaml"))
    env = Settings()
    try:
        url = (cfg.wallex.base_url.rstrip("/") + (cfg.wallex.endpoints.get("balances", "/v1/account/balances")))
        headers = { (cfg.wallex.auth_headers or {}).get("key", "x-api-key"): (env.wallex_api_key or "") }
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(url, headers=headers)
            data = {"status": r.status_code, "body": r.json()}
        return {"ok": True, "data": data}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/place-test-order")
async def api_place_test_order(symbol: Optional[str] = None, side: str = "buy", dry_run: bool = True):
    """Prepare a tiny test order using min_notional and current price.
    By default (dry_run=True) it DOES NOT place an order; set dry_run=false to actually place.
    """
    cfg = Config.load(Path("config.yaml"))
    env = Settings()
    sym = symbol or cfg.app.symbol
    sym_cfg = cfg.symbols[sym]
    # market client for price
    market = WallexRestClient(
        cfg.wallex.base_url,
        env.wallex_api_key,
        env.wallex_api_secret,
        env.app_user_agent,
        cfg.wallex.endpoints,
        cfg.wallex.symbol_transform,
        cfg.wallex.candle_resolution,
    )
    try:
        ticker = await market.get_ticker(sym)
        price = ticker.price
    finally:
        try:
            await market.close()
        except Exception:
            pass

    # compute minimal qty from min_notional: ceil to ensure notional meets/exceeds min_notional after rounding
    raw_qty = sym_cfg.min_notional / max(price, 1e-9)
    step = 10 ** (-sym_cfg.quantity_precision)
    # Ceil to nearest step
    import math
    qty = round(math.ceil(raw_qty / step) * step, sym_cfg.quantity_precision)
    if qty <= 0:
        return {"ok": False, "error": "Computed quantity is zero; adjust min_notional or precisions."}

    if dry_run:
        return {
            "ok": True,
            "dry_run": True,
            "symbol": sym,
            "side": side,
            "price": price,
            "quantity": qty,
            "notional": round(qty * price, sym_cfg.price_precision),
        }

    # place real order (CAUTION)
    from domain.models import Order, OrderSide
    from infrastructure.wallex.trading_port import WallexTradingPort
    port = WallexTradingPort(cfg.wallex, env)
    try:
        order = Order(symbol=sym, side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL, quantity=qty, price=None)
        rep = await port.place_order(order)
        return {"ok": True, "dry_run": False, "report": {
            "status": rep.status,
            "executed_qty": rep.executed_qty,
            "avg_price": rep.avg_price,
            "id": rep.id,
        }}
    except Exception as e:
        return {"ok": False, "error": str(e)}
    finally:
        try:
            await port._close()
        except Exception:
            pass


# Enhanced TradeBot API endpoints

@app.get("/api/strategies")
async def api_get_strategies():
    """Get list of available trading strategies"""
    from application.strategy_factory import StrategyFactory
    return {"ok": True, "strategies": StrategyFactory.get_available_strategies()}


@app.get("/api/market-performance")
async def api_market_performance(symbols: str = "BTC-TMN,ETH-TMN,DOGE-TMN"):
    """Get 24h market performance for specified symbols"""
    try:
        cfg = Config.load(Path("config.yaml"))
        env = Settings()
        
        from application.scanner import MarketScanner
        from application.memory import TradingMemory
        from infrastructure.wallex.rest_client import WallexRestClient
        
        market = WallexRestClient(
            cfg.wallex.base_url,
            env.wallex_api_key,
            env.wallex_api_secret,
            env.app_user_agent,
            cfg.wallex.endpoints,
            cfg.wallex.symbol_transform,
            cfg.wallex.candle_resolution,
        )
        
        memory = TradingMemory()
        scanner = MarketScanner(market, memory)
        
        symbol_list = [s.strip() for s in symbols.split(",")]
        performances = await scanner.scan_symbols(symbol_list)
        
        await market.close()
        
        return {
            "ok": True,
            "performances": [
                {
                    "symbol": p.symbol,
                    "price_change_24h": p.price_change_24h,
                    "volume_24h": p.volume_24h,
                    "current_price": p.current_price,
                    "high_24h": p.high_24h,
                    "low_24h": p.low_24h,
                    "timestamp": p.timestamp
                } for p in performances
            ],
            "market_sentiment": scanner.get_market_sentiment(performances)
        }
        
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/api/learning/insights")
async def api_learning_insights(symbol: Optional[str] = None):
    """Get learning insights from trading history"""
    try:
        from application.memory import TradingMemory
        memory = TradingMemory()
        insights = memory.get_learning_insights(symbol)
        return {"ok": True, "insights": insights}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/api/learning/events")
async def api_recent_events(symbol: Optional[str] = None, hours: int = 24):
    """Get recent trading events"""
    try:
        from application.memory import TradingMemory
        memory = TradingMemory()
        events = memory.get_recent_events(symbol, hours)
        
        return {
            "ok": True,
            "events": [
                {
                    "timestamp": e.timestamp,
                    "symbol": e.symbol,
                    "action": e.action,
                    "reason": e.reason,
                    "entry_price": e.entry_price,
                    "exit_price": e.exit_price,
                    "pnl": e.pnl,
                    "outcome": e.outcome,
                    "strategy_used": e.strategy_used,
                    "duration_minutes": e.duration_minutes
                } for e in events
            ]
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/api/learning/strategy-performance")
async def api_strategy_performance():
    """Get performance metrics for all strategies"""
    try:
        from application.memory import TradingMemory
        memory = TradingMemory()
        strategies = memory.get_top_performing_strategies(limit=10)
        
        return {
            "ok": True,
            "strategies": [
                {
                    "strategy_name": s.strategy_name,
                    "total_trades": s.total_trades,
                    "win_rate": s.win_rate,
                    "total_pnl": s.total_pnl,
                    "profit_factor": s.profit_factor,
                    "avg_win": s.avg_win,
                    "avg_loss": s.avg_loss,
                    "last_updated": s.last_updated
                } for s in strategies
            ]
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/api/portfolio/analytics")
async def api_portfolio_analytics():
    """Get comprehensive portfolio analytics"""
    try:
        cfg = Config.load(Path("config.yaml"))
        env = Settings()
        
        # This is a simplified version - in a full implementation, 
        # you'd maintain the portfolio engine as part of the bot state
        from application.memory import TradingMemory
        from application.scanner import MarketScanner
        from infrastructure.wallex.rest_client import WallexRestClient
        
        market = WallexRestClient(
            cfg.wallex.base_url,
            env.wallex_api_key,
            env.wallex_api_secret,
            env.app_user_agent,
            cfg.wallex.endpoints,
            cfg.wallex.symbol_transform,
            cfg.wallex.candle_resolution,
        )
        
        memory = TradingMemory()
        scanner = MarketScanner(market, memory)
        
        # Get portfolio config symbols or default
        symbols = getattr(cfg.portfolio, 'symbols_to_scan', ["BTC-TMN", "ETH-TMN", "DOGE-TMN"]) if cfg.portfolio else ["BTC-TMN"]
        
        performances = await scanner.scan_symbols(symbols)
        sentiment = scanner.get_market_sentiment(performances)
        insights = memory.get_learning_insights()
        
        await market.close()
        
        return {
            "ok": True,
            "analytics": {
                "market_sentiment": sentiment,
                "top_performers": [
                    {
                        "symbol": p.symbol,
                        "change_24h": p.price_change_24h,
                        "volume_24h": p.volume_24h
                    } for p in performances[:5]
                ],
                "learning_insights": insights,
                "total_symbols_scanned": len(symbols),
                "last_updated": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/analyze-pattern")
async def api_analyze_pattern(symbol: str, hours: int = 168):
    """Analyze trading patterns for a specific symbol"""
    try:
        from application.memory import TradingMemory
        memory = TradingMemory()
        analysis = memory.analyze_market_patterns(symbol, hours)
        return {"ok": True, "analysis": analysis}
    except Exception as e:
        return {"ok": False, "error": str(e)}
