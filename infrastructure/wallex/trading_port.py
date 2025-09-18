from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Optional, Dict, Any

import httpx
from loguru import logger

from app.config import Settings, WallexConfig
from domain.models import AccountBalance, ExecutionReport, Order
from domain.ports import TradingPort
from .auth import auth_headers, wallex_signed_headers


@dataclass
class _Ctx:
    base_url: str
    endpoints: dict[str, str]
    api_key: Optional[str]
    api_secret: Optional[str]
    symbol_transform: str
    header_names: dict[str, str]


class WallexTradingPort(TradingPort):
    def __init__(self, cfg: WallexConfig, env: Settings):
        self.ctx = _Ctx(
            base_url=cfg.base_url.rstrip("/"),
            endpoints=cfg.endpoints or {},
            api_key=env.wallex_api_key,
            api_secret=env.wallex_api_secret,
            symbol_transform=getattr(cfg, "symbol_transform", "as_is"),
            header_names=getattr(cfg, "auth_headers", {"key": "x-api-key", "sign": "", "ts": ""}),
        )
        self.recv_window_ms = getattr(cfg, "recv_window_ms", 5000)
        self._client = httpx.AsyncClient(base_url=self.ctx.base_url, headers={"User-Agent": env.app_user_agent}, timeout=15.0)

    async def _close(self):
        await self._client.aclose()

    async def place_order(self, order: Order) -> ExecutionReport:
        # Wallex Spot: POST /v1/account/orders with x-api-key
        path = self.ctx.endpoints.get("orders", "/v1/account/orders")
        url = path
        symbol = self._transform_symbol(order.symbol)
        # Always submit MARKET orders for immediate execution
        payload: Dict[str, Any] = {
            "symbol": symbol,
            "side": order.side.value,
            "type": "market",
            # Swagger shows string types; convert to str
            "quantity": str(order.quantity),
        }
        headers: Dict[str, str] = {}
        if self.ctx.api_key:
            headers[self.ctx.header_names.get("key", "x-api-key")] = self.ctx.api_key  # type: ignore[arg-type]
        safe_headers = {**headers}
        if self.ctx.header_names.get("key") in safe_headers:
            # mask api key in logs
            k = safe_headers[self.ctx.header_names.get("key")]
            if isinstance(k, str) and len(k) > 6:
                safe_headers[self.ctx.header_names.get("key")] = k[:3] + "***" + k[-3:]
        logger.info("POST order: {url} payload={payload} headers={headers}", url=url, payload=payload, headers=safe_headers)
        try:
            r = await self._client.post(url, json=payload, headers=headers)
            logger.debug("Order response status: {status}", status=r.status_code)
            r.raise_for_status()
            data = r.json()
        except httpx.HTTPStatusError as e:
            try:
                err = e.response.json()
            except Exception:
                err = {"message": str(e)}
            msg = err.get("message") or err.get("error") or str(e)
            logger.error("Order rejected: {msg} body={body}", msg=msg, body=err)
            return ExecutionReport(order=order, executed_qty=0.0, avg_price=order.price or 0.0, status=f"rejected: {msg}")
        # Wallex responses vary; try common fields, fallback gracefully
        d = data.get("result", data.get("data", data))
        executed_qty = float(d.get("executedQty") or d.get("filledQty") or d.get("quantity") or order.quantity)
        avg_price = float(d.get("avgPrice") or d.get("price") or (order.price or 0.0))
        status = (d.get("status") or "filled").lower()
        oid = d.get("orderId") or d.get("id")
        logger.info("Order result: status={status} executed_qty={qty} avg_price={avg} id={oid}", status=status, qty=executed_qty, avg=avg_price, oid=oid)
        return ExecutionReport(order=order, executed_qty=executed_qty, avg_price=avg_price, status=status, id=oid)

    async def get_balance(self, currency: str) -> AccountBalance:
        # Wallex balances map: GET /v1/account/balances, header x-api-key
        path = self.ctx.endpoints.get("balances", "/v1/account/balances")
        headers: Dict[str, str] = {}
        if self.ctx.api_key:
            headers[self.ctx.header_names.get("key", "x-api-key")] = self.ctx.api_key  # type: ignore[arg-type]
        safe_headers = {**headers}
        if self.ctx.header_names.get("key") in safe_headers:
            k = safe_headers[self.ctx.header_names.get("key")]
            if isinstance(k, str) and len(k) > 6:
                safe_headers[self.ctx.header_names.get("key")] = k[:3] + "***" + k[-3:]
        logger.debug("GET balances: {path} headers={headers}", path=path, headers=safe_headers)
        try:
            r = await self._client.get(path, headers=headers)
            logger.debug("Balances response status: {status}", status=r.status_code)
            r.raise_for_status()
            data = r.json()
        except httpx.HTTPStatusError:
            logger.error("Balances request failed for currency={cur}", cur=currency)
            return AccountBalance(currency=currency, total=0.0, available=0.0)

        # Response example shows: { result: { balances: { ASSET: { value, locked, ... } } } }
        d = data.get("result", data.get("data", {}))
        balances_map = d.get("balances", {}) or {}
        # Standardize currency key: try exact, upper, lower, and aliases (TMN/IRT/IRR)
        item = balances_map.get(currency) or balances_map.get(currency.upper()) or balances_map.get(currency.lower())
        if not item:
            aliases = []
            cur_up = currency.upper()
            if cur_up in ("TMN", "IRT", "IRR"):
                # Try common fiat aliases used across Iranian exchanges
                aliases = [c for c in ("TMN", "IRT", "IRR") if c != cur_up]
            for a in aliases:
                item = balances_map.get(a)
                if item:
                    logger.debug("Balance alias hit: requested {req} found under {alias}", req=currency, alias=a)
                    break
        if item:
            try:
                # Support multiple key variants
                def _first_float(keys, default=0.0):
                    for k in keys:
                        if k in item and item.get(k) is not None:
                            try:
                                return float(item.get(k))
                            except Exception:
                                continue
                    # If default is None, propagate None instead of casting
                    if default is None:
                        return None  # type: ignore[return-value]
                    return float(default)

                total = _first_float(["value", "total", "balance", "amount"], 0.0)
                locked = _first_float(["locked", "freeze", "frozen"], 0.0)
                # Some APIs expose available/free directly
                available_direct = _first_float(["available", "free"], None)
                if available_direct is not None and available_direct >= 0:  # type: ignore[operator]
                    available = available_direct
                    # If total is zero but available is set, set total = available + locked
                    if total == 0.0:
                        total = float(available) + max(locked, 0.0)  # type: ignore[arg-type]
                else:
                    available = max(total - locked, 0.0)
                logger.debug("Balance {cur}: total={tot} locked={lock} available={avail}", cur=currency, tot=total, lock=locked, avail=available)
                return AccountBalance(currency=currency, total=total, available=available)
            except Exception:
                logger.exception("Failed to parse balance item for {cur}: {itm}", cur=currency, itm=item)
        # fallback: zero
        logger.warning("Balance not found for {cur}; returning zeros", cur=currency)
        return AccountBalance(currency=currency, total=0.0, available=0.0)

    def _transform_symbol(self, symbol: str) -> str:
        t = self.ctx.symbol_transform
        if t == "as_is":
            return symbol
        if t == "remove_dash":
            return symbol.replace("-", "")
        if t == "lowercase":
            return symbol.lower()
        if t == "uppercase":
            return symbol.upper()
        if t == "remove_dash_uppercase":
            return symbol.replace("-", "").upper()
        return symbol
