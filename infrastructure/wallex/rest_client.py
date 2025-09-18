from __future__ import annotations

from typing import Iterable, List, Optional
import time

import httpx
from loguru import logger

from domain.models import Candle, Ticker
from domain.ports import MarketDataPort


class WallexRestClient(MarketDataPort):
    def __init__(self, base_url: str, api_key: Optional[str] = None, api_secret: Optional[str] = None, user_agent: str = "TradeBot/1.0", endpoints: Optional[dict[str, str]] = None, symbol_transform: str = "as_is", candle_resolution: str = "60"):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.api_secret = api_secret
        self.endpoints = endpoints or {}
        self.symbol_transform = symbol_transform
        self.resolution = str(candle_resolution or "60")
        self._client = httpx.AsyncClient(base_url=self.base_url, headers={"User-Agent": user_agent}, timeout=10.0)

    async def close(self):
        await self._client.aclose()

    async def _get_json(self, path: str, *, params: Optional[dict] = None, headers: Optional[dict] = None, retries: int = 2) -> dict:
        """GET helper with small retry for transient network issues."""
        attempt = 0
        last_exc: Exception | None = None
        while attempt <= retries:
            try:
                logger.debug("HTTP GET {path} attempt={a} params={params}", path=path, a=attempt+1, params=params)
                r = await self._client.get(path, params=params, headers=headers)
                logger.debug("HTTP GET status: {status}", status=r.status_code)
                r.raise_for_status()
                return r.json()
            except (httpx.ConnectTimeout, httpx.ReadTimeout) as e:
                last_exc = e
                backoff = min(1.0 * (2 ** attempt), 3.0)
                logger.warning("GET timeout on {path} (attempt {a}); retrying in {b}s", path=path, a=attempt+1, b=backoff)
                attempt += 1
                if attempt <= retries:
                    try:
                        import asyncio
                        await asyncio.sleep(backoff)
                    except Exception:
                        pass
                else:
                    break
            except httpx.HTTPError as e:
                # Non-timeout HTTP errors: don't retry blindly
                logger.error("GET error on {path}: {err}", path=path, err=str(e))
                raise
        # Exhausted retries
        if last_exc:
            raise last_exc
        raise RuntimeError("GET failed without explicit exception")

    async def get_latest_candles(self, symbol: str, limit: int) -> Iterable[Candle]:
        # Wallex TradingView UDF history endpoint: GET /v1/udf/history
        # Params: symbol (e.g., BTCTMN), resolution (e.g., "60" minutes per bar), from, to (unix seconds)
        s = self._transform_symbol(symbol)
        now = int(time.time())
        resolution = self.resolution  # minutes per bar or strings like '1D', '4H'
        # Compute the time window in SECONDS to request at least `limit` bars.
        # For numeric resolutions (in minutes), seconds_per_bar = int(resolution) * 60.
        try:
            seconds_per_bar = int(resolution) * 60
        except ValueError:
            # Fallback for non-numeric resolutions like '1D', '4H' if ever used
            if resolution.endswith("D"):
                seconds_per_bar = int(resolution[:-1]) * 86400
            elif resolution.endswith("H"):
                seconds_per_bar = int(resolution[:-1]) * 3600
            else:
                seconds_per_bar = 60  # default to 1 minute
        window_sec = max(1, limit) * seconds_per_bar
        params = {
            "symbol": s,
            "resolution": resolution,
            "from": str(now - window_sec),
            "to": str(now),
        }
        path = self.endpoints.get("candles", "/v1/udf/history")
        headers = {"x-api-key": self.api_key} if self.api_key else None

        def _parse_candles(data: dict) -> List[Candle]:
            t = data.get("t") or []
            o = data.get("o") or []
            h = data.get("h") or []
            l = data.get("l") or []
            c = data.get("c") or []
            v = data.get("v") or []
            candles: List[Candle] = []
            count = min(len(t), len(o), len(h), len(l), len(c), len(v))
            for i in range(count):
                try:
                    candles.append(Candle(
                        timestamp=int(t[i]),
                        open=float(o[i]),
                        high=float(h[i]),
                        low=float(l[i]),
                        close=float(c[i]),
                        volume=float(v[i]),
                    ))
                except Exception:
                    continue
            return candles

        # First try with from/to window
        logger.debug("GET candles: {path} params={params}", path=path, params=params)
        data = await self._get_json(path, params=params, headers=headers, retries=2)
        candles = _parse_candles(data)
        if len(candles) >= min(20, limit // 2 if limit > 1 else 1):
            logger.debug("Parsed candles: {n}", n=len(candles))
            return candles

        # Fallback 1: try countback without from/to (if API supports it)
        candles2: List[Candle] = []
        try:
            try_params = {"symbol": s, "resolution": resolution, "countback": str(max(10, limit))}
            logger.debug("GET candles (fallback countback): {path} params={params}", path=path, params=try_params)
            data2 = await self._get_json(path, params=try_params, headers=headers, retries=2)
            candles2 = _parse_candles(data2)
        except Exception as e:
            logger.warning("Countback fallback failed for {sym} res={res}: {err}", sym=s, res=resolution, err=e)
            candles2 = []
        if candles2:
            logger.debug("Parsed candles (countback): {n}", n=len(candles2))
            return candles2[-limit:]

        # Fallback 2: fallback to 60-minute resolution if custom resolution returns empty
        if resolution != "60":
            try_params2 = {
                "symbol": s,
                "resolution": "60",
                "from": str(now - max(1, limit) * 3600),
                "to": str(now),
            }
            logger.debug("GET candles (fallback 60m): {path} params={params}", path=path, params=try_params2)
            data3 = await self._get_json(path, params=try_params2, headers=headers, retries=2)
            candles3 = _parse_candles(data3)
            logger.debug("Parsed candles (60m): {n}", n=len(candles3))
            return candles3[-limit:]

        logger.debug("Parsed candles: {n}", n=len(candles))
        return candles

    async def get_ticker(self, symbol: str) -> Ticker:
        # Use latest trade price as ticker proxy: GET /v1/trades?symbol=SYMBOL
        path = self.endpoints.get("latest_trades", "/v1/trades")
        s = self._transform_symbol(symbol)
        headers = {"x-api-key": self.api_key} if self.api_key else None
        logger.debug("GET ticker: {path} symbol={s}", path=path, s=s)
        data = await self._get_json(path, params={"symbol": s}, headers=headers, retries=2)
        result = data.get("result", {})
        trades = result.get("latestTrades") or []
        price = 0.0
        if trades:
            try:
                # assume first is most recent
                price = float(trades[0].get("price") or 0.0)
            except Exception:
                price = 0.0
        logger.info("Ticker {sym}: price={p}", sym=symbol, p=price)
        return Ticker(symbol=symbol, price=price)

    def _transform_symbol(self, symbol: str) -> str:
        t = self.symbol_transform
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
