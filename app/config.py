from __future__ import annotations

from pathlib import Path
from typing import Optional, List

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class RiskConfig(BaseModel):
    risk_per_trade_pct: float = Field(0.5, ge=0, le=100)
    max_daily_loss_pct: float = Field(2.0, ge=0, le=100)
    take_profit_rr: float = Field(1.5, ge=0.1)


class SymbolConfig(BaseModel):
    min_notional: float
    price_precision: int
    quantity_precision: int


class StrategyParams(BaseModel):
    rsi_period: int = 14
    rsi_buy: int = 35
    rsi_sell: int = 65
    ma_fast: int = 9
    ma_slow: int = 21
    # For additional strategies
    breakout_lookback: int = 20
    bb_period: int = 20
    bb_std: float = 2.0
    ensemble_threshold: int = 2
    # Top performer strategy params
    min_volume_24h: float = 1000000
    min_price_change: float = 5.0
    momentum_period: int = 10
    # Portfolio management
    max_active_symbols: int = 3
    performance_update_interval_min: int = 15


class StrategyConfig(BaseModel):
    name: str = "rsi_ma"
    params: StrategyParams = StrategyParams()


class LearningConfig(BaseModel):
    enabled: bool = True
    db_path: str = "data/trading_memory.db"


class PortfolioConfig(BaseModel):
    multi_symbol_trading: bool = False
    symbols_to_scan: List[str] = ["BTC-TMN", "ETH-TMN", "DOGE-TMN"]
    intelligent_allocation: bool = True
    max_active_symbols: int = 3


class WallexRateLimit(BaseModel):
    requests_per_min: int = 120
    burst: int = 20


class WallexConfig(BaseModel):
    base_url: str
    ws_url: str
    recv_window_ms: int = 5000
    rate_limit: WallexRateLimit = WallexRateLimit()
    # Optional: endpoint paths (override here if API changes)
    endpoints: dict[str, str] | None = None
    # Optional: symbol transform to match Wallex format: as_is | remove_dash | lowercase | uppercase | remove_dash_uppercase
    symbol_transform: str = Field("as_is", pattern=r"^(as_is|remove_dash|lowercase|uppercase|remove_dash_uppercase)$")
    # Optional: auth header names override
    auth_headers: dict[str, str] | None = Field(default_factory=lambda: {
        "key": "X-API-KEY",
        "sign": "X-API-SIGN",
        "ts": "X-API-TS",
    })
    # Candle resolution for history endpoint, e.g., '1', '5', '60', '4H', '1D'
    candle_resolution: str = "60"


class AppConfig(BaseModel):
    mode: str = Field("paper", pattern=r"^(paper|live)$")
    loop_interval_sec: int = 5
    symbol: str = "BTC-TMN"
    base_currency: str = "BTC"
    quote_currency: str = "TMN"
    budget: float = 1_000_000.0


class PortfolioConfig(BaseModel):
    enabled: bool = False
    symbols: list[str] = []
    multi_symbol_trading: bool = False
    symbols_to_scan: List[str] = ["BTC-TMN", "ETH-TMN", "DOGE-TMN"]
    intelligent_allocation: bool = True
    max_active_symbols: int = 3
    allocation: str = Field("equal", pattern=r"^(equal|volatility|risk-parity)$")
    # Automatic symbol selection
    selection: str = Field("manual", pattern=r"^(manual|auto)$")
    universe: list[str] = []  # candidate symbols when selection=auto
    top_k: int = 3
    lookback_candles: int = 200


class Settings(BaseSettings):
    wallex_api_key: Optional[str] = None
    wallex_api_secret: Optional[str] = None
    app_user_agent: str = "TradeBot/1.0"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


class Config(BaseModel):
    app: AppConfig
    risk: RiskConfig
    wallex: WallexConfig
    symbols: dict[str, SymbolConfig]
    strategy: StrategyConfig
    portfolio: PortfolioConfig | None = None
    learning: LearningConfig | None = None

    @staticmethod
    def load(path: Path) -> "Config":
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return Config(**data)
