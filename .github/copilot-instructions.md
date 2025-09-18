# Copilot Instructions for TradeBot

## Project Overview
- **TradeBot** is a clean-architecture trading bot for the Wallex exchange, supporting both paper and live trading, with risk controls and a baseline RSI+MA strategy.
- The architecture is layered: `domain` (models/interfaces), `application` (services/strategies), `infrastructure` (exchange clients/adapters), `app` (CLI/config/logging), and `backtest` (backtesting harness).
- Configuration is split: secrets in `.env`, operational settings in `config.yaml`.

## Key Workflows
- **Setup (Windows PowerShell):**
  - Create venv, activate, install deps: `python -m venv .venv; . .venv\Scripts\Activate.ps1; pip install -r requirements.txt`
  - Copy config: `Copy-Item config.example.yaml config.yaml`
- **Run in Paper Mode:**
  - `python -m app.main --mode paper --symbol BTC-TMN --budget 10000000 --loop-interval 5`
- **Backtest:**
  - `python -m backtest.backtester --csv data\sample_candles.csv --symbol BTC-TMN`
- **Web Dashboard:**
  - `uvicorn web.server:app --reload --host 127.0.0.1 --port 8000` (see `web/`)

## Patterns & Conventions
- **Strategy Pattern:**
  - Strategies implement a common interface in `application/strategies.py`.
  - Add new strategies by subclassing and registering in the strategy factory.
- **Risk Management:**
  - Position sizing, stop-loss, and take-profit are enforced in `application/risk.py`.
- **Configurable Endpoints:**
  - All API endpoints and keys are set in `config.yaml` and `.env`.
- **Symbol Mapping:**
  - Symbol names (e.g., `BTC-TMN`) are mapped to Wallex format (`BTCTMN`) automatically.
- **Paper vs Live Mode:**
  - Controlled by `--mode` flag and config; paper mode simulates, live mode uses real API.

## Integration Points
- **Wallex API:**
  - REST and WebSocket clients in `infrastructure/wallex/`.
  - Auth via `x-api-key` from `.env`.
- **Web UI:**
  - FastAPI/Uvicorn server in `web/server.py`, static files in `web/static/`.

## Safety & Testing
- Never commit real API keys; use `.env`.
- Always test in paper/backtest mode before live trading.
- Validate symbol formats and min order sizes per `config.yaml`.

## Examples
- Add a new strategy: subclass in `application/strategies.py`, register in factory.
- Add a new exchange: implement adapter in `infrastructure/`, update config.

Refer to `README.md` for more details and up-to-date commands.
