# TradeBot (Wallex)

A clean-architecture trading bot skeleton targeting Wallex exchange with fully-automatic operation, paper/live modes, risk controls, and a baseline RSI+MA strategy.

Note: Endpoints, auth headers, and some constraints are parameterized in `config.yaml` because Wallex API details can change; adjust them per the official PDF reference and your API dashboard.

## Features
- Clean architecture (domain, application, infrastructure, app)
- Paper mode (no real orders) and Live mode (real API once configured)
- Risk management: position sizing by risk %, stop-loss / take-profit targets
- Strategy interface + baseline RSI+MA strategy
- Backtesting harness (CSV OHLCV)
- WebSocket/REST client stubs for Wallex (configurable base URLs and auth)

## Quickstart (Windows PowerShell)

1) Create & activate a virtual env, install dependencies:

```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2) Copy config and env templates, then edit:

```powershell
Copy-Item config.example.yaml config.yaml
Copy-Item .env.example .env
```

3) Run in paper mode (no real orders):

```powershell
python -m app.main --mode paper --symbol BTC-TMN --budget 10000000 --loop-interval 5
```

- `--budget` is in quote currency units (e.g., IRT). Use a small budget to start.
- For backtest with a CSV of candles (timestamp,open,high,low,close,volume):

```powershell
python -m backtest.backtester --csv data\sample.csv --symbol BTC-TMN
```

4) When ready for Live mode (be careful):
- Put your API key in `.env` as `WALLEX_API_KEY` (x-api-key)
- Set `mode: live` and verify `base_url`/`ws_url` in `config.yaml` (now pointing to api.wallex.ir)
- Verify min order sizes and symbol conventions (we map `BTC-TMN` → `BTCTMN` automatically)

```powershell
python -m app.main --mode live --symbol BTC-TMN --loop-interval 5
```

5) Web Dashboard (One‑Click Start/Stop + Charts)

```powershell
uvicorn web.server:app --reload --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` in your browser. Choose `mode`, optional `symbol`, `budget` (quote), and `interval (sec)`, then click `Start` for a one‑click run. Charts show recent price and equity; the table lists recent trades. Click `Check Live Balance` to verify your key before live trading.

## Configuration
- `.env`: secrets like `WALLEX_API_KEY`
- `config.yaml`: non-secret settings, endpoints, risk, symbols, rate-limit

## Important Assumptions (adjust per Wallex docs)
- For the endpoints implemented here, only `x-api-key` is required by Swagger.
- Base REST URL and WS URL are pre-set to Wallex official servers in `config.example.yaml`.
- Min order notional/size varies by market; set per-symbol in `config.yaml`

## Safety Notes
- Trading is risky. Paper-trade and backtest thoroughly before going live.
- Never commit real keys. Use `.env` and keep it private.
- Start with very small size and strict risk limits.
 - In live mode: verify symbol format, min notional, and available balances; begin with the minimum allowed size.

## Project Structure
- `app/`: CLI, config, logging bootstrap
- `domain/`: models and interfaces (ports)
- `application/`: services and strategies (use-cases)
- `infrastructure/`: exchange clients, execution, repositories (adapters)
- `backtest/`: simple backtester

## Troubleshooting
- If you see auth/permission errors in live mode, re-check headers per the latest PDF and dashboard settings.
- For pandas/NumPy versions on Windows, use Python 3.11+.
- Use a stable internet and small loop interval (>= 3-5s) to respect rate limits.