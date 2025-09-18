from __future__ import annotations

import argparse
import pandas as pd

from application.strategies import RSIMAStrategy


def build_args():
    p = argparse.ArgumentParser(description="Backtest RSI+MA strategy on CSV OHLCV")
    p.add_argument("--csv", required=True)
    p.add_argument("--symbol", default="BTC-IRT")
    return p.parse_args()


def main():
    args = build_args()
    df = pd.read_csv(args.csv)

    strat = RSIMAStrategy()
    equity = 1_000_000.0
    position = 0.0
    entry = None
    for i in range(len(df)):
        if i < 50:
            continue
        window = df.iloc[: i + 1].tail(100)
        price = float(window["close"].iloc[-1])
        # simulate candle objects
        from domain.models import Candle

        candles = [
            Candle(
                timestamp=int(r["timestamp"]),
                open=float(r["open"]),
                high=float(r["high"]),
                low=float(r["low"]),
                close=float(r["close"]),
                volume=float(r.get("volume", 0.0)),
            )
            for _, r in window.iterrows()
        ]

        decision = strat.on_candles(candles, price)
        if decision.action == "buy" and position == 0:
            # buy with 10% equity
            size_quote = equity * 0.1
            qty = size_quote / price
            equity -= size_quote
            position += qty
            entry = price
        elif decision.action == "sell" and position > 0:
            equity += position * price
            position = 0
            entry = None

    if position > 0:
        # liquidate at last price
        equity += position * float(df["close"].iloc[-1])
        position = 0

    print(f"Final equity: {equity:,.0f} IRT")


if __name__ == "__main__":
    main()
