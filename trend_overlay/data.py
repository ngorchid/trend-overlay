"""Standalone price loader (yfinance) — keeps this repo independent of the research framework.

Only what the overlay needs: adjusted-close panels for the ETF proxies. Returns the same
{field: wide DataFrame} shape the original `download_ohlcv` used, so the overlay code is unchanged.
"""
from __future__ import annotations

import pandas as pd
import yfinance as yf


def download_ohlcv(tickers, start: str, end: str | None = None) -> dict[str, pd.DataFrame]:
    tickers = list(dict.fromkeys(tickers))
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                      progress=False, threads=True)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    close = close.dropna(how="all").ffill()
    return {"adj_close": close, "close": close}
