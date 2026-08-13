"""Futures contract registry for the trend overlay.

Maps each market to its IBKR futures contract (standard *and* micro, where a micro exists —
micros give a retail account the granularity to size whole contracts). A yfinance ETF proxy
per market lets the trend signal be computed offline/cached; production would compute it from
the future's own history instead.

`notional_*` are *approximate* current dollar notionals used only for OFFLINE dry-run sizing;
when connected to IB they're replaced by the live `price × multiplier`. Refresh them if they
drift a lot. Rates (ZN) have no micro, so they're chunky for small accounts — flagged below.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FutureSpec:
    market: str            # human label / sector
    proxy_etf: str         # yfinance ticker for the (offline) trend signal
    symbol: str            # IB standard-contract root
    exchange: str
    currency: str
    multiplier: float      # $ per point, standard contract
    notional_std: float    # approx $ notional, standard contract (offline sizing)
    micro_symbol: str | None = None
    micro_multiplier: float | None = None
    notional_micro: float | None = None   # approx $ notional, micro contract
    # Calendar days before LAST-TRADE by which a position MUST be out to avoid delivery.
    # For physically-delivered bonds/metals first-notice is ~a month before last-trade, so
    # these are large; oil is monthly (tighter); ES is cash-settled (no delivery, liquidity only).
    notice_buffer_days: int = 10

    def sym(self, use_micro: bool) -> str:
        return self.micro_symbol if (use_micro and self.micro_symbol) else self.symbol

    def mult(self, use_micro: bool) -> float:
        return self.micro_multiplier if (use_micro and self.micro_symbol) else self.multiplier

    def notional(self, use_micro: bool) -> float:
        return self.notional_micro if (use_micro and self.micro_symbol) else self.notional_std


# Core diversified basket (equity / rates / metals / energy / FX). Extend as desired.
# notice_buffer_days: how early to be OUT to dodge physical delivery (bonds/metals ~month;
# oil monthly so tighter but > weekly cadence; ES cash-settled so small = liquidity only).
FUTURES: list[FutureSpec] = [
    FutureSpec("equity_us", "SPY", "ES", "CME",   "USD", 50,    275_000, "MES", 5,    27_500, notice_buffer_days=5),
    FutureSpec("rates_10y", "IEF", "ZN", "CBOT",  "USD", 1000,  112_000, notice_buffer_days=25),
    FutureSpec("gold",      "GLD", "GC", "COMEX", "USD", 100,   300_000, "MGC", 10,   30_000, notice_buffer_days=30),
    FutureSpec("copper",    "CPER","HG", "COMEX", "USD", 25000, 112_000, "MHG", 2500, 11_200, notice_buffer_days=28),
    FutureSpec("oil",       "USO", "CL", "NYMEX", "USD", 1000,  70_000,  "MCL", 100,  7_000,  notice_buffer_days=14),
    FutureSpec("fx_eur",    "FXE", "6E", "CME",   "USD", 125000,135_000, "M6E", 12500,13_500, notice_buffer_days=10),
    FutureSpec("fx_aud",    "FXA", "6A", "CME",   "USD", 100000,66_000,  "M6A", 10000, 6_600,  notice_buffer_days=10),
    # "10-correlated" expansion (added 2026-07-14): 30y bond / silver / yen. Diversify the
    # existing rates/metals/FX sleeves; specs verified vs CME. NB sizing rounds to whole
    # contracts — at BUDGET=100k these only take a position as OVERLAY_MULT scales toward 1.0
    # (and ZB has NO micro at ~$115k/contract, so it needs a larger budget than the micros).
    FutureSpec("rates_30y", "TLT", "ZB", "CBOT",  "USD", 1000,   115_000, notice_buffer_days=25),                 # phys. delivered; no micro
    FutureSpec("silver",    "SLV", "SI", "COMEX", "USD", 5000,   175_000, "SI",  1000,    35_000, notice_buffer_days=30),  # 1,000oz micro shares the "SI" symbol; the MULTIPLIER (1000 vs full 5000) selects it — localSymbol SILQ6, tradingClass SIL. phys. delivered
    FutureSpec("fx_jpy",    "FXY", "6J", "CME",   "USD", 12500000, 82_000, "MJY", 1250000, 8_200, notice_buffer_days=10),   # MJY = 1/10 micro (IB symbol; "M6J" does not resolve — Error 200)
]

BY_MARKET: dict[str, FutureSpec] = {s.market: s for s in FUTURES}
PROXY_ETFS: list[str] = [s.proxy_etf for s in FUTURES]
