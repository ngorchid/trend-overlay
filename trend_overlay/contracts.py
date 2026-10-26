"""Futures contract registry for the trend overlay.

Maps each market to its IBKR futures contract (standard *and* micro, where a micro exists —
micros give a retail account the granularity to size whole contracts). A yfinance ETF proxy
per market lets the trend signal be computed offline/cached; production would compute it from
the future's own history instead.

`notional_*` are *approximate* dollar values per contract. The live run values each contract from
IB's price x multiplier (`contract_prices` / `checked_values` in execution.py); these are only the
FALLBACK for a market IB cannot price, the reference for the 3x sanity band, and the values used
by offline dry runs and backtests. Refreshed 2026-10-14 from futures prices; refresh again if they
drift a lot. Rates (ZN, ZB) have no micro, so they're chunky for small accounts.
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
    # The micro's own buffer, when it settles differently from the standard contract: gold's
    # micro is the CASH-settled 1-ounce contract (1OZ), so it needs only a liquidity buffer, while
    # the physically-delivered GC keeps 30 days. None = same as notice_buffer_days.
    micro_notice_buffer_days: int | None = None

    def sym(self, use_micro: bool) -> str:
        return self.micro_symbol if (use_micro and self.micro_symbol) else self.symbol

    def mult(self, use_micro: bool) -> float:
        return self.micro_multiplier if (use_micro and self.micro_symbol) else self.multiplier

    def notional(self, use_micro: bool) -> float:
        return self.notional_micro if (use_micro and self.micro_symbol) else self.notional_std

    def buffer_days(self, use_micro: bool) -> int:
        """Days before last trade by which a position in this contract must be out."""
        if use_micro and self.micro_symbol and self.micro_notice_buffer_days is not None:
            return self.micro_notice_buffer_days
        return self.notice_buffer_days


# Core diversified basket (equity / rates / metals / energy / FX), 11 markets. Extend as desired.
# notice_buffer_days: how early to be OUT to dodge physical delivery (bonds/metals ~month;
# oil monthly so tighter but > weekly cadence; ES cash-settled so small = liquidity only).
FUTURES: list[FutureSpec] = [
    FutureSpec("equity_us", "SPY", "ES", "CME",   "USD", 50,    391_000, "MES", 5,    39_100, notice_buffer_days=5),
    FutureSpec("rates_10y", "IEF", "ZN", "CBOT",  "USD", 1000,  104_000, notice_buffer_days=25),
    # Gold's small contract is the 1-OUNCE future (1OZ, launched 2025-01-13), not the 10oz micro
    # (MGC): at ~$41k one MGC no longer fits the $30k per-market cap at the live size, while 1OZ
    # (~$4.1k) sizes in fine steps. It is CASH-settled against GC, so no delivery risk and a short
    # buffer (5d, like ES). Its last trade is the 3rd-last business day of the month BEFORE the
    # contract month; the roll logic works from IB's last-trade date, so that is handled as is.
    FutureSpec("gold",      "GLD", "GC", "COMEX", "USD", 100,   411_000, "1OZ", 1,    4_100,
               notice_buffer_days=30, micro_notice_buffer_days=5),
    FutureSpec("copper",    "CPER","HG", "COMEX", "USD", 25000, 167_500, "MHG", 2500, 16_800, notice_buffer_days=28),
    FutureSpec("oil",       "USO", "CL", "NYMEX", "USD", 1000,  95_400,  "MCL", 100,  9_540,  notice_buffer_days=14),
    FutureSpec("fx_eur",    "FXE", "6E", "CME",   "USD", 125000,139_400, "M6E", 12500,13_900, notice_buffer_days=10),
    FutureSpec("fx_aud",    "FXA", "6A", "CME",   "USD", 100000,69_600,  "M6A", 10000, 6_960,  notice_buffer_days=10),
    # "10-correlated" expansion (added 2026-07-14): 30y bond / silver / yen. Diversify the
    # existing rates/metals/FX sleeves; specs verified vs CME. NB sizing rounds to whole
    # contracts, and ZB has NO micro (~$103k/contract), so it needs a much larger budget.
    FutureSpec("rates_30y", "TLT", "ZB", "CBOT",  "USD", 1000,   102_800, notice_buffer_days=25),                 # phys. delivered; no micro
    FutureSpec("silver",    "SLV", "SI", "COMEX", "USD", 5000,   285_000, "SI",  1000,    57_000, notice_buffer_days=30),  # 1,000oz micro shares the "SI" symbol; the MULTIPLIER (1000 vs full 5000) selects it — localSymbol SILQ6, tradingClass SIL. phys. delivered
    FutureSpec("fx_jpy",    "FXY", "6J", "CME",   "USD", 12500000, 79_100, "MJY", 1250000, 7_910, notice_buffer_days=10),   # MJY = 1/10 micro (IB symbol; "M6J" does not resolve — Error 200)
    # Natural gas (added 2026-10-26, doc H.25): passed the pre-registered new-markets test
    # (algo_trading/scripts/trend_new_markets_lab.py); weekly correlation with oil only +0.18.
    # Micro Henry Hub (MNG, 1,000 MMBtu, ~$3.5k) is FINANCIALLY settled, so no delivery risk and a
    # short liquidity buffer (5d, like 1OZ). NG itself is physically delivered, hence 14d on the
    # standard contract. MNG's last trade is the 4th-last business day of the month BEFORE the
    # contract month; the roll logic works from IB's last-trade date, so that is handled as is.
    FutureSpec("natgas",    "UNG", "NG", "NYMEX", "USD", 10000,  35_000, "MNG", 1000,   3_500,
               notice_buffer_days=14, micro_notice_buffer_days=5),
]

BY_MARKET: dict[str, FutureSpec] = {s.market: s for s in FUTURES}
PROXY_ETFS: list[str] = [s.proxy_etf for s in FUTURES]
