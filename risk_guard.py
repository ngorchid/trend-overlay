"""Shared pre-trade risk guard. Identical copy in magic-formula, trend-overlay and options-vrp.

WHY THIS EXISTS. Going from paper to real money, the dominant threat is not market risk — it is
a bug or a bad data feed producing an order nobody intended. Two silent-failure modes turned up
in a single day of testing (2026-08-10/11): off-hours yfinance chains return bid=0/ask=0 and a
constant junk IV (SPY 38d read 1.56% against 12.2% during market hours), which makes VRP hugely
negative for every name so the book reports a normal-looking "nothing passed the filters"; and a
sizing bug in magic-formula let gross reach ~2x budget. The first fails safe; the second does not.

DESIGN PRINCIPLE: this module does NOT trust the strategy. The strategy computes what it wants;
these checks independently validate it against the budget before anything reaches the broker. A
guard that re-uses the strategy's own arithmetic cannot catch the strategy's own bug.

FAIL CLOSED, BUT LOUDLY. Every rejection returns a reason so the caller can alert. A system that
silently does nothing is indistinguishable from a quiet market — which is exactly how you lose
weeks before noticing something broke. Callers must surface `Check.reason`, not just `Check.ok`.

WHAT THIS IS NOT. Not a drawdown circuit breaker (that is portfolio-level and stateful) and not a
stop-loss. Sizing and exposure caps are the real risk control; these are the failsafes for when
something is wrong.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Check:
    ok: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.ok


PASS = Check(True)


@dataclass
class RiskLimits:
    """Hard bounds, as fractions of budget unless stated. Deliberately loose — these are meant to
    catch BUGS, not to shape the strategy. A limit tight enough to bind in normal operation would
    silently distort the book, which is a worse failure than the one it prevents."""
    budget: float
    max_order_frac: float = 0.15       # one order's notional vs budget
    max_position_frac: float = 0.45    # one instrument's total notional vs budget
    max_gross_frac: float = 1.10       # all positions vs budget; trend overlay overrides (levered)
    price_collar: float = 0.10         # limit price must sit within this of the reference mark
    max_daily_move: float = 0.35       # |spot / prior close - 1| above this = suspect feed
    max_data_age_days: int = 4         # calendar days; 4 covers a long holiday weekend
    min_iv: float = 0.03               # options only: below this the chain is junk, not calm
    max_iv: float = 3.00
    halt_file: str = "HALT"            # presence of this file stops all trading

    # ---- presets. ONE set of defaults cannot serve all three books, and getting this wrong is
    # not harmless: the first version of this module rejected a single MES contract ($27,500 =
    # 28% of a $100k budget) as a fat finger, because equities-calibrated caps assume you can
    # size continuously. Futures cannot — one contract is the minimum tradeable unit and may
    # legitimately be a large slice of a small book. Use the preset that matches the instrument.
    @classmethod
    def for_equities(cls, budget: float, **kw) -> "RiskLimits":
        """Shares are divisible, so caps can be tight — nothing legitimate should approach them."""
        return cls(budget=budget, max_order_frac=0.15, max_position_frac=0.20,
                   max_gross_frac=1.10, **kw)

    @classmethod
    def for_futures(cls, budget: float, **kw) -> "RiskLimits":
        """Levered and INDIVISIBLE. Caps track the trend overlay's own per_market_cap=0.40 and
        gross_cap=3.0 with a little headroom, so the guard backstops those limits rather than
        duplicating (and contradicting) them."""
        return cls(budget=budget, max_order_frac=0.45, max_position_frac=0.50,
                   max_gross_frac=3.30, price_collar=0.05, **kw)

    @classmethod
    def for_options(cls, budget: float, **kw) -> "RiskLimits":
        """Defined-risk spreads: notional is small, but the 100x multiplier makes typos violent.
        Collar is wide because option mids move far in percentage terms on small absolute moves."""
        return cls(budget=budget, max_order_frac=0.20, max_position_frac=0.25,
                   max_gross_frac=1.10, price_collar=0.25, **kw)


# ---------------------------------------------------------------- kill switch
def halted(root: Path | str, limits: RiskLimits | None = None) -> Check:
    """Is the manual kill switch engaged? A file, not an env var, so it can be dropped in over
    SSH/RDP or a synced folder without touching the scheduler or the code."""
    name = limits.halt_file if limits else "HALT"
    f = Path(root) / name
    if f.exists():
        note = ""
        try:
            note = f.read_text().strip()[:200]
        except OSError:
            pass
        return Check(False, f"HALT file present at {f}" + (f": {note}" if note else ""))
    if os.getenv("TRADING_HALT", "").lower() in ("1", "true", "yes"):
        return Check(False, "TRADING_HALT env var set")
    return PASS


# ---------------------------------------------------------------- layer 1: data
def data_fresh(index: pd.Index, today: pd.Timestamp, limits: RiskLimits) -> Check:
    """Is the newest observation recent enough to act on?

    Staleness is the quiet killer: a frozen feed keeps returning yesterday's prices and every
    downstream calculation looks perfectly reasonable.
    """
    if index is None or len(index) == 0:
        return Check(False, "empty data index")
    last = pd.Timestamp(index[-1]).normalize()
    age = (pd.Timestamp(today).normalize() - last).days
    if age > limits.max_data_age_days:
        return Check(False, f"stale data: newest bar {last.date()} is {age}d old "
                            f"(limit {limits.max_data_age_days}d)")
    if age < 0:
        return Check(False, f"data from the FUTURE: newest bar {last.date()} > today")
    return PASS


def price_sane(ticker: str, price: float, prior: float | None, limits: RiskLimits) -> Check:
    """Is this a plausible price given the previous close?

    Catches unadjusted splits, feed glitches and wrong-instrument mix-ups — all of which produce a
    number that is perfectly valid as a float and catastrophic as a position size.
    """
    if price is None or not np.isfinite(price) or price <= 0:
        return Check(False, f"{ticker}: non-positive/NaN price {price!r}")
    if prior is not None and np.isfinite(prior) and prior > 0:
        move = abs(price / prior - 1.0)
        if move > limits.max_daily_move:
            return Check(False, f"{ticker}: price {price:,.2f} moved {move:.0%} vs prior "
                                f"{prior:,.2f} (limit {limits.max_daily_move:.0%}) — "
                                f"suspect split/feed error")
    return PASS


def chain_sane(ticker: str, atm_iv: float, bids: pd.Series | None,
               limits: RiskLimits) -> Check:
    """Is this option chain usable, or an off-hours husk?

    THE CASE THIS IS BUILT FOR (2026-08-11): outside US hours the provider returned bid=0, ask=0,
    openInterest=0 and a constant IV of 0.0156 for every strike. Nothing errored. ATM IV that low
    drives VRP = IV - RV deeply negative for every name, so the book reported "gate open but no
    name passed the VRP filters" — indistinguishable from a genuinely quiet day. Without this
    check the system can sit dead for weeks looking healthy.
    """
    if atm_iv is None or not np.isfinite(atm_iv):
        return Check(False, f"{ticker}: ATM IV is NaN — chain unusable")
    if atm_iv < limits.min_iv:
        return Check(False, f"{ticker}: ATM IV {atm_iv:.2%} below {limits.min_iv:.0%} — "
                            f"stale/off-hours chain, NOT a calm market")
    if atm_iv > limits.max_iv:
        return Check(False, f"{ticker}: ATM IV {atm_iv:.0%} above {limits.max_iv:.0%} — junk quote")
    if bids is not None and len(bids):
        b = pd.to_numeric(bids, errors="coerce")
        if (b.fillna(0) <= 0).all():
            return Check(False, f"{ticker}: every bid is zero — market closed or chain empty")
    return PASS


# ---------------------------------------------------------------- layer 2: orders
def check_order(ticker: str, side: str, qty: float, price: float, multiplier: float,
                limits: RiskLimits, *, current_position_notional: float = 0.0,
                gross_notional: float = 0.0, reference_price: float | None = None,
                max_gross_frac: float | None = None) -> Check:
    """Validate ONE order against the budget, independently of whatever the strategy computed.

    `multiplier` is 1 for shares, the contract multiplier for futures, 100 for options.
    `max_gross_frac` overrides the default for levered books (the trend overlay runs a 3x gross
    cap by design, so the unlevered 1.1x default would reject every legitimate order).
    """
    if side not in ("BUY", "SELL"):
        return Check(False, f"{ticker}: bad side {side!r}")
    if qty is None or not np.isfinite(qty) or qty <= 0:
        return Check(False, f"{ticker}: non-positive/NaN qty {qty!r}")
    if float(qty) != int(qty):
        return Check(False, f"{ticker}: fractional qty {qty}")
    if price is None or not np.isfinite(price) or price <= 0:
        return Check(False, f"{ticker}: non-positive/NaN price {price!r}")

    notional = abs(qty) * price * multiplier
    if notional > limits.max_order_frac * limits.budget:
        return Check(False, f"{ticker}: order ${notional:,.0f} = "
                            f"{notional/limits.budget:.0%} of budget, over the "
                            f"{limits.max_order_frac:.0%} single-order cap")

    after = abs(current_position_notional + (notional if side == "BUY" else -notional))
    if after > limits.max_position_frac * limits.budget:
        return Check(False, f"{ticker}: position would reach ${after:,.0f} = "
                            f"{after/limits.budget:.0%} of budget, over the "
                            f"{limits.max_position_frac:.0%} per-instrument cap")

    gcap = limits.max_gross_frac if max_gross_frac is None else max_gross_frac
    if gross_notional + notional > gcap * limits.budget:
        return Check(False, f"{ticker}: gross would reach "
                            f"${gross_notional + notional:,.0f} = "
                            f"{(gross_notional+notional)/limits.budget:.1f}x budget, over the "
                            f"{gcap:.1f}x cap")

    if reference_price is not None and np.isfinite(reference_price) and reference_price > 0:
        dev = abs(price / reference_price - 1.0)
        if dev > limits.price_collar:
            return Check(False, f"{ticker}: limit {price:,.2f} is {dev:.0%} from reference "
                                f"{reference_price:,.2f}, outside the "
                                f"{limits.price_collar:.0%} collar")
    return PASS


def check_batch(orders: list[dict], limits: RiskLimits,
                max_gross_frac: float | None = None) -> tuple[list[dict], list[str]]:
    """Validate a day's orders together, accumulating gross as it goes.

    Order-by-order checks cannot catch a batch that is individually fine and collectively absurd —
    thirty orders at 10% of budget each pass singly and blow the book together.

    Returns (accepted, rejection_reasons). Rejections are DROPPED, not raised: one bad order
    should not stop the other twenty-nine, but every reason must be surfaced by the caller.
    """
    accepted, reasons = [], []
    gross = float(sum(o.get("existing_gross", 0.0) for o in orders[:1]))
    for o in orders:
        c = check_order(o["ticker"], o["side"], o["qty"], o["price"], o.get("multiplier", 1.0),
                        limits,
                        current_position_notional=o.get("current_position_notional", 0.0),
                        gross_notional=gross, reference_price=o.get("reference_price"),
                        max_gross_frac=max_gross_frac)
        if c.ok:
            accepted.append(o)
            gross += abs(o["qty"]) * o["price"] * o.get("multiplier", 1.0)
        else:
            reasons.append(c.reason)
            logging.warning("RISK REJECT: %s", c.reason)
    return accepted, reasons
