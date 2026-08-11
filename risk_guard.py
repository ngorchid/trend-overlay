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


# ---------------------------------------------------------------- NAV-linked budget
def effective_budget(base: float, realized_pnl: float, unrealized_pnl: float = 0.0,
                     step: float = 0.10, max_grow: float = 3.0, warn_below: float = 0.70
                     ) -> tuple[float, str]:
    """Compound sizing off the strategy's OWN equity instead of a frozen env var.

    WHY NOT IB's NetLiquidation: three strategies share one account. NetLiq is the WHOLE
    account, so every strategy would size as if it owned all of it — a silent 3x over-allocation.
    Each strategy must compound only its own realised + unrealised P&L.

    WHY QUANTISED (`step`): a budget that moves every day makes every position target move every
    day, which pushes contract targets back and forth across the whole-contract boundary — the
    exact churn hysteresis exists to stop. Rounding equity to the nearest `step` (10% of base)
    means the budget changes in discrete jumps a few times a year instead of daily. Stateless, so
    there is nothing extra to persist or to get out of sync.

    WHY THE CAP IS ONE-SIDED: `realized_pnl` is an accounting figure produced by our own code.
    If it is ever wrong — a double-counted fill, a sign error, a corrupted state file — an
    uncapped budget turns that bug directly into position size, so growth is capped at `max_grow`.
    But there is deliberately NO FLOOR. An earlier version clipped both ways, which meant a real
    95% loss still sized at 0.5x base: the floor rounds exposure UP, forcing you to trade capital
    you no longer have. Letting the budget shrink freely is fail-safe in both readings — if the
    loss is real you de-risk correctly, and if the ledger is broken you trade less, not more.
    Deep losses should stop trading via the circuit breaker, not be papered over by a floor.

    Returns (budget, note) — the note is for the daily log, so a change in size is never silent.
    """
    if not base or base <= 0 or base != base:
        return 0.0, "invalid base budget"
    equity = base + float(realized_pnl or 0.0) + float(unrealized_pnl or 0.0)
    raw = equity / base
    q = round(raw / step) * step if step and step > 0 else raw
    r = float(max(min(q, max_grow), 0.0))          # cap growth; never floor the downside
    note = f"equity ${equity:,.0f} = {raw:.2f}x base -> sizing at {r:.2f}x (${base*r:,.0f})"
    if r < q:
        note += f" [CAPPED at {max_grow:.1f}x — check the P&L ledger for a double-counted fill]"
    if r <= warn_below:
        note += (f" [** DOWN {1-r:.0%} FROM BASE — this is a drawdown, not a sizing decision; "
                 f"consider halting **]")
    return base * r, note


# ---------------------------------------------------------------- layer 3: circuit breaker
@dataclass
class BreakerLevels:
    """Drawdown-from-peak thresholds. Set these BEYOND the strategy's expected drawdown.

    This is the part people get wrong. The trend overlay's backtested maxDD is -20%, so a halt at
    -20% would have fired at the historical worst point — capitulating at the bottom, converting a
    recovered drawdown into a realised loss. That is the same mistake as the 2x stop we removed
    from options-vrp.

    A circuit breaker is an OPERATIONAL failsafe for "the model is broken, the data is wrong,
    there is a bug" — NOT a risk-management tool for normal losses. Normal losses are handled by
    SIZING. So the thresholds must sit outside the range the strategy is expected to produce, and
    tripping one should be treated as evidence that something is wrong, not as bad luck.
    """
    derisk: float = 0.15        # halve exposure
    reduce_only: float = 0.25   # no new risk; closing trades only
    halt: float = 0.35          # stop entirely, manual restart


def circuit_breaker(equity: float, peak_equity: float,
                    levels: BreakerLevels | None = None) -> tuple[str, float, str]:
    """(level, exposure_scale, reason). Levels: ok | derisk | reduce_only | halt.

    `exposure_scale` multiplies target sizes; `reduce_only` and `halt` both return 0.0 but mean
    different things to the caller — reduce_only still permits CLOSING trades, halt permits
    nothing. The caller must honour that distinction, because a breaker that blocks closing
    orders traps you in the position it is trying to protect you from.
    """
    lv = levels or BreakerLevels()
    if not peak_equity or peak_equity <= 0 or equity != equity:
        return "ok", 1.0, ""
    dd = 1.0 - (equity / peak_equity)
    if dd >= lv.halt:
        return "halt", 0.0, (f"drawdown {dd:.1%} from peak ${peak_equity:,.0f} >= halt "
                             f"{lv.halt:.0%} — STOPPED, manual restart required")
    if dd >= lv.reduce_only:
        return "reduce_only", 0.0, (f"drawdown {dd:.1%} >= {lv.reduce_only:.0%} — "
                                    f"closing trades only, no new risk")
    if dd >= lv.derisk:
        return "derisk", 0.5, f"drawdown {dd:.1%} >= {lv.derisk:.0%} — exposure halved"
    return "ok", 1.0, ""


def peak_equity(history: list[dict], base: float, key: str = "total_pnl") -> float:
    """Highest equity ever reached, from the strategy's own snapshot history.

    Uses the strategy's OWN P&L series, not the account — same reason as `effective_budget`.
    Falls back to `base` so a fresh book with no history cannot show a drawdown.
    """
    if not history:
        return float(base)
    vals = [float(base) + float(h.get(key, 0.0) or 0.0) for h in history]
    return max([float(base)] + vals)


# ---------------------------------------------------------------- shared-account margin ceiling
@dataclass
class MarginLimits:
    """Margin-utilisation thresholds, as a fraction of net liquidation value.

    THE RISK THIS ADDRESSES is specific to running several strategies in ONE IB account, which is
    the right choice at small size: the trend overlay is a margin overlay on shared collateral
    ($311,900 of notional against ~$10,900 of margin), so separating it into its own account would
    require ~$27-33k of dedicated idle cash — unaffordable below roughly $250k of total capital.

    The price of sharing is that the equity book must sit in a MARGIN account, where a severe
    drawdown can force-liquidate stocks; and the collateral is CORRELATED with what it backs — in
    a crash equities fall while futures margin requirements RISE, so headroom shrinks exactly when
    the requirement grows. No code removes that. This ceiling just makes sure we notice early and
    stop adding to it, rather than discovering it at the liquidation.

    Same shape as the treasury system's MAX_MARGIN_FRAC=0.25, which is already proven in this book.
    """
    max_new_risk: float = 0.25   # above this, no NEW positions
    derisk: float = 0.40         # above this, halve target exposure
    halt: float = 0.60           # above this, closing trades only


def margin_check(margin_used: float, net_liq: float,
                 limits: MarginLimits | None = None) -> tuple[str, float, str]:
    """(level, exposure_scale, reason). Levels: ok | no_new_risk | derisk | halt.

    Note this is ACCOUNT-WIDE: in a shared account every strategy sees the total, so one strategy
    can be blocked by another's usage. That is correct — the constraint really is account-wide,
    and being blocked is strictly better than being liquidated.

    As everywhere else here, `halt` must still permit CLOSING trades. A margin guard that blocks
    the orders which would REDUCE margin is the worst possible failure mode.
    """
    lim = limits or MarginLimits()
    if not net_liq or net_liq <= 0 or margin_used != margin_used:
        # Distinct from "ok" ON PURPOSE: unknown must be visible in the log, not indistinguishable
        # from healthy. Scale stays 1.0 (fail-OPEN) because the order and gross caps already bound
        # notional, and therefore bound margin indirectly — and because blocking everything on a
        # transient API gap would also block the contract ROLLS that prevent physical delivery.
        # The caller may still choose to treat "unknown" as blocking.
        return "unknown", 1.0, "MARGIN DATA UNAVAILABLE — ceiling not enforced this run"
    used = margin_used / net_liq
    if used >= lim.halt:
        return "halt", 0.0, (f"margin {used:.0%} of NAV >= {lim.halt:.0%} — CLOSING TRADES ONLY; "
                             f"liquidation risk")
    if used >= lim.derisk:
        return "derisk", 0.5, f"margin {used:.0%} of NAV >= {lim.derisk:.0%} — exposure halved"
    if used >= lim.max_new_risk:
        return "no_new_risk", 0.0, (f"margin {used:.0%} of NAV >= {lim.max_new_risk:.0%} — "
                                    f"holding existing positions, no new ones")
    return "ok", 1.0, ""


# ---------------------------------------------------------------- surfacing alerts
class AlertCollector(logging.Handler):
    """Collects WARNING+ records so the daily email can carry them.

    WHY THIS EXISTS: every guard in this module logs its rejections, and on the Windows box the
    scheduled task redirects stdout to `results\\paper\\run.log`. Nobody opens that file daily. So
    a stale feed, a rejected order or a margin ceiling hit would sit in a log while the email
    reported a perfectly normal-looking book — the exact silent failure the guards were written to
    prevent. An alert that is not delivered is not an alert.

    Attach once at startup; read `.records` when building the email, and put a marker in the
    SUBJECT so it is visible without opening anything.
    """

    def __init__(self, level: int = logging.WARNING) -> None:
        super().__init__(level)
        self.records: list[tuple[str, str]] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.records.append((record.levelname, record.getMessage()))
        except Exception:  # noqa: BLE001 — a broken alert must never break the run
            pass

    @property
    def worst(self) -> str | None:
        if any(lv in ("ERROR", "CRITICAL") for lv, _ in self.records):
            return "ERROR"
        return "WARN" if self.records else None

    def html(self) -> str:
        """Alert block for the top of the email; empty string when there is nothing to say."""
        if not self.records:
            return ""
        rows = "".join(
            f"<tr><td style='padding:2px 8px;color:#b00'><b>{lv}</b></td>"
            f"<td style='padding:2px 8px'>{msg}</td></tr>" for lv, msg in self.records)
        return ("<div style='border:2px solid #b00;padding:8px;margin:8px 0'>"
                f"<b style='color:#b00'>{len(self.records)} ALERT(S) THIS RUN</b>"
                f"<table style='font-family:monospace;font-size:12px'>{rows}</table></div>")


def install_alert_collector(level: int = logging.WARNING) -> AlertCollector:
    """Attach a collector to the root logger and return it."""
    h = AlertCollector(level)
    logging.getLogger().addHandler(h)
    return h
