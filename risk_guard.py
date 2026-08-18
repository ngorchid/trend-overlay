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

import json
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
        # bool() of the raw value, because Python REQUIRES __bool__ to return a real bool and
        # raises TypeError on a numpy.bool_. Guard conditions are routinely numpy comparisons
        # (np.isfinite, a Series test, an ndarray element), so without this coercion a guard
        # built from one would raise at its own `if not check:` site — killing the run it exists
        # to protect, which is strictly worse than the risk it was watching for. Found 2026-08-14
        # while writing the sizing suite.
        return bool(self.ok)


PASS = Check(True)


def _num(x) -> float | None:
    """Coerce to a finite float, or None. Never raises.

    Every numeric guard takes values that ultimately come from a JSON state file or a broker
    response, so a NaN, a None or a string can arrive without warning. A guard that raises on one
    kills the run it exists to protect -- which is strictly worse than the risk it was watching
    for. Returning None lets each guard decide what "no information" means for it.
    """
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


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
    # 5%, not 3%. Measured on the off-hours chain of 2026-08-11: most names returned a constant
    # 1.56% but XOM and SBUX returned 3.13% -- the junk values are quantised, and a 3% floor let
    # the 3.13% ones through. A REAL 36-day ATM IV below 5% does not occur for these underlyings
    # (SPY's calmest ever is ~6-7%, TLT ~7%, single names rarely under 10%), so 5% cannot bind on
    # good data while catching the whole junk cluster.
    min_iv: float = 0.05               # options only: below this the chain is junk, not calm
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
HALT_NONE, HALT_NEW, HALT_ALL = "none", "new_risk", "all"


def halt_state(root: Path | str) -> tuple[str, str]:
    """(mode, reason). Modes: none | new_risk | all.

    TWO FILES, deliberately, because "stop trading" has two meanings and conflating them is
    dangerous:

      HALT      -> NEW_RISK. Open nothing new; keep MANAGING what is already on — profit targets,
                   time stops, rolls and delivery closes all still run. This is the one you want
                   almost always: "something looks wrong, stop adding until I have looked."
      HALT_ALL  -> ALL. The run exits immediately having done nothing.
                   ⚠ THIS ALSO BLOCKS DELIVERY CLOSES. A physically-delivered contract (ZB, ZN,
                   SIL) left past its notice date goes to DELIVERY. Only use it when you can watch
                   the account, and clear it before any notice date.

    A FILE rather than an env var so it can be dropped in from a phone via a synced folder,
    without touching the scheduler, the code, or a remote session. Whatever text the file contains
    is echoed into the alert, so leave yourself a note about why.
    """
    r = Path(root)
    for name, mode in ((r / "HALT_ALL", HALT_ALL), (r / "HALT", HALT_NEW)):
        if name.exists():
            note = ""
            try:
                note = name.read_text().strip()[:200]
            except OSError:
                pass
            return mode, f"{name.name} present" + (f": {note}" if note else "")
    env = os.getenv("TRADING_HALT", "").strip().lower()
    if env in ("all", "2"):
        return HALT_ALL, "TRADING_HALT=all"
    if env in ("1", "true", "yes", "new", "new_risk"):
        return HALT_NEW, f"TRADING_HALT={env}"
    return HALT_NONE, ""


def halted(root: Path | str, limits: RiskLimits | None = None) -> Check:
    """Back-compat wrapper: False (not ok) when ANY halt is in force."""
    mode, why = halt_state(root)
    return PASS if mode == HALT_NONE else Check(False, why)


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


def stale_columns(frame, today: pd.Timestamp, limits: RiskLimits,
                  ) -> tuple[dict[str, int], Check]:
    """Per-COLUMN staleness: {column -> days since it last moved}, and a Check over the frame.

    `data_fresh` only inspects the INDEX, so it answers "did the panel update?" — not "did this
    market's price update?". A single dead or frozen column leaves the index perfectly current
    and every downstream number plausible. Found 2026-08-16: one market's feed dying was
    invisible to every guard, and reached sizing as a legitimate signal.

    TWO FAILURE MODES, deliberately both:
      ABSENT  the column's last non-NaN observation is old (provider dropped the ticker).
      FROZEN  the column still reports, but the VALUE has not changed. This is the dangerous
              one: prices are valid, just repeated, so nothing is NaN and nothing errors —
              while realised vol collapses toward zero and inverse-vol sizing divides by it.

    Returns the offenders rather than a verdict on the whole run, because the right response is
    per-market (hold that one, keep trading the rest), not a global halt.
    """
    out: dict[str, int] = {}
    if frame is None or getattr(frame, "empty", True):
        return out, Check(False, "empty price frame")
    now = pd.Timestamp(today).normalize()
    for col in frame.columns:
        ser = frame[col].dropna()
        if ser.empty:
            out[col] = 10_000
            continue
        # Last date the value actually MOVED. A column repeating one price is stale even though
        # its last observation is today.
        changed = ser[ser.diff().fillna(1.0) != 0]
        last = pd.Timestamp(changed.index[-1] if len(changed) else ser.index[0]).normalize()
        age = int((now - last).days)
        if age > limits.max_data_age_days:
            out[col] = age
    if out:
        worst = ", ".join(f"{c} {a}d" for c, a in sorted(out.items(), key=lambda kv: -kv[1]))
        return out, Check(False, f"stale columns (limit {limits.max_data_age_days}d): {worst}")
    return out, PASS


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

    ⚠ NOT WIRED ANYWHERE (audited 2026-08-13). The batch property it provides is already achieved
    in every strategy by passing a RUNNING `gross_notional` into per-order `check_order` calls:
    magic-formula passes `state.positions_value_usd(...)`, trend accumulates `gross_seen` inside
    `execute()`, options-vrp passes the summed max-loss of open spreads. Kept because it is
    tested and is the cleaner API if a caller ever has the full day's orders up front — but do
    not assume it is protecting anything today.

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
    b = _num(base)
    if not b or b <= 0:
        return 0.0, "invalid base budget"
    base = b
    # A NaN anywhere in the P&L ledger must NOT propagate: `round(nan)` raises, and this runs at
    # CONFIG time, so it would kill the run before any guard, order or email. One bad fill price
    # coerced into realized_pnl would poison state.json permanently.
    rp, up = _num(realized_pnl), _num(unrealized_pnl)
    if rp is None or up is None:
        return base, (f"P&L not finite (realised={realized_pnl!r}, unrealised={unrealized_pnl!r})"
                      f" -- sizing at BASE ${base:,.0f}; CHECK THE LEDGER")
    equity = base + rp + up
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
    SIZING.

    ⚠ HONEST CALIBRATION NOTE. Only `reduce_only` and `halt` sit beyond the expected range;
    trend's backtested maxDD is −20%, so `derisk` at 15% WOULD have fired during the historical
    worst drawdown. That is deliberate for a soft, recoverable response (halve exposure), but it
    means derisk is a NORMAL-OPERATION response, not a failsafe. Do not read all three levels the
    same way. These three numbers are JUDGMENT CALLS, unlike the hysteresis band (backtested,
    sub-period robust) or min_iv (bracketed by observed junk vs historical minimum IV).
    """
    derisk: float = 0.15        # halve exposure
    reduce_only: float = 0.25   # no new risk; closing trades only
    halt: float = 0.35          # stop entirely, manual restart

    # In SIGMAS of the strategy's own annual vol. The defaults above ARE these multiples at
    # trend's 12.4% vol -- which is the whole problem with using them everywhere.
    SIGMAS = (1.2, 2.0, 2.8)

    @classmethod
    def from_vol(cls, ann_vol: float | None, sigmas: tuple = SIGMAS,
                 lo: float = 0.10, hi: float = 0.70) -> "BreakerLevels":
        """Levels scaled to the strategy's OWN volatility.

        ⚠ A FIXED PERCENTAGE MEANS DIFFERENT THINGS AT DIFFERENT VOLATILITIES, and using one set
        everywhere was measurably harmful. Backtested 2026-08-13: at 12.4% vol (trend) 15/25/35
        is 1.2σ/2.0σ/2.8σ and costs 0.01 Sharpe. At 21.6% vol (the enhanced magic formula) the
        same percentages are 0.7σ/1.2σ/1.6σ — routine moves, not failures — and cost **0.34
        Sharpe and 7.5%/yr**, with the capitulation test showing 13 of 13 triggers followed by a
        POSITIVE 63 days averaging +9%. It fired at the bottom every single time.

        Vol-scaled, trend keeps 15/25/35 and magic-formula becomes 26/43/61; both then cost ~0.
        Bounds guard against a degenerate vol estimate turning the breaker into a hair trigger
        (lo) or disabling it entirely (hi).
        """
        if not ann_vol or ann_vol != ann_vol or ann_vol <= 0:
            return cls()                      # no estimate -> the documented defaults
        d, r, h = (float(np.clip(m * ann_vol, lo, hi)) for m in sigmas)
        return cls(derisk=d, reduce_only=max(r, d + 0.01), halt=max(h, r + 0.01))


def blended_vol(history, base: float, prior: float, key: str = "total_pnl",
                absolute: bool = False, prior_obs: int = 250) -> float:
    """Live realised vol shrunk toward a BACKTEST PRIOR by how much history exists.

    TWO PROBLEMS THIS SOLVES, both live rather than theoretical:

    COLD START. Below the min_obs cut, `realised_vol` returns None and the breaker fell back to
    the generic 15/25/35 — which was measured as HARMFUL on a 21.6%-vol equity book (−0.34
    Sharpe, fired at the bottom 13 of 13 times). A new book would therefore run its first ~3
    months on precisely the thresholds we know are wrong for it, which is the window you are
    watching most closely.

    STRUCTURAL CHANGE. The trend overlay's OVERLAY_MULT went 0.5 -> 1.0 on 2026-08-13, doubling
    its exposure and so its vol. Its recorded history still describes the half-sized book, so a
    pure live estimate understates the new vol and sets the thresholds too tight until the
    history turns over. **When a config change materially alters a strategy's risk, UPDATE ITS
    PRIOR — the history will not tell you for months.**

    Shrinkage weight n/(n+prior_obs): at 60 observations the prior carries ~80%, at 250 half, at
    1000 about a fifth. No cliff at the min_obs boundary.
    """
    live = realised_vol(history, base, key=key, absolute=absolute, min_obs=20)
    if live is None or not prior or prior <= 0:
        return float(live if live is not None else (prior or 0.0))
    n = len([h for h in history]) if history else 0
    w = n / (n + float(prior_obs))
    return float(w * live + (1.0 - w) * prior)


def realised_vol(history, base: float, key: str = "total_pnl", absolute: bool = False,
                 min_obs: int = 60) -> float | None:
    """Annualised vol of the strategy's OWN equity curve, or None if too little history.

    None on purpose: with a handful of observations the estimate is noise, and scaling the
    breaker off noise is worse than using the documented defaults.
    """
    if not history or len(history) < min_obs:
        return None
    vals = []
    for h in history:
        v = h.get(key) if isinstance(h, dict) else (h[1] if len(h) > 1 else None)
        if v is None or v != v:
            continue
        vals.append(float(v) if absolute else float(base) + float(v))
    if len(vals) < min_obs:
        return None
    eq = pd.Series(vals, dtype=float)
    r = eq.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) < min_obs - 1 or r.std() == 0:
        return None
    return float(r.std() * np.sqrt(252))


def circuit_breaker(equity: float, peak_equity: float,
                    levels: BreakerLevels | None = None) -> tuple[str, float, str]:
    """(level, exposure_scale, reason). Levels: ok | derisk | reduce_only | halt.

    `exposure_scale` multiplies target sizes; `reduce_only` and `halt` both return 0.0 but mean
    different things to the caller — reduce_only still permits CLOSING trades, halt permits
    nothing. The caller must honour that distinction, because a breaker that blocks closing
    orders traps you in the position it is trying to protect you from.
    """
    lv = levels or BreakerLevels()
    eq, pk = _num(equity), _num(peak_equity)
    if eq is None or pk is None or pk <= 0:
        return "ok", 1.0, ""
    dd = 1.0 - (eq / pk)
    if dd >= lv.halt:
        return "halt", 0.0, (f"drawdown {dd:.1%} from peak ${peak_equity:,.0f} >= halt "
                             f"{lv.halt:.0%} — STOPPED, manual restart required")
    if dd >= lv.reduce_only:
        return "reduce_only", 0.0, (f"drawdown {dd:.1%} >= {lv.reduce_only:.0%} — "
                                    f"closing trades only, no new risk")
    if dd >= lv.derisk:
        return "derisk", 0.5, f"drawdown {dd:.1%} >= {lv.derisk:.0%} — exposure halved"
    return "ok", 1.0, ""


def peak_equity(history, base: float, key: str = "total_pnl",
                absolute: bool = False) -> float:
    """Highest equity ever reached, from the strategy's own snapshot history.

    ⚠ THE THREE STATES STORE DIFFERENT THINGS, and getting this wrong silently invents a
    drawdown or hides one:
        magic-formula  {"date","nav"}        -> ABSOLUTE NAV        -> absolute=True
        trend-overlay  {"date","total_pnl"}  -> P&L delta vs base   -> absolute=False
        options-vrp    (date, total_pnl)     -> P&L delta, a TUPLE  -> absolute=False

    With `absolute=False` the value is added to `base`; with True it is the equity itself.
    Accepts dicts or tuples. Falls back to `base`, so a fresh book cannot show a drawdown.

    Uses the strategy's OWN series, never the account — same reason as `effective_budget`:
    three strategies share one IB account.
    """
    if not history:
        return float(base)
    vals = []
    for h in history:
        v = h.get(key) if isinstance(h, dict) else (h[1] if len(h) > 1 else None)
        if v is None or v != v:
            continue
        vals.append(float(v) if absolute else float(base) + float(v))
    return max([float(base)] + vals)


# ---------------------------------------------------- shared-account capital allocation
@dataclass(frozen=True)
class Allocation:
    """One strategy's sizing base, as a FRACTION of account NetLiquidation.

    `peak_margin_coef` is that strategy's WORST-CASE maintenance margin expressed as a fraction
    of its OWN budget. These are structural, not estimates:

      magic-formula  0.25   Reg-T maintenance on long stock is 25% of position value, and
                            `_gross_scalar` is clipped to <=1 so gross never exceeds 1.0x budget.
      options-vrp    0.18   max_positions (6) x risk_per_trade (3%). A defined-risk spread's
                            margin IS its max loss, so this is a hard ceiling, not a typical case.
      trend-overlay  0.109  ~3.5% SPAN on ~3.12x budget of notional ($311,900 against ~$10,900).

    Multiply coef by fraction to get each sleeve's peak margin as a share of NAV; the sum is what
    the account must carry simultaneously.
    """
    fraction: float
    peak_margin_coef: float
    note: str = ""


# WHY THIS EXISTS. Until 2026-08-14 each strategy sized off its OWN independently-configured
# budget -- magic-formula off its own NAV, trend off BUDGET=100000, options-vrp off
# BASE_BUDGET=75000 -- with no strategy aware of any other. Those three numbers summed to $225k of
# sizing base on a $50k account and would have put total maintenance at 74% of NAV, tripping
# `no_new_risk` on an ordinary day. Nothing in the system could see that, because the only shared
# mechanism is a BACKSTOP that blocks, not an allocator that reserves.
#
# ⚠ REG-T DOES NOT NET THESE. Long-stock maintenance and a short put spread's max loss ADD; they
# do not offset even though one arguably hedges the other. Futures sit in a separate SPAN segment.
# Cross-margining requires Portfolio Margin ($110k minimum at IB), so below that the sum is simply
# additive and this table is the real constraint, not a conservative one.
#
# Fractions of 1.0 are deliberate and are NOT "3x over-allocation": the strategies share
# COLLATERAL rather than carving up cash, so what must be bounded is the summed MARGIN (53.9% of
# NAV here), not the summed budget. `effective_budget`'s docstring warns against sizing off
# NetLiq precisely because it was unallocated; making the fraction explicit and validating the
# margin sum is the principled version of the same idea.
# ⚠ SCOPE: these THREE sleeves only. Any other strategy trading the same IB account draws on the
# same collateral without appearing here, so the cushion below is an upper bound on what is
# actually free. Add it to this table before relying on the number.
ALLOCATIONS: dict[str, Allocation] = {
    "magic-formula": Allocation(1.00, 0.25, "Reg-T 25% of gross; gross <= 1.0x budget"),
    "trend-overlay": Allocation(1.00, 0.109, "~3.5% SPAN on ~3.12x budget notional"),
    "options-vrp": Allocation(1.00, 0.18, "max_positions 6 x risk_per_trade 3% -- structural"),
}
MIN_ALLOCATION_CUSHION = 0.40

# Quantisation anchor for `allocated_budget` — the account's nominal size, NOT a cap. Budgets are
# `fraction x NetLiq` rounded to 10% steps of `fraction x NOMINAL_NAV`, so this only sets where the
# steps fall. It does bound growth at `max_grow` (3x) as a guard against a corrupted P&L ledger,
# so RAISE IT once the account is durably past ~2x, or budgets will silently stop tracking NetLiq.
NOMINAL_NAV = 50_000.0


def allocation_cushion(allocations: dict[str, Allocation] | None = None) -> float:
    """Fraction of NAV left free once every strategy is at its PEAK margin simultaneously."""
    a = allocations if allocations is not None else ALLOCATIONS
    return 1.0 - sum(x.fraction * x.peak_margin_coef for x in a.values())


def check_allocations(allocations: dict[str, Allocation] | None = None,
                      min_cushion: float = MIN_ALLOCATION_CUSHION) -> Check:
    """Do the configured fractions leave enough room when everything is at peak at once?

    Deliberately checked against SIMULTANEOUS peaks. The three sleeves are not independent —
    options-vrp runs +0.23 tail-conditional to magic-formula, and in a crash equities fall while
    futures margin requirements RISE — so the cushion is thinnest exactly when all three want
    capital. Assuming the peaks are staggered is the assumption that fails when it matters.
    """
    a = allocations if allocations is not None else ALLOCATIONS
    for name, x in a.items():
        if not (0.0 <= x.fraction) or not (0.0 <= x.peak_margin_coef):
            return Check(False, f"{name}: negative fraction/coefficient")
    c = allocation_cushion(a)
    if c < min_cushion:
        worst = ", ".join(f"{n} {x.fraction * x.peak_margin_coef:.1%}" for n, x in a.items())
        return Check(False, f"allocations leave only {c:.1%} cushion at simultaneous peak "
                            f"(floor {min_cushion:.0%}) — {worst}")
    return PASS


def allocated_budget(strategy: str, net_liq: float | None, nominal_nav: float,
                     allocations: dict[str, Allocation] | None = None,
                     step: float = 0.10, max_grow: float = 3.0) -> tuple[float, str]:
    """(budget, note) — this strategy's sizing base as its share of the LIVE account.

    Replaces a hand-set per-strategy BUDGET. Because the fractions live in one table whose margin
    sum is validated, the sleeves can no longer be resized independently into a combination that
    does not fit; and each scales with the account automatically, so options-vrp reaches its
    measured $75-100k plateau as NAV grows without anyone editing a file.

    Quantised through `effective_budget` (the anchor being `fraction x nominal_nav`, the growth
    term `fraction x (net_liq - nominal_nav)`), so this inherits its step, its one-sided growth
    cap and its no-floor-on-losses behaviour rather than reimplementing them. It does NOT compound
    on the strategy's own realised P&L: NetLiq already contains it, and adding it again would
    double-count.

    `net_liq` unavailable -> falls back to the nominal share and says so, rather than sizing off
    a number it does not have.
    """
    a = allocations if allocations is not None else ALLOCATIONS
    alloc = a.get(strategy)
    if alloc is None:
        return 0.0, f"no allocation configured for {strategy!r} — refusing to size"
    base = alloc.fraction * float(nominal_nav)
    nl = _num(net_liq)
    if nl is None or nl <= 0:
        return base, (f"NETLIQ UNAVAILABLE — sizing at the nominal share "
                      f"{alloc.fraction:.0%} x ${nominal_nav:,.0f} = ${base:,.0f}")
    budget, note = effective_budget(base, alloc.fraction * (nl - float(nominal_nav)),
                                    step=step, max_grow=max_grow)
    return budget, f"{alloc.fraction:.0%} of NetLiq ${nl:,.0f}: {note}"


# ------------------------------------------------------- shared-account liquidity floor
@dataclass
class MarginLimits:
    """LIQUIDITY-CUSHION thresholds, as a fraction of net liquidation value.

    Cushion = ExcessLiquidity / NetLiquidation — how far the account is from a forced
    liquidation, which happens when ExcessLiquidity reaches zero. HIGH is safe; these are FLOORS,
    not ceilings.

    ⚠ WHY NOT GROSS MAINTENANCE MARGIN (the 2026-08-14 fix). This previously measured
    MaintMarginReq / NetLiquidation against ceilings of 25/40/60%. Reg-T maintenance on long
    stock is 25% of position value REGARDLESS OF LEVERAGE, so a fully-invested, entirely
    unborrowed equity book sits at exactly 25% and tripped `no_new_risk` — blocking every new buy
    in normal operation while sells continued, bleeding the book toward cash. Magic-formula
    targets gross 1.0x, so that was its steady state, not an edge case. The old levels were
    calibrated on the trend overlay's FUTURES margin, where usage does track leverage; for a cash
    equity book it measured "how invested am I", which is not a risk at all.

    ExcessLiquidity is the quantity an actual liquidation is measured against, and it is
    leverage-aware by construction: an unlevered equity book has a ~75% cushion, while a book
    levered to the edge has ~0 regardless of what asset class produced it.

    THE RISK THIS ADDRESSES is specific to running several strategies in ONE IB account, which is
    the right choice at small size: the trend overlay is a margin overlay on shared collateral
    ($311,900 of notional against ~$10,900 of margin), so separating it into its own account would
    require ~$27-33k of dedicated idle cash — unaffordable below roughly $250k of total capital.

    The price of sharing is that the equity book must sit in a MARGIN account, where a severe
    drawdown can force-liquidate stocks; and the collateral is CORRELATED with what it backs — in
    a crash equities fall while futures margin requirements RISE, so headroom shrinks exactly when
    the requirement grows. No code removes that. This floor just makes sure we notice early and
    stop adding to it, rather than discovering it at the liquidation.

    Calibration: unlevered equities ~0.75, the intended three-strategy steady state ~0.52
    (25% equity maintenance + ~5% futures + 18% VRP), so 0.30 leaves real room before it speaks.
    At 0.10 a single 10% adverse move wipes the cushion, which is why that one is `halt`.
    """
    min_cushion_new_risk: float = 0.30   # below this, no NEW positions
    min_cushion_derisk: float = 0.20     # below this, halve target exposure
    min_cushion_halt: float = 0.10       # below this, closing trades only


def liquidity_check(excess_liquidity: float, net_liq: float,
                    limits: MarginLimits | None = None) -> tuple[str, float, str]:
    """(level, exposure_scale, reason). Levels: ok | no_new_risk | derisk | halt.

    Takes EXCESS LIQUIDITY, not margin used — renamed from `margin_check` so that a caller still
    passing a maintenance-margin figure fails loudly instead of silently inverting the test.

    Note this is ACCOUNT-WIDE: in a shared account every strategy sees the total, so one strategy
    can be blocked by another's usage. That is correct — the constraint really is account-wide,
    and being blocked is strictly better than being liquidated.

    As everywhere else here, `halt` must still permit CLOSING trades. A liquidity guard that
    blocks the orders which would RESTORE the cushion is the worst possible failure mode.
    """
    lim = limits or MarginLimits()
    xl, nl = _num(excess_liquidity), _num(net_liq)
    if xl is None or nl is None or nl <= 0:
        # Distinct from "ok" ON PURPOSE: unknown must be visible in the log, not indistinguishable
        # from healthy. Scale stays 1.0 (fail-OPEN) because the order and gross caps already bound
        # notional, and therefore bound margin indirectly — and because blocking everything on a
        # transient API gap would also block the contract ROLLS that prevent physical delivery.
        # The caller may still choose to treat "unknown" as blocking.
        return "unknown", 1.0, "LIQUIDITY DATA UNAVAILABLE — cushion not enforced this run"
    cushion = xl / nl
    if cushion <= lim.min_cushion_halt:
        return "halt", 0.0, (f"liquidity cushion {cushion:.0%} of NAV <= "
                             f"{lim.min_cushion_halt:.0%} — CLOSING TRADES ONLY; "
                             f"liquidation risk")
    if cushion <= lim.min_cushion_derisk:
        return "derisk", 0.5, (f"liquidity cushion {cushion:.0%} of NAV <= "
                               f"{lim.min_cushion_derisk:.0%} — exposure halved")
    if cushion <= lim.min_cushion_new_risk:
        return "no_new_risk", 0.0, (f"liquidity cushion {cushion:.0%} of NAV <= "
                                    f"{lim.min_cushion_new_risk:.0%} — holding existing "
                                    f"positions, no new ones")
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


# ---------------------------------------------------------------- heartbeat (retroactive)
def missed_runs(history, today: str, date_key: str = "date") -> tuple[int, str | None, str]:
    """(missed_weekdays, last_run_date, note) from the strategy's own snapshot history.

    WHAT THIS CATCHES: the scheduler skipping days — a machine asleep, a failed IB connect, a
    crash mid-run. On the next SUCCESSFUL run the email says so, instead of the gap passing
    unnoticed because each individual email looked fine.

    WHAT IT CANNOT CATCH, and this is the honest limit: a task that is dead for good. If the run
    never executes there is no email to carry the warning, and silence is indistinguishable from a
    quiet day. Closing that hole needs something OUTSIDE these systems — a dead-man's-switch
    service pinged at the end of each run, which alerts when the ping stops. Until then, "no email
    this evening" is the only signal and nothing watches for it.

    Accepts list-of-dicts ({date: ...}) or list-of-tuples ((date, pnl)), since the three states
    differ.
    """
    if not history:
        return 0, None, ""
    last = None
    for h in history:
        d = h.get(date_key) if isinstance(h, dict) else (h[0] if len(h) else None)
        if d and (last is None or str(d) > last):
            last = str(d)
    if not last:
        return 0, None, ""
    try:
        gap = len(pd.bdate_range(last, today)) - 2   # exclude both endpoints
    except Exception:  # noqa: BLE001
        return 0, last, ""
    gap = max(int(gap), 0)
    if gap <= 0:
        return 0, last, ""
    return gap, last, (f"{gap} weekday run(s) MISSED since {last} — scheduler, machine or IB "
                       f"connection failed on those days")


# ---------------------------------------------------------------- deployed code version
def code_version(root: Path | str, upstream: str = "origin/master", warn_behind: int = 5,
                 fetch: bool = True, timeout: float = 15.0) -> tuple[str, int | None]:
    """(note, commits_behind) — which commit is actually running, and how stale it is.

    WHY. On 2026-08-17 the live branch drifted 10 commits behind master within 48 hours of
    go-live, so the real-money account ran with the margin-ceiling bug (maintenance margin vs a
    25% ceiling, which blocks every buy once the book is fully invested), an inert price_sane, no
    negative-share guard and no per-column staleness. NOTHING SURFACED IT. The remaining link in
    the deploy chain was "remember to merge", and this replaces remembering with reporting.

    Fetches first (read-only, never touches the working tree) because otherwise the comparison is
    against whatever ref was last pulled, which is exactly the state that goes stale. Best-effort:
    no network, no git, not a checkout, or a slow remote all degrade to reporting the SHA alone.
    This runs inside a trading process, so it must never raise and never hang.

    Logs at INFO normally and WARNING only past `warn_behind`, because on the live box being a
    commit or two behind between deploys is NORMAL. Warning on every non-zero value would put a
    line in the email subject and a push on your phone every single day, which is how an alert
    channel gets ignored — the failure this whole module keeps guarding against.
    """
    import subprocess

    def _git(*args, t=5.0):
        try:
            r = subprocess.run(("git", "-C", str(root)) + args, capture_output=True,
                               text=True, timeout=t)
            return r.stdout.strip() if r.returncode == 0 else None
        except Exception:  # noqa: BLE001  (subprocess, OSError, timeout — none may propagate)
            return None

    sha = _git("rev-parse", "--short", "HEAD")
    if sha is None:
        return "code version unknown (not a git checkout)", None
    branch = _git("rev-parse", "--abbrev-ref", "HEAD") or "?"
    if fetch:
        _git("fetch", "--quiet", *upstream.split("/", 1), t=timeout)
    behind_s = _git("rev-list", "--count", f"HEAD..{upstream}")
    try:
        behind = int(behind_s) if behind_s is not None else None
    except ValueError:
        behind = None
    if behind is None:
        return f"running {sha} ({branch}) — cannot compare to {upstream}", None
    note = f"running {sha} ({branch}) — {behind} commit(s) behind {upstream}"
    if behind >= warn_behind:
        logging.warning("CODE IS STALE: %s — merge master into this branch and redeploy", note)
    else:
        logging.info("code version: %s", note)
    return note, behind


# ---------------------------------------------------------------- out-of-band alerting
def push_alert(title: str, message: str, api_key: str | None = None) -> bool:
    """Send a Pushbullet note. Returns True if it went out. NEVER raises.

    WHY A SECOND CHANNEL: the daily email carries the alert records, but SMTP is a failure point
    INSIDE the thing being monitored. If sending fails, `logging.error("Email failed")` fires —
    and that alert would be delivered by the email that just failed. The alert channel cannot
    report its own failure. Worse, if the box loses EMAIL_USER/EMAIL_PASS the sender logs a
    warning and returns silently, so the system trades normally and tells you nothing, forever.
    A push over HTTPS to a different host fails independently.

    WHAT THIS IS NOT: a dead-man's switch. Push is initiated BY the run, so if the run never
    executes nothing is sent and silence looks like a quiet day — exactly like email. Detecting
    absence requires an EXTERNAL watcher holding its own schedule; no amount of pushing from
    inside the process can do it.

    Deliberately only called for WARNING+ — a daily "all fine" push is ignored within a week, and
    an alert channel people ignore is not an alert channel.
    """
    key = api_key or os.getenv("PUSHBULLET_API_KEY")
    if not key:
        return False
    try:
        from pushbullet import Pushbullet
        Pushbullet(key).push_note(title[:120], message[:1800])
        return True
    except Exception as e:  # noqa: BLE001 — monitoring must never break trading
        logging.warning("push_alert failed: %s", e)
        return False


def push_if_alerts(collector, subject_prefix: str, api_key: str | None = None) -> bool:
    """Push the collected alerts, if any. Call AFTER the email attempt, so an email failure is
    itself included in what gets pushed."""
    if collector is None or not getattr(collector, "records", None):
        return False
    lines = [f"[{lv}] {msg}" for lv, msg in collector.records]
    return push_alert(f"{subject_prefix}: {collector.worst} x{len(collector.records)}",
                      "\n".join(lines), api_key)


# ---------------------------------------------------------------- layer 4: reconciliation
def reconcile(expected: dict[str, float], actual: dict[str, float], label: str = "",
              tol: float = 1e-9) -> tuple[list[dict], str]:
    """Compare what the strategy THINKS it holds against what the broker REPORTS.

    Motivated by two real incidents (options-vrp, 2026-08-07..11): a close order left
    PreSubmitted was booked as filled, so state dropped a spread that was still open at IB
    (ORPHAN — a live position nothing manages); and a crash before `state.save()` meant two days
    of broker activity were never persisted. Neither is visible from inside the strategy's own
    records, because those records are exactly what is wrong.

    Three discrepancy kinds, and they are not equally bad:
      ORPHAN    at the broker, absent from state. WORST — nothing will manage, close or roll it.
      PHANTOM   in state, absent at the broker. P&L and sizing are computed off a position that
                does not exist.
      MISMATCH  both sides hold it, different size.

    ⚠ THE CALLER MUST FILTER `actual` TO THIS STRATEGY'S OWN INSTRUMENTS. Several strategies
    share one IB account, so an unfiltered comparison reports the others' positions as orphans —
    and acting on that is precisely the bug that once had one strategy flattening another's book.

    REPORTS ONLY, never corrects. Auto-squaring a difference you do not understand can close a
    real position or open an unintended one; the safe move is to alert and let a human look.
    """
    out: list[dict] = []
    for k in sorted(set(expected) | set(actual)):
        e, a = float(expected.get(k, 0.0)), float(actual.get(k, 0.0))
        if abs(e - a) <= tol:
            continue
        kind = "ORPHAN" if abs(e) <= tol else ("PHANTOM" if abs(a) <= tol else "MISMATCH")
        out.append({"key": k, "expected": e, "actual": a, "kind": kind})
    if not out:
        return out, ""
    parts = [f"{d['kind']} {d['key']}: state {d['expected']:g} vs broker {d['actual']:g}"
             for d in out]
    return out, f"RECONCILE{' ' + label if label else ''} — {len(out)} discrepancy(ies): " + \
                "; ".join(parts)


# ---------------------------------------------------------------- book-level equity
@dataclass
class BookLevels(BreakerLevels):
    """⚠ SUPERSEDED — prefer `BreakerLevels.from_vol(book_vol)` from `book_drawdown`.

    Hard-coding the book levels tighter was WRONG whenever fewer than all three strategies are
    running. The tightness is justified by DIVERSIFICATION — the book is quieter than any single
    sleeve because the sleeves are uncorrelated — but during a STAGED GO-LIVE with one strategy
    the book curve IS that strategy's curve, so 10/18/25 fired at -12.7% and -18.2% on a book
    whose own correct levels are 23/38/53. That is precisely the capitulation zone measured on
    the enhanced magic formula (13 of 13 triggers followed by gains, -0.34 Sharpe).

    Vol-scaling the BOOK's own curve gets the same property for free and adapts to how many
    strategies are actually live: one sleeve => book vol equals its vol => identical levels, no
    double-counting; three uncorrelated sleeves => book vol is ~1/sqrt(3) of a sleeve's => levels
    tighten automatically. Kept only so an explicit caller still works.

    ⚠ A book drawdown computed as 1 - sum(equity)/sum(peak) is a peak-WEIGHTED AVERAGE of the
    individual drawdowns, so it can never exceed the worst strategy. With the SAME thresholds a
    book breaker is mathematically incapable of firing when no strategy fires — it adds nothing.
    It only earns its place with tighter levels, which is defensible on its own terms: the book is
    diversified, so a given drawdown there means more than the same number in one sleeve.
    """
    derisk: float = 0.10
    reduce_only: float = 0.18
    halt: float = 0.25


def write_equity(root: Path | str, strategy: str, equity: float, peak: float) -> None:
    """Publish this strategy's equity, and append the BOOK TOTAL to a running history.

    Each strategy's own breaker sees only its own slice, so a correlated bleed across all three
    is invisible to every one of them individually.

    ⚠ THE HISTORY MATTERS: peak-of-the-total is NOT the sum of individual peaks. Those highs
    occur at DIFFERENT times, so summing them invents a book peak that never existed and inflates
    every subsequent drawdown. Appending the total each run and taking its running max measures
    the book's actual equity curve. Still approximate — the three runs fire at different times of
    day — but it is the right quantity rather than a systematically biased one.

    Best-effort: never raises. Failing to publish must not stop trading.
    """
    try:
        f = Path(root) / "book_equity.json"
        d = json.loads(f.read_text()) if f.exists() else {}
        cur = d.get("strategies", {})
        cur[strategy] = {"equity": float(equity), "peak": float(peak),
                         "ts": pd.Timestamp.utcnow().isoformat()}
        d["strategies"] = cur
        total = sum(float(v["equity"]) for v in cur.values())
        hist = d.get("book_history", [])
        today = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
        hist = [h for h in hist if h.get("date") != today]
        hist.append({"date": today, "total": total})
        d["book_history"] = hist[-500:]
        f.write_text(json.dumps(d, indent=2))
    except Exception as e:  # noqa: BLE001
        logging.warning("write_equity failed: %s", e)


def book_vol(root: Path | str, min_obs: int = 60) -> float | None:
    """Annualised vol of the BOOK's own recorded equity curve, or None if too little history.

    Lets the book breaker scale to what is ACTUALLY running: with one strategy live the book vol
    equals that strategy's, so its levels come out identical and the book check adds nothing
    (correct — there is no diversification to reward). As strategies are added the curve quietens
    and the levels tighten by themselves.
    """
    try:
        f = Path(root) / "book_equity.json"
        if not f.exists():
            return None
        hist = [float(h["total"]) for h in (json.loads(f.read_text()).get("book_history") or [])
                if "total" in h]
    except Exception:  # noqa: BLE001
        return None
    if len(hist) < min_obs:
        return None
    r = pd.Series(hist, dtype=float).pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) < min_obs - 1 or r.std() == 0:
        return None
    return float(r.std() * np.sqrt(252))


def book_drawdown(root: Path | str, max_age_days: int = 5
                  ) -> tuple[float | None, float, float, str]:
    """(drawdown, total_equity, book_peak, note) from the book's OWN equity curve.

    Peak is the running max of the recorded TOTAL, not the sum of individual peaks — see
    `write_equity`. Stale strategies are dropped from the current total: one that stopped running
    would otherwise freeze its last equity in, understating the drawdown exactly when it matters.
    """
    try:
        f = Path(root) / "book_equity.json"
        if not f.exists():
            return None, 0.0, 0.0, ""
        d = json.loads(f.read_text())
    except Exception as e:  # noqa: BLE001
        return None, 0.0, 0.0, f"book_equity unreadable: {e}"
    now = pd.Timestamp.utcnow()
    eq = 0.0
    used, stale = [], []
    for name, v in (d.get("strategies") or {}).items():
        try:
            age = (now - pd.Timestamp(v["ts"])).days
        except Exception:  # noqa: BLE001
            age = 999
        if age > max_age_days:
            stale.append(f"{name}({age}d)")
            continue
        eq += float(v["equity"])
        used.append(name)
    hist = [float(h["total"]) for h in (d.get("book_history") or []) if "total" in h]
    peak = max(hist + [eq]) if (hist or eq) else 0.0
    if not used or peak <= 0:
        return None, 0.0, 0.0, ("book drawdown: no fresh entries" +
                                (f"; stale: {', '.join(stale)}" if stale else ""))
    dd = 1.0 - eq / peak
    note = f"book across {len(used)} ({', '.join(used)}): ${eq:,.0f} vs curve peak ${peak:,.0f}"
    if stale:
        note += f"  [IGNORED stale: {', '.join(stale)}]"
    return dd, eq, peak, note
