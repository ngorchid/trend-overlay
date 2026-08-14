"""Tests for risk_guard. Every case is a failure we have actually seen or specifically fear.

Run: python scripts/test_risk_guard.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from risk_guard import (RiskLimits, chain_sane, check_batch, check_order,  # noqa: E402
                        data_fresh, halted, liquidity_check, price_sane, Check,
                        ALLOCATIONS, Allocation, MIN_ALLOCATION_CUSHION,
                        allocated_budget, allocation_cushion, check_allocations)

L = RiskLimits(budget=100_000.0)
TODAY = pd.Timestamp("2026-08-11")
fails, ran = [], 0


def expect(label: str, got, want_ok: bool = True) -> None:
    """`got` MUST be a Check (or anything defining __bool__).

    A plain ad-hoc object is ALWAYS truthy, so `expect(..., True)` on one can never
    fail. Four cases here were written that way and silently passed regardless of
    the code under test; a mutation run (2026-08-14) caught it.
    """
    global ran
    ran += 1
    if bool(got) != want_ok:
        fails.append(f"{label}: expected {'PASS' if want_ok else 'REJECT'}, got "
                     f"{'PASS' if got else 'REJECT'} ({got.reason})")
    flag = "ok " if bool(got) == want_ok else "FAIL"
    print(f"  [{flag}] {label:52s} -> {'PASS' if got else 'REJECT'}"
          f"{'' if got else '  | ' + got.reason[:72]}")


print("KILL SWITCH")
with tempfile.TemporaryDirectory() as d:
    expect("no HALT file", halted(d, L), True)
    Path(d, "HALT").write_text("manual stop: investigating fills")
    expect("HALT file present", halted(d, L), False)

print("\nLAYER 1 — DATA FRESHNESS")
idx = pd.bdate_range("2026-07-01", "2026-08-11")
expect("current data", data_fresh(idx, TODAY, L), True)
expect("3 days old (long weekend)", data_fresh(pd.bdate_range("2026-07-01", "2026-08-08"), TODAY, L), True)
expect("11 days old (frozen feed)", data_fresh(pd.bdate_range("2026-07-01", "2026-07-31"), TODAY, L), False)
expect("empty index", data_fresh(pd.Index([]), TODAY, L), False)
expect("data from the future", data_fresh(pd.bdate_range("2026-08-01", "2026-08-20"), TODAY, L), False)

print("\nLAYER 1 — PRICE SANITY")
expect("normal price", price_sane("AAPL", 305.0, 308.0, L), True)
expect("NaN price", price_sane("AAPL", float("nan"), 308.0, L), False)
expect("zero price", price_sane("AAPL", 0.0, 308.0, L), False)
expect("unadjusted 4:1 split", price_sane("NVDA", 54.0, 217.0, L), False)
expect("legitimate -20% gap", price_sane("PFE", 21.6, 27.0, L), True)
expect("no prior available", price_sane("NEW", 42.0, None, L), True)

print("\nLAYER 1 — OPTION CHAIN SANITY  (the 2026-08-11 off-hours failure)")
expect("healthy chain", chain_sane("SPY", 0.122, pd.Series([0.55, 0.30, 0.43]), L), True)
expect("off-hours junk IV 1.56%", chain_sane("SPY", 0.0156, None, L), False)
expect("all bids zero", chain_sane("SPY", 0.122, pd.Series([0.0, 0.0, 0.0]), L), False)
expect("NaN IV", chain_sane("SPY", float("nan"), None, L), False)
expect("absurd IV 400%", chain_sane("XYZ", 4.0, None, L), False)
expect("genuinely high IV 90%", chain_sane("NVDA", 0.90, pd.Series([1.2]), L), True)

print("\nLAYER 2 — ORDER SANITY  (budget $100,000)")
expect("normal stock buy 30sh @ $305", check_order("AAPL", "BUY", 30, 305.0, 1.0, L), True)
expect("fat finger 3000sh @ $305 (915% of budget)", check_order("AAPL", "BUY", 3000, 305.0, 1.0, L), False)
expect("fractional qty", check_order("AAPL", "BUY", 10.5, 305.0, 1.0, L), False)
expect("negative qty", check_order("AAPL", "BUY", -10, 305.0, 1.0, L), False)
expect("bad side", check_order("AAPL", "HOLD", 10, 305.0, 1.0, L), False)
expect("would breach per-instrument cap", check_order("AAPL", "BUY", 30, 305.0, 1.0, L,
                                                      current_position_notional=42_000), False)
expect("limit 3% off the mark", check_order("AAPL", "BUY", 30, 314.0, 1.0, L,
                                            reference_price=305.0), True)
expect("limit 40% off the mark (stale ref)", check_order("AAPL", "BUY", 30, 427.0, 1.0, L,
                                                         reference_price=305.0), False)
print("  -- futures: INDIVISIBLE, so equity caps are wrong (this is why presets exist) --")
F = RiskLimits.for_futures(200_000.0)
expect("1 MES @ $27.5k vs EQUITY limits (correctly rejected)",
       check_order("MES", "BUY", 1, 5500.0, 5.0, L), False)
expect("1 MES @ $27.5k vs FUTURES limits on $200k", check_order("MES", "BUY", 1, 5500.0, 5.0, F), True)
expect("1 ES @ $275k on a $200k futures book (137%)",
       check_order("ES", "BUY", 1, 5500.0, 50.0, F), False)
expect("1 ES @ $275k on a $1M futures book",
       check_order("ES", "BUY", 1, 5500.0, 50.0, RiskLimits.for_futures(1_000_000.0)), True)
expect("futures book stays under the 3.3x gross backstop",
       check_order("MGC", "BUY", 1, 3000.0, 10.0, F, gross_notional=640_000), False)
print("  -- options: 100x multiplier --")
O = RiskLimits.for_options(100_000.0)
expect("2 SPY spreads @ $4.40 x 100", check_order("SPY", "SELL", 2, 4.40, 100.0, O), True)
expect("200 SPY spreads (typo: 100x too many)", check_order("SPY", "SELL", 200, 4.40, 100.0, O), False)

print("\nLAYER 2 — BATCH  (individually fine, collectively absurd)")
orders = [{"ticker": f"T{i}", "side": "BUY", "qty": 40, "price": 300.0, "multiplier": 1.0}
          for i in range(12)]
acc, rej = check_batch(orders, L)
print(f"  12 orders x $12,000 = $144,000 against a $100k budget and 1.10x gross cap")
print(f"  -> accepted {len(acc)}, rejected {len(rej)}")
expect("batch stops at the gross cap", Check(len(acc) < 12), True)

# ---------------------------------------------------------------- liquidity floor
# Rewritten 2026-08-14 from gross maintenance margin to EXCESS LIQUIDITY. The old form measured
# MaintMarginReq/NetLiq against ceilings; Reg-T maintenance on long stock is 25% of position value
# regardless of leverage, so a fully-invested UNBORROWED equity book read as 25% "used" and tripped
# no_new_risk in normal operation -- blocking every buy while sells continued. These cases pin the
# property that broke: an unlevered book must be OK, and a levered one must still fire.
print("\n--- liquidity floor (excess liquidity / NAV) ---")
NAV = 50_000.0


def lvl_of(maint_frac):
    return liquidity_check(NAV - NAV * maint_frac, NAV)[0]


def scale_of(maint_frac):
    return liquidity_check(NAV - NAV * maint_frac, NAV)[1]


# Scale is asserted alongside the level: a `derisk` that returned 1.0 would report correctly in
# the log and silently not halve anything. A mutation run caught exactly that gap.
for mf, want, want_scale, why in [
    (0.25, "ok", 1.0, "fully invested unlevered equities (THE phase-1 regression)"),
    (0.40, "ok", 1.0, "unlevered equities at 40% house maintenance"),
    (0.48, "ok", 1.0, "three-strategy steady state"),
    (0.70, "no_new_risk", 0.0, "levered: 30% cushion"),
    (0.80, "derisk", 0.5, "levered hard: 20% cushion"),
    (0.92, "halt", 0.0, "8% cushion -- near liquidation"),
    (1.05, "halt", 0.0, "negative cushion -- already past it"),
]:
    got, got_scale = lvl_of(mf), scale_of(mf)
    print(f"  maint {mf:>5.0%} -> cushion {1-mf:>5.0%}  {got:<12} scale {got_scale:.1f}  ({why})")
    expect(f"liquidity floor at {mf:.0%} maintenance -> {want}",
           Check(got == want, f"got {got}"), True)
    expect(f"liquidity floor at {mf:.0%} maintenance -> scale {want_scale}",
           Check(got_scale == want_scale, f"got scale {got_scale}"), True)

# halt must still permit CLOSING trades: scale 0.0 means "no NEW risk", and every caller gates
# only its open path on it. A liquidity guard that blocked the orders which RESTORE the cushion
# would be the worst possible failure mode.
expect("halt returns scale 0 (no new risk), not a blocked close",
       Check(scale_of(1.05) == 0.0), True)

# Unknown must be distinguishable from healthy AND fail-open: the order and gross caps already
# bound notional, and blocking everything on a transient API gap would also block contract rolls.
for lab, a, b in [("nan", float("nan"), NAV), ("None", None, NAV), ("zero NAV", 1000.0, 0.0)]:
    l_, s_, _ = liquidity_check(a, b)
    expect(f"liquidity {lab} -> unknown, fail-open",
           Check(l_ == "unknown" and s_ == 1.0, f"got {l_}/{s_}"), True)


# ---------------------------------------------------------------- capital allocation
# Added 2026-08-14. Before this, each strategy sized off its OWN independently-set budget with no
# strategy aware of any other: $50k + $75k + $100k of sizing base on a $50k account, which put
# total maintenance at 74% of NAV and tripped no_new_risk on an ordinary day. The only shared
# mechanism was a backstop that BLOCKS, not an allocator that RESERVES.
print("\n--- capital allocation (fraction of NetLiq, validated against simultaneous peak) ---")

_c = allocation_cushion()
print(f"  configured cushion at simultaneous peak: {_c:.1%} (floor {MIN_ALLOCATION_CUSHION:.0%})")
for _n, _a in ALLOCATIONS.items():
    print(f"    {_n:16} {_a.fraction:>5.0%} of NetLiq x {_a.peak_margin_coef:.3f} peak "
          f"= {_a.fraction*_a.peak_margin_coef:>6.1%} of NAV")
expect("shipped allocations leave >= the required cushion", check_allocations())
expect("cushion matches 1 - sum(fraction x coef)",
       Check(abs(_c - (1 - sum(a.fraction*a.peak_margin_coef for a in ALLOCATIONS.values())))
             < 1e-12, f"{_c}"))

# The combination the capital sweep PREFERS for options-vrp ($75k on a $50k account) does not
# fit alongside the other two, and must be refused rather than silently sized.
_bad = {"magic-formula": Allocation(1.0, 0.25), "options-vrp": Allocation(1.5, 0.18),
        "trend-overlay": Allocation(1.0, 0.109)}
expect("an over-allocated table is REJECTED, not silently sized",
       check_allocations(_bad), want_ok=False)
expect("  ... and the rejection names the cushion it computed",
       Check("37.1%" in check_allocations(_bad).reason, check_allocations(_bad).reason))
expect("a single sleeve over the whole account is REJECTED",
       check_allocations({"magic-formula": Allocation(3.0, 0.25)}), want_ok=False)
expect("negative fraction is REJECTED",
       check_allocations({"x": Allocation(-1.0, 0.25)}), want_ok=False)
expect("an empty table trivially passes (nothing allocated)", check_allocations({}))

# Budget must track NetLiq, so a sleeve reaches its measured plateau without an edit.
_b25, _ = allocated_budget("options-vrp", 25_000.0, 50_000.0)
_b50, _ = allocated_budget("options-vrp", 50_000.0, 50_000.0)
_b100, _ = allocated_budget("options-vrp", 100_000.0, 50_000.0)
print(f"  options-vrp budget at NetLiq 25k/50k/100k = ${_b25:,.0f} / ${_b50:,.0f} / ${_b100:,.0f}")
expect("budget scales with NetLiq", Check(_b25 < _b50 < _b100, f"{_b25},{_b50},{_b100}"))
expect("budget at nominal == fraction x nominal", Check(abs(_b50 - 50_000.0) < 1e-6, f"{_b50}"))
expect("budget FALLS with the account (no floor propping up a loss)",
       Check(_b25 < _b50, f"{_b25} vs {_b50}"))
_bcap, _ = allocated_budget("options-vrp", 10_000_000.0, 50_000.0)
expect("growth capped at max_grow (guards a corrupted P&L ledger)",
       Check(_bcap <= 3.0 * 50_000.0 + 1e-6, f"${_bcap:,.0f}"))

_bn, _note = allocated_budget("options-vrp", None, 50_000.0)
expect("NetLiq unavailable -> nominal share, and SAYS so",
       Check(abs(_bn - 50_000.0) < 1e-6 and "UNAVAILABLE" in _note, _note))
for _bad_nl in (float("nan"), 0.0, -1.0):
    _b, _n = allocated_budget("options-vrp", _bad_nl, 50_000.0)
    expect(f"NetLiq {_bad_nl!r} -> nominal fallback, not a poisoned size",
           Check(abs(_b - 50_000.0) < 1e-6 and "UNAVAILABLE" in _n, _n))
# A fraction != 1 must actually be APPLIED. Every shipped fraction is 1.0, so without this the
# multiplication is a no-op and dropping it entirely survives mutation testing unnoticed.
_half = {"half": Allocation(0.50, 0.20)}
_bh, _ = allocated_budget("half", 50_000.0, 50_000.0, allocations=_half)
expect("fraction is applied: 50% of a $50k account -> $25k budget",
       Check(abs(_bh - 25_000.0) < 1e-6, f"${_bh:,.0f}"))
_bq, _ = allocated_budget("half", 100_000.0, 50_000.0, allocations=_half)
expect("  ... and still applied when NetLiq differs from nominal",
       Check(abs(_bq - 50_000.0) < 1e-6, f"${_bq:,.0f}"))
_bhn, _ = allocated_budget("half", None, 50_000.0, allocations=_half)
expect("  ... and applied to the nominal fallback too",
       Check(abs(_bhn - 25_000.0) < 1e-6, f"${_bhn:,.0f}"))

_bu, _nu = allocated_budget("not-a-strategy", 50_000.0, 50_000.0)
expect("unknown strategy -> 0 budget and refuses, rather than defaulting to something",
       Check(_bu == 0.0 and "refusing" in _nu, _nu))

# The peak-margin coefficients are the load-bearing numbers; pin them to their derivations.
expect("options-vrp coef == max_positions x risk_per_trade (6 x 3%)",
       Check(abs(ALLOCATIONS["options-vrp"].peak_margin_coef - 6 * 0.03) < 1e-9,
             f"{ALLOCATIONS['options-vrp'].peak_margin_coef}"))
expect("magic-formula coef == Reg-T maintenance on long stock (25%)",
       Check(abs(ALLOCATIONS["magic-formula"].peak_margin_coef - 0.25) < 1e-9, ""))
expect("trend coef == ~3.5% SPAN on ~3.12x notional",
       Check(abs(ALLOCATIONS["trend-overlay"].peak_margin_coef - 0.035 * 3.12) < 0.002,
             f"{ALLOCATIONS['trend-overlay'].peak_margin_coef}"))

# End-to-end: the shipped table must actually clear the liquidity floor it is judged against.
_tot = sum(a.fraction * a.peak_margin_coef for a in ALLOCATIONS.values()) * 50_000.0
_lvl, _sc, _ = liquidity_check(50_000.0 - _tot, 50_000.0)
expect("shipped allocations clear the liquidity floor at simultaneous peak",
       Check(_lvl == "ok" and _sc == 1.0, f"got {_lvl}/{_sc}"))

print("\n" + "=" * 78)
if fails:
    print(f"{len(fails)} FAILURE(S) of {ran}:")
    for f in fails:
        print("   " + f)
    sys.exit(1)
print(f"all {ran} checks behaved as expected")
