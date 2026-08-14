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
                        data_fresh, halted, liquidity_check, price_sane, Check)

L = RiskLimits(budget=100_000.0)
TODAY = pd.Timestamp("2026-08-11")
fails, ran = [], 0


def expect(label: str, got, want_ok: bool) -> None:
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


print("\n" + "=" * 78)
if fails:
    print(f"{len(fails)} FAILURE(S) of {ran}:")
    for f in fails:
        print("   " + f)
    sys.exit(1)
print(f"all {ran} checks behaved as expected")
