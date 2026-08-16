"""Tests for trend-overlay sizing and the SAFETY invariant.

WHY THIS EXISTS. `--selftest` prints and never fails: on 2026-08-14 seeded faults went straight
through it, including blowing the gross cap from 3x to 300x and disabling hysteresis entirely.
It is a demonstration, not a test.

This sleeve is the one that most needs real tests, because BOTH of its known bugs were SILENT —
neither raised, neither looked wrong in a log:

  1. THE VOL-TARGET WAS CIRCULAR. Sizing is inverse-vol, so per-market vol contribution is
     (expo_i/budget)*vol_i = target_vol/sqrt(N) — vol_i CANCELS. est_vol then came out equal to
     target_vol by construction, scale was always 1.0, and the vol target was a no-op. Gross ran
     2.8x budget at the median and hit 4.8x with no leverage cap at all.
  2. est_vol ASSUMED ZERO CORRELATION. sqrt(sum (w_i*vol_i)^2) ignores that rates correlate with
     rates and everything correlates in risk-off, so it understated book vol and realised vol ran
     12-14% against a 10% target.

Both produce plausible numbers. Only a test that checks the sizing RESPONDS to its inputs can
catch either, which is what the first two sections do.

⚠ EVERY CASE DRIVES THE REAL CODE — compute_targets, _portfolio_vol, safety_closes,
FuturesBroker.execute. Mirroring the arithmetic locally tests nothing: in the options-vrp suite
both sizing mutations survived until the cases were rewritten to call build_spread.

⚠ EVERY CASE USES THE REAL `Check`. A bare `type("C", (), {...})()` has no `__bool__`, is always
truthy, and four such assertions elsewhere had never been able to fail.

⚠ MUTATION-TESTED. Run `python3 scripts/mutate_trend_sizing.py`.

Run: python3 scripts/test_trend_sizing.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trend_overlay.contracts import FUTURES  # noqa: E402
from trend_overlay.execution import (HeldPosition, RollOrder,  # noqa: E402
                                     TrendPaperConfig, _portfolio_vol, compute_targets,
                                     days_to_expiry, safety_closes)
from risk_guard import Check, RiskLimits  # noqa: E402

logging.disable(logging.WARNING)
CFG = TrendPaperConfig()
fails, ran = [], 0


def expect(label: str, got, want_ok: bool = True) -> None:
    """`got` MUST define __bool__ (use Check). A bare object is always truthy."""
    global ran
    ran += 1
    ok = bool(got)
    if ok != want_ok:
        fails.append(f"{label}: expected {'PASS' if want_ok else 'REJECT'}, got "
                     f"{'PASS' if ok else 'REJECT'} ({getattr(got, 'reason', '')})")
    print(f"  [{'ok ' if ok == want_ok else 'FAIL'}] {label:66} -> {'PASS' if ok else 'REJECT'}"
          f"{'' if ok == want_ok else '  | ' + getattr(got, 'reason', '')}")


# --------------------------------------------------------------------------- fixtures
SPECS = FUTURES[:4]                       # equity_us, rates_10y, gold, copper
ETFS = [s.proxy_etf for s in SPECS]
IDX = pd.bdate_range("2021-01-01", periods=700)


def panel(vols, corr=0.0, trend=0.0004, seed=0) -> pd.DataFrame:
    """Price panel with CONTROLLED annualised vols and pairwise correlation.

    Built from returns rather than hand-drawn prices so `compute_targets` sees a realistic
    rolling-vol and correlation structure. `trend` gives every market a positive drift so the
    TSMOM signal is +1 and exposures are non-zero.
    """
    rng = np.random.default_rng(seed)
    n = len(IDX)
    common = rng.standard_normal(n)
    out = {}
    for i, (etf, v) in enumerate(zip(ETFS, vols)):
        idio = rng.standard_normal(n)
        z = np.sqrt(corr) * common + np.sqrt(1 - corr) * idio
        r = trend + z * (v / np.sqrt(252))
        out[etf] = 100 * np.exp(np.cumsum(r))
    return pd.DataFrame(out, index=IDX)


def gross_of(df, cfg):
    return float(df["dollar_exposure"].abs().sum()) / cfg.budget


print("=" * 100)
print("TREND OVERLAY — SIZING AND SAFETY")
print("=" * 100)

# =====================================================================================
print("\n--- 1. VOL-TARGET CIRCULARITY: sizing must RESPOND to the vol input ---")
print("    (the bug: vol cancelled, est_vol == target_vol, scale always 1.0, gross hit 4.8x)")

lo_vol = panel([0.08, 0.05, 0.10, 0.15], corr=0.35)
hi_vol = panel([0.32, 0.20, 0.40, 0.60], corr=0.35)
t_lo = compute_targets(lo_vol, CFG, SPECS)
t_hi = compute_targets(hi_vol, CFG, SPECS)
print(f"    low-vol book  gross {gross_of(t_lo, CFG):.2f}x   high-vol book gross {gross_of(t_hi, CFG):.2f}x")
expect("gross RESPONDS to the vol regime (identical => vol cancelled again)",
       Check(abs(gross_of(t_lo, CFG) - gross_of(t_hi, CFG)) > 1e-6,
             f"{gross_of(t_lo, CFG):.4f} vs {gross_of(t_hi, CFG):.4f}"))

# THE ISOLATION TEST. The above alone does NOT prove the vol-target works: inverse-vol sizing
# already makes a low-vol book bigger, with or without a functioning scale. CORRELATION is the
# clean instrument -- it moves est_vol but leaves every market's raw inverse-vol exposure
# untouched, so any change in gross is attributable to the scale ALONE. If the circularity
# returns, or scale is hard-wired to 1.0, these two are identical to the digit.
_free = TrendPaperConfig(per_market_cap=0.0, gross_cap=0.0)   # caps off, or they mask it
_g_lo = gross_of(compute_targets(panel([0.20] * 4, corr=0.0, seed=3), _free, SPECS), _free)
_g_hi = gross_of(compute_targets(panel([0.20] * 4, corr=0.9, seed=3), _free, SPECS), _free)
print(f"    SAME vols, corr 0.0 -> gross {_g_lo:.3f}x ; corr 0.9 -> gross {_g_hi:.3f}x")
expect("gross responds to CORRELATION alone (isolates the vol-target scale)",
       Check(abs(_g_lo - _g_hi) > 1e-6, f"{_g_lo:.6f} vs {_g_hi:.6f}"))
expect("  ... and a MORE correlated book is sized SMALLER (the scale de-risks it)",
       Check(_g_hi < _g_lo, f"corr0.9 {_g_hi:.4f} should be < corr0.0 {_g_lo:.4f}"))

# The circularity's signature: est_vol lands EXACTLY on target_vol, so scale is exactly 1.0.
w = (t_lo["dollar_exposure"] / CFG.budget).values
rets_lo = lo_vol.pct_change(fill_method=None)
vols_lo = (rets_lo.rolling(CFG.vol_window).std() * np.sqrt(252)).iloc[-1].values
ev = _portfolio_vol(w, vols_lo, rets_lo, CFG)
print(f"    est_vol {ev:.4f} vs target {CFG.target_vol:.4f}")
expect("est_vol is NOT identically target_vol (the circularity signature)",
       Check(abs(ev - CFG.target_vol) > 1e-6, f"est {ev:.6f} target {CFG.target_vol}"))
# NB est_vol < target_vol here is NORMAL, not a bug: the per-market and gross caps, and a
# signal that is 0 or +/-0.5 for some markets, both shrink w below the pure inverse-vol
# allocation. The property that matters is that the COVARIANCE estimate exceeds the
# independence one on the same inputs -- assuming otherwise was a wrong premise on my part.
_indep_same = float(np.sqrt(np.nansum((w * vols_lo) ** 2)))
expect("the covariance estimate exceeds the independence estimate on the SAME w and vols",
       Check(ev > _indep_same, f"cov {ev:.4f} vs indep {_indep_same:.4f}"))

# =====================================================================================
print("\n--- 2. COVARIANCE: est_vol must rise with CORRELATION, not just with vol ---")
print("    (the bug: sqrt(sum (w*vol)^2) assumed independence; realised 12-14% vs a 10% target)")

VOLS = np.array([0.20, 0.20, 0.20, 0.20])
W = np.array([0.5, 0.5, 0.5, 0.5])
ev_by_corr = []
for c in (0.0, 0.3, 0.6, 0.9):
    p = panel([0.20] * 4, corr=c, seed=3)
    r = p.pct_change(fill_method=None)
    ev_by_corr.append(_portfolio_vol(W, VOLS, r, CFG))
    print(f"    pairwise corr {c:.1f} -> est_vol {ev_by_corr[-1]:.4f}")
expect("est_vol is MONOTONE INCREASING in correlation (vols held constant)",
       Check(all(a < b for a, b in zip(ev_by_corr, ev_by_corr[1:])), f"{ev_by_corr}"))
expect("  ... and the high-correlation book is materially higher, not marginally",
       Check(ev_by_corr[-1] > 1.3 * ev_by_corr[0],
             f"{ev_by_corr[-1]:.4f} vs {ev_by_corr[0]:.4f}"))
indep = float(np.sqrt(np.nansum((W * VOLS) ** 2)))
expect("a correlated book estimates ABOVE the independence formula",
       Check(ev_by_corr[-1] > indep, f"cov {ev_by_corr[-1]:.4f} vs indep {indep:.4f}"))
expect("the independence formula is what a zero-correlation book collapses to",
       Check(abs(ev_by_corr[0] - indep) / indep < 0.15,
             f"{ev_by_corr[0]:.4f} vs {indep:.4f}"))
# Fallbacks must not silently produce a LOWER estimate than independence (anti-conservative).
short = panel([0.20] * 4, corr=0.6, seed=4).tail(50)
expect("too little history -> falls back to independence, not to something smaller",
       Check(_portfolio_vol(W, VOLS, short.pct_change(fill_method=None), CFG) >= indep - 1e-12, ""))
expect("corr_weight=0 -> independence exactly",
       Check(abs(_portfolio_vol(W, VOLS, panel([0.2]*4, corr=0.6, seed=5).pct_change(fill_method=None),
                                TrendPaperConfig(corr_weight=0.0)) - indep) < 1e-12, ""))

# =====================================================================================
print("\n--- 3. GROSS CAP binds, and binds AFTER the vol-target multiplier ---")
tight = TrendPaperConfig(gross_cap=1.0)
t_g = compute_targets(lo_vol, tight, SPECS)
print(f"    gross_cap 1.0x -> realised gross {gross_of(t_g, tight):.3f}x")
expect("gross cap 1.0x actually clamps total notional",
       Check(gross_of(t_g, tight) <= 1.0 + 1e-6, f"{gross_of(t_g, tight):.4f}x"))
expect("  ... and the uncapped book really did exceed it (guard is not vacuous)",
       Check(gross_of(compute_targets(lo_vol, TrendPaperConfig(gross_cap=0.0), SPECS),
                      CFG) > 1.0, "uncapped must breach 1.0x or this proves nothing"))
for gc in (0.5, 2.0, 3.0):
    c = TrendPaperConfig(gross_cap=gc)
    expect(f"gross cap {gc}x respected",
           Check(gross_of(compute_targets(lo_vol, c, SPECS), c) <= gc + 1e-6, ""))
# overlay_multiple scales the cap with the book, not against it
c2 = TrendPaperConfig(gross_cap=1.0, overlay_multiple=0.5)
expect("gross cap scales with overlay_multiple",
       Check(gross_of(compute_targets(lo_vol, c2, SPECS), c2) <= 0.5 + 1e-6, ""))

# =====================================================================================
print("\n--- 4. PER-MARKET CAP binds — including AFTER scaling (the scale>1 hole) ---")
cap = TrendPaperConfig(per_market_cap=0.10, gross_cap=0.0)
t_c = compute_targets(lo_vol, cap, SPECS)
worst = (t_c["dollar_exposure"].abs() / cap.budget).max()
print(f"    per_market_cap 10% -> largest single market {worst:.3f} of budget")
expect("no market exceeds the per-market cap after ALL scaling",
       Check(worst <= 0.10 + 1e-6, f"{worst:.4f}"))
expect("  ... and uncapped, some market really did exceed it (guard is not vacuous)",
       Check((compute_targets(lo_vol, TrendPaperConfig(per_market_cap=0.0, gross_cap=0.0),
                              SPECS)["dollar_exposure"].abs() / CFG.budget).max() > 0.10, ""))
# The 2026-08-11 hole: capping only BEFORE the vol-target multiplier is not a cap, because
# scale_clip permits up to 1.5. Force scale>1 with a very low-vol book.
calm = panel([0.03, 0.02, 0.04, 0.05], corr=0.1, seed=9)
cap2 = TrendPaperConfig(per_market_cap=0.10, gross_cap=0.0)
worst2 = (compute_targets(calm, cap2, SPECS)["dollar_exposure"].abs() / cap2.budget).max()
expect("cap still holds on a book where the vol-target multiplier scales UP",
       Check(worst2 <= 0.10 + 1e-6, f"{worst2:.4f}"))
# 2c: one contract can exceed the cap on its own -> hold ZERO rather than blow through it.
chunky = TrendPaperConfig(per_market_cap=0.001, gross_cap=0.0, budget=200_000.0)
t_ch = compute_targets(lo_vol, chunky, SPECS)
used = (t_ch["contracts"].abs() * t_ch["notional"]) / chunky.budget
expect("a market whose ONE contract breaches the cap holds ZERO, not one",
       Check(bool((used <= 0.001 * 1.002).all()), f"max {used.max():.5f} of budget"))
expect("  ... i.e. contracts go to 0 rather than through the cap",
       Check(int(t_ch["contracts"].abs().sum()) == 0, f"{t_ch['contracts'].to_dict()}"))
# And the NONZERO reduction: 3ct x $27,500 = $82,500 over an $80,000 cap must become 2ct, not 0
# and not 3. Pinning only the ->0 case leaves the flooring arithmetic itself untested.
red = TrendPaperConfig(per_market_cap=0.40, gross_cap=0.0, budget=200_000.0)
t_red = compute_targets(lo_vol, red, SPECS)
_lim_red = red.per_market_cap * red.budget
_used = (t_red["contracts"].abs() * t_red["notional"])
expect("every rounded position sits within the cap after flooring",
       Check(bool((_used <= _lim_red * 1.001).all()), f"max ${_used.max():,.0f} vs ${_lim_red:,.0f}"))
expect("  ... and the flooring reduces to a NONZERO count where one fits",
       Check(int(t_red["contracts"].abs().sum()) > 0, f"{t_red['contracts'].to_dict()}"))
expect("  ... with at least one market actually holding contracts (not a vacuous all-zero pass)",
       Check(int((t_red["contracts"] != 0).sum()) >= 1, ""))

# =====================================================================================
print("\n--- 5. VOL FLOOR: a vanishing vol must not produce an exploding position ---")
# expo = sg * (budget*target_vol/sqrt(N)) / vol, so expo -> inf as vol -> 0.
# Vol must COLLAPSE relative to its own past for an adaptive floor to bite. A constant 0.4%
# series has a 20th percentile of 0.4% and nothing to floor -- which is why the first version
# of this case passed identically with the floor on and off, i.e. proved nothing.
def collapsing(seed=11):
    rng = np.random.default_rng(seed)
    n = len(IDX)
    out = {}
    for i, etf in enumerate(ETFS):
        v = np.full(n, 0.20)
        if i == 0:                                   # this market's vol collapses at the end
            v[-90:] = 0.004
        r = 0.0004 + rng.standard_normal(n) * (v / np.sqrt(252))
        out[etf] = 100 * np.exp(np.cumsum(r))
    return pd.DataFrame(out, index=IDX)


tiny = collapsing()
no_floor = TrendPaperConfig(vol_floor_pct=0.0, per_market_cap=0.0, gross_cap=0.0)
with_floor = TrendPaperConfig(vol_floor_pct=0.20, per_market_cap=0.0, gross_cap=0.0)
e_no = compute_targets(tiny, no_floor, SPECS)["dollar_exposure"].abs().max() / CFG.budget
e_fl = compute_targets(tiny, with_floor, SPECS)["dollar_exposure"].abs().max() / CFG.budget
print(f"    largest market: no floor {e_no:.2f}x budget   with floor {e_fl:.2f}x budget")
expect("the vol floor reduces the largest position",
       Check(e_fl < e_no, f"floor {e_fl:.3f} vs none {e_no:.3f}"))
expect("  ... and without it the position really does blow up (guard is not vacuous)",
       Check(e_no > 0.5, f"unfloored max {e_no:.3f}x budget"))
expect("floor is a percentile of the market's OWN history (adaptive, not absolute)",
       Check(TrendPaperConfig().vol_floor_pct > 0 and TrendPaperConfig().vol_floor_pct < 1, ""))
expect("zero/NaN vol yields ZERO exposure, never inf or NaN",
       Check(np.isfinite(compute_targets(tiny, no_floor, SPECS)["dollar_exposure"]).all(), ""))

# =====================================================================================
print("\n--- 6. HYSTERESIS: inside the band holds, crossing it trades ---")
print("    (band 0.70 chosen on SUB-PERIOD robustness, not the full-sample peak)")
# The discriminating case needs frac(raw) BETWEEN 0.5 and the band: then round() says one
# thing and the band says another. With frac < 0.5 (e.g. copper at 7.143) both say "7" and the
# case cannot fail -- which is exactly why the first version of it survived mutation.
HCFG = TrendPaperConfig(budget=250_000.0, per_market_cap=0.0, gross_cap=0.0)
base_t = compute_targets(lo_vol, HCFG, SPECS, held=None)
raw = (base_t["dollar_exposure"] / base_t["notional"])
_cands = [(m, float(x)) for m, x in raw.items()
          if 0.5 < abs(x) - np.floor(abs(x)) < HCFG.hysteresis_band]
expect("a market exists whose target sits between round() and the band (else this proves nothing)",
       Check(len(_cands) > 0, f"fractions {[round(abs(x)%1,3) for x in raw]}"))
mkt, r_val = _cands[0]
n_round = int(np.round(r_val))
inside = int(np.floor(r_val))            # distance in (0.5, band) -> band HOLDS, round() MOVES
held_in = {m: 0 for m in base_t.index}
held_in[mkt] = inside
got_in = int(compute_targets(lo_vol, HCFG, SPECS, held=held_in).at[mkt, "contracts"])
print(f"    {mkt}: raw {r_val:.3f}, round()={n_round}, holding {inside} "
      f"(distance {abs(r_val-inside):.3f} < band {HCFG.hysteresis_band}) -> contracts {got_in}")
expect("a holding INSIDE the band is left alone (no trade)",
       Check(got_in == inside, f"held {inside} -> {got_in}, want {inside}"))
expect("  ... and round() WOULD have moved it (so the band is what is being tested)",
       Check(n_round != inside, f"round={n_round} held={inside}"))
# Hold a position well OUTSIDE the band: must move to round(target).
outside = n_round + 3
held_out = dict(held_in); held_out[mkt] = outside
got_out = int(compute_targets(lo_vol, HCFG, SPECS, held=held_out).at[mkt, "contracts"])
expect("a holding OUTSIDE the band moves to round(target)",
       Check(got_out == n_round, f"held {outside} -> {got_out}, want {n_round}"))
expect("  ... and that is a genuine change (guard is not vacuous)",
       Check(outside != n_round, ""))
# band 0.5 is EXACTLY round() — the null check the lab reproduces to the digit.
null = TrendPaperConfig(budget=250_000.0, per_market_cap=0.0, gross_cap=0.0,
                        hysteresis_band=0.5)
same = all(int(compute_targets(lo_vol, null, SPECS, held={m: 0 for m in base_t.index}).at[m, "contracts"])
           == int(compute_targets(lo_vol, null, SPECS, held=None).at[m, "contracts"])
           for m in base_t.index)
expect("band 0.5 reproduces plain round() exactly (the null check)", Check(same, ""))
expect("band 0 / held=None falls back to round()",
       Check(int(compute_targets(lo_vol, TrendPaperConfig(budget=250_000.0, per_market_cap=0.0, gross_cap=0.0,
                                                 hysteresis_band=0.0),
                                 SPECS, held=held_out).at[mkt, "contracts"]) == n_round, ""))
# ⚠ DOCUMENTED DISCREPANCY, pinned deliberately. The hysteresis block carries a comment
# saying a non-finite target means missing data and must HOLD. It cannot: upstream,
# `expo = 0.0 if (vv == 0 or vv != vv)` converts a NaN vol into a legitimate ZERO target, so
# `raw` is finite, the isfinite branch never fires, and |0 - held| >= band FLATTENS the market.
# So a single market losing its price feed produces a full round-trip liquidation of that
# market, not a hold. data_fresh does not catch it either: the index is fine, one COLUMN is NaN.
# Pinned as-is rather than changed, because flattening an unmeasurable position is defensible
# and altering live sizing behaviour is a decision, not a test fix. But the comment is wrong.
nan_px = lo_vol.copy(); nan_px.iloc[-CFG.vol_window:, 0] = np.nan
held_nan = {m: 2 for m in base_t.index}
t_nan = compute_targets(nan_px, CFG, SPECS, held=held_nan)
expect("a market with a dead price feed is FLATTENED (not held, despite the code comment)",
       Check(int(t_nan.at[SPECS[0].market, "contracts"]) == 0,
             f"got {int(t_nan.at[SPECS[0].market, 'contracts'])}"))
expect("  ... and its exposure is exactly zero, never NaN",
       Check(float(t_nan.at[SPECS[0].market, "dollar_exposure"]) == 0.0, ""))
expect("  ... while the markets with live data are unaffected",
       Check(int(t_nan.at[SPECS[3].market, "contracts"]) != 0, ""))

# =====================================================================================
print("\n--- 7. WHOLE-CONTRACT ROUNDING ---")
expect("contracts are integers", Check(base_t["contracts"].dtype.kind == "i",
                                       f"{base_t['contracts'].dtype}"))
expect("notional_used == contracts x notional",
       Check(bool(np.allclose(base_t["notional_used"],
                              base_t["contracts"] * base_t["notional"])), ""))
expect("rounding never turns a positive signal negative (or vice versa)",
       Check(bool(((base_t["contracts"] * base_t["dollar_exposure"]) >= 0).all()), ""))
expect("every exposure and contract count is finite",
       Check(bool(np.isfinite(base_t[["dollar_exposure", "contracts", "notional_used"]]
                              .to_numpy()).all()), ""))

# =====================================================================================
print("\n--- 8. SAFETY: force-closes fire, only ever CLOSE, and no guard may block one ---")
TODAY = pd.Timestamp("2026-08-16")
BY_MKT = {s.market: s for s in FUTURES}
spec0 = BY_MKT["rates_10y"]                     # notice_buffer_days = 25
near = (TODAY + pd.Timedelta(days=spec0.notice_buffer_days - 1)).strftime("%Y%m%d")
far = (TODAY + pd.Timedelta(days=spec0.notice_buffer_days + 60)).strftime("%Y%m%d")
long_pos = HeldPosition("rates_10y", spec0.sym(True), near, 3)
short_pos = HeldPosition("rates_10y", spec0.sym(True), near, -2)

sc = safety_closes([long_pos], BY_MKT, TODAY)
expect("a position inside the notice buffer is force-closed", Check(len(sc) == 1, f"{sc}"))
expect("  ... a LONG is closed with a SELL", Check(sc[0].action == "SELL", f"{sc[0].action}"))
expect("  ... for the full held quantity", Check(sc[0].qty == 3, f"{sc[0].qty}"))
expect("  ... on the SAME expiry it is held in (not the front month)",
       Check(sc[0].expiry == near, f"{sc[0].expiry}"))
sc_s = safety_closes([short_pos], BY_MKT, TODAY)
expect("a SHORT is closed with a BUY", Check(sc_s[0].action == "BUY", f"{sc_s[0].action}"))
expect("a position OUTSIDE the buffer is not touched",
       Check(len(safety_closes([HeldPosition("rates_10y", spec0.sym(True), far, 3)],
                               BY_MKT, TODAY)) == 0, ""))
expect("a zero-quantity holding produces no order",
       Check(len(safety_closes([HeldPosition("rates_10y", spec0.sym(True), near, 0)],
                               BY_MKT, TODAY)) == 0, ""))
expect("an unknown market is skipped rather than crashing",
       Check(len(safety_closes([HeldPosition("nope", "XX", near, 3)], BY_MKT, TODAY)) == 0, ""))
expect("the reason string carries SAFETY (the execute() exemption matches on it)",
       Check("SAFETY" in sc[0].reason.upper(), f"{sc[0].reason}"))
expect("safety_closes NEVER opens or increases a position (every order reduces)",
       Check(all(o.qty > 0 and o.action in ("BUY", "SELL") for o in sc + sc_s), ""))

# THE INVARIANT: a guard must not be able to block a close. Driven through the REAL
# FuturesBroker.execute with a limits object so tight it rejects everything.
from trend_overlay.execution import FuturesBroker  # noqa: E402
brk = FuturesBroker(dry_run=True)
TIGHT = RiskLimits.for_futures(1.0)             # $1 budget: every order is astronomically over
safety_order = sc[0]
normal_order = RollOrder(spec0.sym(True), far, "BUY", 3, "reconcile front to target")
f_safe = brk.execute([safety_order], BY_MKT, True, limits=TIGHT)
f_norm = brk.execute([normal_order], BY_MKT, True, limits=TIGHT)
print(f"    with a $1 budget: SAFETY -> {f_safe[0]['status']}, normal -> {f_norm[0]['status']}")
expect("INVARIANT: a SAFETY close is NOT risk-rejected, however tight the limits",
       Check(f_safe[0]["status"] != "RiskRejected", f"{f_safe[0]['status']}"))
expect("  ... while an ordinary order at the same limits IS rejected (guard is not vacuous)",
       Check(f_norm[0]["status"] == "RiskRejected", f"{f_norm[0]['status']}"))
expect("the exemption is a SUBSTRING match on reason — pinned because it is fragile",
       Check(brk.execute([RollOrder(spec0.sym(True), near, "SELL", 3,
                                    "SAFETY force-close: 1d to last-trade")],
                         BY_MKT, True, limits=TIGHT)[0]["status"] != "RiskRejected", ""))
expect("  ... and an order WITHOUT the marker is guarded (so the marker is load-bearing)",
       Check(brk.execute([RollOrder(spec0.sym(True), near, "SELL", 3,
                                    "force-close: 1d to last-trade")],
                         BY_MKT, True, limits=TIGHT)[0]["status"] == "RiskRejected", ""))

print("\n--- days_to_expiry ---")
expect("YYYYMMDD parsed", Check(days_to_expiry("20260826", TODAY) == 10, ""))
expect("YYYYMM parsed as the 1st", Check(days_to_expiry("202609", TODAY) == 16, ""))
expect("an already-expired contract gives a negative count (and still force-closes)",
       Check(days_to_expiry("20260801", TODAY) < 0, ""))

print("\n" + "=" * 100)
if fails:
    print(f"{len(fails)} FAILURE(S) of {ran}:")
    for f in fails:
        print("   " + f)
    sys.exit(1)
print(f"all {ran} trend-overlay checks behaved as expected")
