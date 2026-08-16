"""Mutation-test `scripts/test_trend_sizing.py`: seed real faults, demand the suite catches them.

WHY. This sleeve's two known bugs were both SILENT — a circular vol-target that made the scale
a permanent 1.0, and an est_vol that assumed zero correlation. Neither raised, neither looked
wrong in a log, and `--selftest` passed cleanly through both. A suite claiming to cover them is
worthless unless breaking the code on purpose makes it fail.

Every fault must be CAUGHT. A survivor means that case is decoration. Non-zero exit, so this can
gate a release.

Run: python3 scripts/mutate_trend_sizing.py
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
TARGET = ROOT / "trend_overlay" / "execution.py"
SUITE = ["python3", "scripts/test_trend_sizing.py"]

MUTATIONS = [
    # --- BUG 1: the circular vol-target (gross hit 4.8x budget, silently) ---
    ("    est_vol = _portfolio_vol(w, sig_vec, rets[[s.proxy_etf for s in specs]], cfg)",
     "    est_vol = cfg.target_vol",
     "vol-target made circular again (est_vol == target, scale always 1.0)"),
    ("    scale = float(np.clip(cfg.target_vol / est_vol, lo, hi)) if est_vol > 0 else 0.0",
     "    scale = 1.0",
     "vol-target scale hard-wired to 1.0 (the no-op the bug produced)"),

    # --- BUG 2: est_vol assuming zero correlation (realised 12-14% vs a 10% target) ---
    ("    var = float(w @ (D @ R @ D) @ w)",
     "    var = float(np.nansum((w * vols) ** 2))",
     "covariance replaced by the independence formula (the original bug)"),
    ("    R = cfg.corr_weight * R + (1.0 - cfg.corr_weight) * target",
     "    R = np.eye(n)",
     "correlation matrix replaced by the identity (off-diagonals dropped)"),
    ("    R = rets.tail(cfg.corr_window).corr().values",
     "    R = np.eye(len(w))",
     "sample correlation never computed"),

    # --- exposure guards ---
    ("            df[\"dollar_exposure\"] *= lim / gross",
     "            pass",
     "gross cap computed but never applied"),
    ("        lim = cfg.gross_cap * cfg.budget * cfg.overlay_multiple",
     "        lim = 1e12",
     "gross cap raised to infinity"),
    # NOT LISTED: disabling the PRE-scale per-market cap ("lim = cfg.per_market_cap *
    # cfg.budget" -> 1e12). Measured 2026-08-16: it produces byte-identical output on every
    # scenario tried -- capped and uncapped books, scale above and below 1, correlated and
    # independent, overlay 1.0 and 3.0. The POST-scale cap (added after the 2026-08-11 hole)
    # subsumes it, and the scale_clip floor absorbs its only other effect, which is on est_vol
    # via w. It is defensive redundancy, not a live guard. Listing a mutation nothing can catch
    # would either force a contrived test or sit as a permanent known-survivor; both are worse
    # than saying so here.
    ("        lim2 = cfg.per_market_cap * cfg.budget * cfg.overlay_multiple",
     "        lim2 = 1e12",
     "POST-scale per-market cap disabled (the scale>1 hole that was found live)"),
    ("        too_big = (df[\"contracts\"].abs() * df[\"notional\"]) > lim3 * 1.001",
     "        too_big = (df[\"contracts\"].abs() * df[\"notional\"]) > 1e12",
     "contract-level cap never detects an over-cap position"),
    ("        lim3 = cfg.per_market_cap * cfg.budget * cfg.overlay_multiple",
     "        lim3 = 1e12",
     "contract-level cap limit set to infinity"),

    # --- vol floor ---
    ("        vol_used = vol.where(floor.isna(), np.maximum(vol, floor))",
     "        vol_used = vol",
     "vol floor computed but never applied (divide-by-small returns)"),
    ("        floor = vol.expanding(min_periods=252).quantile(cfg.vol_floor_pct)",
     "        floor = vol.expanding(min_periods=252).quantile(0.0)",
     "vol floor percentile set to 0 (floors at the all-time minimum, i.e. never)"),

    # --- hysteresis ---
    ("                ct.append(int(np.round(x)) if abs(x - n) >= band else n)",
     "                ct.append(int(np.round(x)))",
     "hysteresis band ignored (always trades to round(target))"),
    ("        band = float(cfg.hysteresis_band)",
     "        band = 1e9",
     "hysteresis band infinite (never trades)"),

    # --- SAFETY invariant ---
    ("            if limits is not None and \"SAFETY\" not in (o.reason or \"\").upper():",
     "            if limits is not None:",
     "SAFETY exemption removed -- a forced close becomes blockable"),
    ("        if d <= spec.notice_buffer_days:",
     "        if d <= 0:",
     "notice buffer ignored (only closes AFTER last-trade, too late)"),
    ("            out.append(RollOrder(h.ib_symbol, h.expiry, \"SELL\" if h.qty > 0 else \"BUY\",",
     "            out.append(RollOrder(h.ib_symbol, h.expiry, \"BUY\" if h.qty > 0 else \"SELL\",",
     "safety close direction inverted (would DOUBLE the position, not close it)"),
]


def main() -> int:
    original = TARGET.read_text()
    results = []
    print("=" * 100)
    print(f"MUTATION TEST — {TARGET.relative_to(ROOT)} against {SUITE[1]}")
    print("=" * 100)
    print(f"  {len(MUTATIONS)} seeded faults; every one must be CAUGHT\n")
    try:
        for find, repl, why in MUTATIONS:
            if find not in original:
                results.append((why, None))
                print(f"  [ ?? ] {why:80} PATTERN MISSING")
                continue
            TARGET.write_text(original.replace(find, repl, 1))
            for pyc in ROOT.rglob("*.pyc"):
                pyc.unlink(missing_ok=True)
            r = subprocess.run(SUITE, cwd=ROOT, capture_output=True, text=True)
            caught = r.returncode != 0
            results.append((why, caught))
            print(f"  [{'ok  ' if caught else 'FAIL'}] {why:80} "
                  f"{'CAUGHT' if caught else '*** SURVIVED ***'}")
    finally:
        TARGET.write_text(original)
        for pyc in ROOT.rglob("*.pyc"):
            pyc.unlink(missing_ok=True)

    survived = [w for w, c in results if c is False]
    missing = [w for w, c in results if c is None]
    print("\n" + "=" * 100)
    if missing:
        print(f"{len(missing)} mutation(s) could not be applied — the code moved, update this file:")
        for w in missing:
            print("   " + w)
    if survived:
        print(f"{len(survived)} MUTATION(S) SURVIVED — those cases cannot fail and are decoration:")
        for w in survived:
            print("   " + w)
        return 1
    if missing:
        return 1
    r = subprocess.run(SUITE, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print("RESTORE FAILED — the suite does not pass on the original file")
        return 1
    print(f"all {len(MUTATIONS)} seeded faults were caught; suite restored and green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
