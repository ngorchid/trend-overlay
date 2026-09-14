"""Tests for the daily-P&L line added to the trend email (2026-09-14 go-live). No network."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

import os  # noqa: E402
from trend_overlay.state import TrendState  # noqa: E402
from trend_overlay.email_report import _daily_pnl, build_email_body  # noqa: E402

_fails, _ran = [], 0


def check(label, cond, detail=""):
    global _ran
    _ran += 1
    if not cond:
        _fails.append(f"{label} | {detail}")
    print(f"  [{'ok ' if cond else 'FAIL'}] {label}")


def _st(rp, hist):
    s = TrendState(inception_date="2026-09-01", realized_pnl=rp)
    s.nav_history = hist
    return s


print("=" * 80)
print("TREND EMAIL — daily P&L (change in total P&L since the last run)")
print("=" * 80)

# Baseline is the most recent PRIOR-day snapshot; today's total minus it.
s = _st(500.0, [{"date": "2026-09-11", "total_pnl": 1200.0}, {"date": "2026-09-12", "total_pnl": 1450.0}])
check("daily = today's total - last prior snapshot", _daily_pnl(s, 580.0, "2026-09-14") == -870.0,
      f"{_daily_pnl(s, 580.0, '2026-09-14')}")

# record_snapshot may already have written TODAY's row — it must be excluded from the baseline,
# otherwise daily P&L would always read 0.
s2 = _st(500.0, [{"date": "2026-09-12", "total_pnl": 1450.0}, {"date": "2026-09-14", "total_pnl": 580.0}])
check("today's own snapshot is excluded from the baseline", _daily_pnl(s2, 580.0, "2026-09-14") == -870.0,
      f"{_daily_pnl(s2, 580.0, '2026-09-14')}")

# First ever run: no prior snapshot -> None (rendered as '— (first run)', never a fake 0).
check("first run -> None", _daily_pnl(_st(0.0, []), 80.0, "2026-09-14") is None)
check("only-today history -> None", _daily_pnl(
    _st(0.0, [{"date": "2026-09-14", "total_pnl": 80.0}]), 80.0, "2026-09-14") is None)

# A None total_pnl row (defensive) is skipped, not treated as 0.
s3 = _st(0.0, [{"date": "2026-09-12", "total_pnl": None}, {"date": "2026-09-13", "total_pnl": 100.0}])
check("rows with a null total_pnl are skipped", _daily_pnl(s3, 250.0, "2026-09-14") == 150.0,
      f"{_daily_pnl(s3, 250.0, '2026-09-14')}")

# The body renders the daily row, and the BOOK_LABEL drives LIVE vs Paper wording.
os.environ["BOOK_LABEL"] = "LIVE"
body = build_email_body(s, [{"market": "fx_aud", "symbol": "M6A", "contracts": 4,
                             "avg_price": 0.71, "mark": 0.71, "unrealized_pnl": 80.0}],
                        [], 0.0, 0.0, "2026-09-14")
check("body shows the Daily P&L row", "Daily P&amp;L (since last run)" in body)
check("LIVE label in header", "Trend Overlay LIVE" in body)
check("LIVE footer says real capital", "Real capital." in body)
os.environ["BOOK_LABEL"] = "Paper"
check("Paper label when BOOK_LABEL=Paper", "Trend Overlay Paper" in
      build_email_body(s, [], [], 0.0, 0.0, "2026-09-14"))

print("\n" + "=" * 80)
if _fails:
    print(f"{len(_fails)} FAILURE(S) of {_ran}:")
    for f in _fails:
        print("   " + f)
    sys.exit(1)
print(f"all {_ran} email checks behaved as expected")
