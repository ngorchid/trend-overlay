"""Tests for TrendState.resync_to_broker — the ledger-vs-broker drift fix.

WHY THIS EXISTS. The P&L ledger drifts from IB whenever a broker position changes without a
recorded fill (an uncaptured fill price, a run that did not save, a manual edit). It cannot
self-correct because it only moves by NEW fills, and the reconcile step only REPORTED the gap
— so rates_10y emailed a "PHANTOM" every run for weeks. resync_to_broker snaps the ledger to
IB (source of truth for positions), adopts the broker's cost basis, and — crucially — does NOT
fabricate realised P&L for the unrecorded change. These pin exactly that behaviour.

Pure state logic, no IB. Real asserts, non-zero exit on failure.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from trend_overlay.state import MarketLedger, TrendState  # noqa: E402

_fails: list[str] = []
_ran = 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global _ran
    _ran += 1
    if not cond:
        _fails.append(f"{label}  | {detail}")
    print(f"  [{'ok ' if cond else 'FAIL'}] {label}" + ("" if cond else f"   <- {detail}"))


def _state(**ledger) -> TrendState:
    s = TrendState(realized_pnl=1234.56)
    s.ledger = {m: MarketLedger(qty=q, avg_price=a, multiplier=mult)
                for m, (q, a, mult) in ledger.items()}
    return s


print("=" * 88)
print("TrendState.resync_to_broker — snap the ledger to IB, never fabricate P&L")
print("=" * 88)

# 1. THE PHANTOM (the actual bug): ledger short 1, broker flat -> ledger goes to 0.
s = _state(rates_10y=(-1.0, 107.65625, 1000.0))
notes = s.resync_to_broker({}, "2026-09-08")   # broker reports NO rates_10y row => flat
check("phantom -1 vs broker 0 -> ledger qty 0", s.ledger["rates_10y"].qty == 0.0,
      str(s.ledger["rates_10y"]))
check("...avg_price cleared on flat", s.ledger["rates_10y"].avg_price == 0.0, str(s.ledger["rates_10y"]))
check("...realized_pnl is NOT fabricated", s.realized_pnl == 1234.56, f"{s.realized_pnl}")
check("...the correction is logged for audit",
      any(t.get("reason", "").startswith("RESYNC") for t in s.trade_log), str(s.trade_log[-1:]))
check("...and it returns a human note", len(notes) == 1 and "rates_10y" in notes[0], str(notes))

# 2. THE ORPHAN: broker holds 4 the ledger never recorded -> adopt qty AND the broker cost basis.
s = _state(rates_10y=(0.0, 0.0, 1000.0))
s.resync_to_broker({"rates_10y": (4.0, 108.25)}, "2026-09-08", {"rates_10y": 1000.0})
check("orphan 0 vs broker 4 -> ledger qty 4", s.ledger["rates_10y"].qty == 4.0, str(s.ledger["rates_10y"]))
check("...adopts the broker's average cost as the go-forward basis",
      s.ledger["rates_10y"].avg_price == 108.25, str(s.ledger["rates_10y"]))
check("...multiplier preserved", s.ledger["rates_10y"].multiplier == 1000.0, str(s.ledger["rates_10y"]))
check("...realized_pnl still untouched", s.realized_pnl == 1234.56, f"{s.realized_pnl}")

# 3. A market with NO prior ledger entry, held at the broker -> uses the supplied multiplier.
s = _state()
s.resync_to_broker({"fx_eur": (2.0, 1.16)}, "2026-09-08", {"fx_eur": 12500.0})
check("brand-new orphan market is created from the broker",
      s.ledger["fx_eur"].qty == 2.0 and s.ledger["fx_eur"].multiplier == 12500.0,
      str(s.ledger.get("fx_eur")))

# 4. IN SYNC: nothing to do -> no change, no note, no spurious trade_log entry.
s = _state(fx_aud=(3.0, 0.7192, 10000.0))
n = len(s.trade_log)
notes = s.resync_to_broker({"fx_aud": (3.0, 0.7192)}, "2026-09-08")
check("a market already in sync is a no-op (no notes)", notes == [], str(notes))
check("...and writes NO trade_log entry", len(s.trade_log) == n, f"{len(s.trade_log)} vs {n}")

# 5. MISMATCH reduce: ledger 3, broker 1 -> snap to 1 (does not book a fictitious close).
s = _state(fx_aud=(3.0, 0.7192, 10000.0))
s.resync_to_broker({"fx_aud": (1.0, 0.7192)}, "2026-09-08")
check("mismatch 3 vs 1 -> ledger qty 1", s.ledger["fx_aud"].qty == 1.0, str(s.ledger["fx_aud"]))
check("...realized_pnl untouched by the snap", s.realized_pnl == 1234.56, f"{s.realized_pnl}")

# 6. IDEMPOTENT: running the SAME resync twice changes nothing the second time.
s = _state(rates_10y=(-1.0, 107.65625, 1000.0))
s.resync_to_broker({}, "2026-09-08")
log_after_first = len(s.trade_log)
notes2 = s.resync_to_broker({}, "2026-09-09")
check("second identical resync is a no-op", notes2 == [] and len(s.trade_log) == log_after_first,
      f"notes={notes2} log={len(s.trade_log)}")

# 7. The signed_qty in the audit row is the actual correction delta (broker - ledger).
s = _state(rates_10y=(-1.0, 107.65625, 1000.0))
s.resync_to_broker({}, "2026-09-08")
row = s.trade_log[-1]
check("audit row records the correction delta (0 - (-1) = +1)", row["signed_qty"] == 1,
      str(row))
check("...and books zero realised P&L", row["realized_pnl"] == 0.0, str(row))

print("\n" + "=" * 88)
if _fails:
    print(f"{len(_fails)} FAILURE(S) of {_ran}:")
    for f in _fails:
        print("   " + f)
    sys.exit(1)
print(f"all {_ran} resync checks behaved as expected")
