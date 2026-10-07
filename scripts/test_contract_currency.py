"""Guard: every trend-overlay contract must be USD-denominated.

WHY A TEST AND NOT A NEW ATTRIBUTION RULE (owner's decision, 2026-10-07). The IB-records
attribution gives foreign-currency interest and FX revaluation to magic-formula, on the premise
that only magic-formula creates foreign cash. Trend could break that premise only by trading a
non-USD contract (variation margin and P&L would then settle in that currency). This suite fails
the moment one is added, so the premise is re-decided instead of silently wrong.

WHY A TEST AND NOT A STARTUP ASSERTION. An assertion in run_trend_paper.py would stop the whole
run -- including the safety closes and rolls that prevent physical delivery. No guard may block a
close, so the check gates the CHANGE (this suite runs before any deploy) rather than the run.

NB the FX futures (6E/6A/6J) are USD-quoted but PHYSICALLY DELIVER foreign currency at expiry; the
per-market notice buffer and safety close exist so that never happens. A delivery would put
foreign cash in the account and, under the attribution rule, it would be charged to magic-formula.

Run: python scripts/test_contract_currency.py
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

from trend_overlay.contracts import BY_MARKET, FUTURES  # noqa: E402

_fails: list[str] = []
_ran = 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global _ran
    _ran += 1
    if not cond:
        _fails.append(f"{label}  | {detail}")
    print(f"  [{'ok ' if cond else 'FAIL'}] {label}" + ("" if cond else f"   <- {detail}"))


non_usd = [f"{s.market} ({s.symbol}) {s.currency}" for s in FUTURES if s.currency != "USD"]
check("every trend contract is USD-denominated", non_usd == [], ", ".join(non_usd))
check("...checked over the whole contract table (not an empty list)",
      len(FUTURES) >= 10 and len(BY_MARKET) == len(FUTURES), f"{len(FUTURES)} specs")
fx = [s for s in FUTURES if s.market.startswith("fx_")]
check("FX futures (delivering foreign currency) keep a notice buffer of at least 5 days",
      fx and all(s.notice_buffer_days >= 5 for s in fx),
      str([(s.market, s.notice_buffer_days) for s in fx]))

print("\nRUNTIME WARNING (never stops a run)")
import dataclasses  # noqa: E402
import inspect  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
import run_trend_paper as runner  # noqa: E402

LOG: list[str] = []
_orig = runner.logging.warning
runner.logging.warning = lambda f, *a: LOG.append(f % a if a else f)
check("the live table produces no warning", runner.warn_non_usd_contracts() == [] and LOG == [], str(LOG))
eur = [dataclasses.replace(FUTURES[0], currency="EUR")] + FUTURES[1:]
out = runner.warn_non_usd_contracts(eur)
check("a non-USD contract is named in a WARNING (so it reaches the email)",
      out == ["equity_us (ES) EUR"] and any("NON-USD" in x and "equity_us" in x for x in LOG), str(LOG))
runner.logging.warning = _orig
src = inspect.getsource(runner.main)
check("main() warns on every run, before the kill switch (so even a halted run says it)",
      0 <= src.find("warn_non_usd_contracts()") < src.find("halt_state(ROOT)"), "")
check("...and the warning cannot stop the run (no raise / exit / return in it)",
      not any(k in inspect.getsource(runner.warn_non_usd_contracts) for k in ("raise", "sys.exit", "return None")),
      "")

print("\n" + "=" * 88)
if _fails:
    print(f"{len(_fails)} FAILURE(S) of {_ran}:")
    for f in _fails:
        print("   " + f)
    sys.exit(1)
print(f"all {_ran} contract-currency checks behaved as expected")
