"""Tests for the order -> IB record links: orderRef tag and execution ids. No IB, no network.

Every futures order must carry orderRef "trend-overlay:<run id>" and every fill's IB execution
ids (= Flex `ibExecID`) must reach trade_log, or the audit trail cannot tie this sleeve's records
to IB's. Driven through the REAL FuturesBroker and TrendState against a fake IB.

Run: python scripts/test_order_links.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from trend_overlay.contracts import BY_MARKET  # noqa: E402
from trend_overlay.execution import FuturesBroker, RollOrder  # noqa: E402
from trend_overlay.state import TrendState  # noqa: E402

REF = "trend-overlay:20261006-203001"
_fails: list[str] = []
_ran = 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global _ran
    _ran += 1
    if not cond:
        _fails.append(f"{label}  | {detail}")
    print(f"  [{'ok ' if cond else 'FAIL'}] {label}" + ("" if cond else f"   <- {detail}"))


class FakeIB:
    """Fills every order at once; `fills_after` sleeps before the execDetails arrive."""

    def __init__(self, fills_after: int = 0):
        self.sent, self.n, self.fills_after, self._pending = [], 0, fills_after, None

    def qualifyContracts(self, *cs):
        for c in cs:
            c.conId = 1
        return list(cs)

    def placeOrder(self, contract, order):
        self.n += 1
        self.sent.append(order)
        fills = [NS(execution=NS(execId=f"0001.{self.n}.01"))]
        t = NS(order=order, fills=[] if self.fills_after else fills,
               orderStatus=NS(status="Filled", avgFillPrice=88.48, filled=order.totalQuantity))
        self._pending = [t, fills, self.fills_after]
        return t

    def sleep(self, s):
        if self._pending and not self._pending[0].fills:
            self._pending[2] -= 1
            if self._pending[2] <= 0:
                self._pending[0].fills = self._pending[1]

    def cancelOrder(self, o):
        pass


def run(ib, ref=REF, set_ref=True):
    b = FuturesBroker(dry_run=False)
    b.ib = ib
    if set_ref:
        b.order_ref = ref
    return b.execute([RollOrder("MCL", "20261119", "BUY", 1, "test")], BY_MARKET, True)


print("ORDER TAGS")
ib = FakeIB()
fills = run(ib)
check("a futures order carries orderRef", ib.sent[-1].orderRef == REF, repr(ib.sent[-1].orderRef))
nb = FuturesBroker.__new__(FuturesBroker)          # built without __init__: no order_ref attribute
nb.ib, nb.dry_run, nb._front = FakeIB(), False, {}
r = nb.execute([RollOrder("MCL", "20261119", "BUY", 1, "test")], BY_MARKET, True)
check("a broker without a tag still places the order (a tag never blocks a trade)",
      r and r[0].get("status") == "Filled", str(r))

print("\nEXECUTION IDS")
check("a filled order returns its IB execution ids", fills[0].get("exec_ids") == ["0001.1.01"], str(fills))
late = run(FakeIB(fills_after=3))
check("execution details arriving just after 'Filled' are still captured",
      late[0].get("exec_ids") == ["0001.1.01"], str(late))

print("\nLEDGER")
st = TrendState()
f = fills[0]
st.record_fill("oil", 1, f["fill_price"], 100, "2026-10-06", "MCL", "20261119", "test",
               order_ref=REF, exec_ids=f.get("exec_ids"))
row = st.trade_log[-1]
check("trade_log keeps the orderRef", row.get("order_ref") == REF, str(row))
check("trade_log keeps the execution ids", row.get("exec_ids") == ["0001.1.01"], str(row))

print("\n" + "=" * 88)
if _fails:
    print(f"{len(_fails)} FAILURE(S) of {_ran}:")
    for x in _fails:
        print("   " + x)
    sys.exit(1)
print(f"all {_ran} order-link checks behaved as expected")
