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

check("a fill carries the tag its ORDER actually had", fills[0].get("order_ref") == REF, str(fills))
check("...and an untagged order's fill says so (never claims a tag IB did not get)",
      r and r[0].get("order_ref") == "", str(r))

# ---------------------------------------------------------------------------------------------
# RUNNER WIRING. Everything above hands the broker its tag by hand, so it cannot see the runner
# forgetting to: deleting that one line in run_trend_paper.py sent every order out untagged
# while all of the above stayed green. These pin the runner's own wiring.
print("\nRUNNER WIRING")
import inspect  # noqa: E402
import re  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
import run_trend_paper as runner  # noqa: E402

check("ORDER_REF is 'trend-overlay:<YYYYmmdd-HHMMSS>'",
      re.fullmatch(r"trend-overlay:\d{8}-\d{6}", runner.ORDER_REF) is not None, runner.ORDER_REF)
check("make_broker tags the broker with this run's ORDER_REF",
      runner.make_broker(dry_run=True).order_ref == runner.ORDER_REF, "")
_src = inspect.getsource(runner.main)
check("main() builds its broker through make_broker, never FuturesBroker() directly",
      "make_broker(" in _src and "FuturesBroker(" not in _src, "")
check("main() books fills through book_fills", "book_fills(" in _src, "")

print("\nRUNNER BOOKING -> trade_log")
_st2, _orders = TrendState(), []
_unfilled = {**fills[0], "fill_price": None, "status": "Submitted", "exec_ids": []}
runner.book_fills(_st2, [fills[0], _unfilled], "2026-10-06", _orders)
check("book_fills writes the order's orderRef into trade_log",
      len(_st2.trade_log) == 1 and _st2.trade_log[-1].get("order_ref") == REF, str(_st2.trade_log))
check("...and its execution ids",
      len(_st2.trade_log) == 1 and _st2.trade_log[-1].get("exec_ids") == ["0001.1.01"],
      str(_st2.trade_log))
check("an unfilled order is listed for the email but NOT booked",
      len(_orders) == 2 and len(_st2.trade_log) == 1, f"{len(_orders)} listed, "
      f"{len(_st2.trade_log)} booked")

print("\n" + "=" * 88)
if _fails:
    print(f"{len(_fails)} FAILURE(S) of {_ran}:")
    for x in _fails:
        print("   " + x)
    sys.exit(1)
print(f"all {_ran} order-link checks behaved as expected")
