"""Mutation-test `scripts/test_order_links.py`: the order -> IB record links (orderRef, execution ids, conid, commission, currency).

Seeds real faults into a TEMP COPY of the repo (the real files are never edited) and demands the
suite catches every one. Engine: _mutate_repo_core.py.

Run: python scripts/mutate_order_links.py
"""
from __future__ import annotations

import sys

from _mutate_repo_core import run

MUTATIONS = [
    ('trend_overlay/execution.py',
     '                    order.orderRef = self.order_ref\n',
     '                    pass\n',
     'broker never stamps orderRef on the order'),
    ('trend_overlay/execution.py',
     'if getattr(self, "order_ref", None):    # a missing tag',
     'if self.order_ref:    # a missing tag',
     'a broker without a tag raises (a missing tag BLOCKS the trade)'),
    ('trend_overlay/execution.py',
     'while not getattr(trade, "fills", None) and waited < wait:',
     'while not getattr(trade, "fills", None) and waited < 0:',
     "no wait for execution details that trail 'Filled'"),
    ('trend_overlay/execution.py',
     '"exec_ids": ex, "order_ref": getattr(order, "orderRef", "") or "",\n                              "conid"',
     '"exec_ids": [], "order_ref": getattr(order, "orderRef", "") or "",\n                              "conid"',
     'a fill drops its execution ids'),
    ('trend_overlay/execution.py',
     '"exec_ids": ex, "order_ref": getattr(order, "orderRef", "") or "",\n                              "conid"',
     '"exec_ids": ex, "order_ref": "trend-overlay:CLAIMED",\n                              "conid"',
     'a fill claims a tag the order did not carry'),
    ('scripts/run_trend_paper.py',
     '    b.order_ref = ORDER_REF\n',
     '',
     'RUNNER: make_broker never sets the tag'),
    ('scripts/run_trend_paper.py',
     '    broker = make_broker(port=',
     '    broker = FuturesBroker(port=',
     'RUNNER: main() bypasses make_broker (untagged broker)'),
    ('scripts/run_trend_paper.py',
     'ORDER_REF = f"trend-overlay:{RUN_ID}"',
     'ORDER_REF = f"trend:{RUN_ID}"',
     'RUNNER: tag names the wrong strategy'),
    ('scripts/run_trend_paper.py',
     'order_ref=f.get("order_ref", ""), exec_ids=f.get("exec_ids"),',
     'order_ref="", exec_ids=f.get("exec_ids"),',
     'RUNNER: booking drops the orderRef'),
    ('scripts/run_trend_paper.py',
     'order_ref=f.get("order_ref", ""), exec_ids=f.get("exec_ids"),',
     'order_ref=f.get("order_ref", ""), exec_ids=None,',
     'RUNNER: booking drops the execution ids'),
    ('scripts/run_trend_paper.py',
     '        if f["fill_price"]:\n            state.record_fill(',
     '        if True:\n            state.record_fill(',
     'RUNNER: unfilled orders booked into the ledger'),
    ('scripts/run_trend_paper.py',
     '                book_fills(state, fills, today, todays_orders)\n',
     '',
     'RUNNER: main() stops booking fills'),
    ('trend_overlay/state.py',
     '"exec_ids": list(exec_ids or []),   # = Flex ibExecID',
     '"exec_ids": [],   # = Flex ibExecID',
     'trade_log drops the execution ids'),
    ('trend_overlay/execution.py',
     '                              "commission": self._commission(trade) if st == "Filled" else None,',
     '                              "commission": None,',
     'commission never recorded'),
    ('trend_overlay/execution.py',
     '                              "conid": int(getattr(q[0], "conId", 0) or 0),',
     '                              "conid": 0,',
     'conid never recorded'),
    ('trend_overlay/execution.py',
     '                if fills and all(r is not None and getattr(r, "execId", "") for r in reps):',
     '                if fills:',
     'commission reports counted before IB sent them'),
    ('trend_overlay/state.py',
     '                               "conid": int(conid or 0), "commission": commission,',
     '                               "conid": 0, "commission": None,',
     'trade_log drops conid and commission'),
    ('scripts/run_trend_paper.py',
     '                              conid=f.get("conid") or 0, commission=f.get("commission"),',
     '                              conid=0, commission=None,',
     'runner booking drops conid and commission'),
]

if __name__ == "__main__":
    sys.exit(run("test_order_links.py", MUTATIONS))
