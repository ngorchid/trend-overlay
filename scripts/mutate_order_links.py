"""Mutation-test `scripts/test_order_links.py`: seed real faults, demand the suite catches them.

WHY. The order -> IB-record links (orderRef tag, execution ids) fail SILENTLY: an untagged order
still fills, a dropped execution id still books. The first version of test_order_links.py handed
the broker its tag and the ledger its ids by hand, so deleting the runner's one tagging line, or
dropping the ids on the way into the ledger, survived with the suite green (measured 2026-10-06).
A passing suite is only evidence if breaking each link on purpose makes it fail.

Never edits the real files: the repo's code (no data, results, .env or .git) is copied to a temp
dir once and each mutant is written there, so an interrupted run cannot leave a broken file for
the scheduled task.

Non-zero exit if any fault survives, a pattern no longer matches exactly once (the code moved:
update this file), or a mutant does not compile (it would be "caught" by the SyntaxError alone).

Run: python scripts/mutate_order_links.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUITE = [sys.executable, "scripts/test_order_links.py"]
IGNORE = shutil.ignore_patterns(".git", "data", "results", "__pycache__", ".idea", ".claude",
                                ".env", ".venv", "venv", "*.parquet", "*.pkl")

# (file, find, replace, what the fault means)
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
     '"exec_ids": ex, "order_ref": getattr(order, "orderRef", "") or ""})',
     '"exec_ids": [], "order_ref": getattr(order, "orderRef", "") or ""})',
     'a fill drops its execution ids'),
    ('trend_overlay/execution.py',
     '"exec_ids": ex, "order_ref": getattr(order, "orderRef", "") or ""})',
     '"exec_ids": ex, "order_ref": "trend-overlay:CLAIMED"})',
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
     'order_ref=f.get("order_ref", ""), exec_ids=f.get("exec_ids"))',
     'order_ref="", exec_ids=f.get("exec_ids"))',
     'RUNNER: booking drops the orderRef'),
    ('scripts/run_trend_paper.py',
     'order_ref=f.get("order_ref", ""), exec_ids=f.get("exec_ids"))',
     'order_ref=f.get("order_ref", ""), exec_ids=None)',
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
     '"exec_ids": list(exec_ids or [])})',
     '"exec_ids": []})',
     'trade_log drops the execution ids'),
]


def main() -> int:
    base = subprocess.run(SUITE, cwd=ROOT, capture_output=True, text=True)
    if base.returncode != 0:
        print("the suite does not pass on the ORIGINAL code — fix that first")
        print(base.stdout[-2000:])
        return 1
    print("=" * 100)
    print(f"  {len(MUTATIONS)} seeded faults; every one must be CAUGHT\n")
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "repo"
        shutil.copytree(ROOT, work, ignore=IGNORE)
        for rel, find, repl, why in MUTATIONS:
            f = work / rel
            orig = f.read_text(encoding="utf-8")
            n = orig.count(find)
            if n != 1:
                results.append((why, None))
                print(f"  [ ?? ] {why:80} PATTERN {'MISSING' if n == 0 else f'AMBIGUOUS x{n}'}")
                continue
            mutant = orig.replace(find, repl, 1)
            try:                    # a mutant that does not even compile is not a test of anything
                compile(mutant, rel, "exec")
            except SyntaxError as e:
                results.append((why, None))
                print(f"  [ ?? ] {why:80} INVALID MUTANT ({e.msg})")
                continue
            f.write_text(mutant, encoding="utf-8")
            try:
                r = subprocess.run(SUITE, cwd=work, capture_output=True, text=True)
            finally:
                f.write_text(orig, encoding="utf-8")
            caught = r.returncode != 0
            results.append((why, caught))
            print(f"  [{'ok  ' if caught else 'FAIL'}] {why:80} "
                  f"{'CAUGHT' if caught else '*** SURVIVED ***'}")
    survived = [w for w, c in results if c is False]
    missing = [w for w, c in results if c is None]
    print("\n" + "=" * 100)
    if missing:
        print(f"{len(missing)} mutation(s) could not be applied (pattern moved or mutant invalid) "
              "— update this file:")
        for w in missing:
            print("   " + w)
    if survived:
        print(f"{len(survived)} MUTATION(S) SURVIVED — those cases cannot fail and are decoration:")
        for w in survived:
            print("   " + w)
    if survived or missing:
        return 1
    print(f"all {len(MUTATIONS)} seeded faults were caught; the real files were never modified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
