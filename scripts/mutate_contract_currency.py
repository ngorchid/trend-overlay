"""Mutation-test `scripts/test_contract_currency.py`: the USD-only guard on trend-overlay contracts.

Seeds real faults into a TEMP COPY of the repo (the real files are never edited) and demands the
suite catches every one. Engine: _mutate_repo_core.py.

Run: python scripts/mutate_contract_currency.py
"""
from __future__ import annotations

import sys

from _mutate_repo_core import run

MUTATIONS = [
    ('trend_overlay/contracts.py',
     '"6E", "CME",   "USD"',
     '"6E", "CME",   "EUR"',
     'an FX future becomes EUR-denominated'),
    ('trend_overlay/contracts.py',
     '"ES", "CME",   "USD"',
     '"ES", "CME",   "GBP"',
     'the equity future becomes GBP-denominated'),
    ('trend_overlay/contracts.py',
     'FutureSpec("silver",    "SLV", "SI", "COMEX", "USD"',
     'FutureSpec("silver",    "SLV", "SI", "COMEX", "CHF"',
     'the LAST contract in the table becomes non-USD'),
    ('trend_overlay/contracts.py',
     '"6A", "CME",   "USD", 100000,66_000,  "M6A", 10000, 6_600,  notice_buffer_days=10)',
     '"6A", "CME",   "USD", 100000,66_000,  "M6A", 10000, 6_600,  notice_buffer_days=2)',
     "an FX future's notice buffer cut below 5 days"),
    ('scripts/run_trend_paper.py',
     '    warn_non_usd_contracts()\n    _halt, _hwhy = halt_state(ROOT)',
     '    _halt, _hwhy = halt_state(ROOT)',
     'the runtime warning is never called'),
    ('scripts/run_trend_paper.py',
     '           for s in (FUTURES if futures is None else futures) if s.currency != "USD"]',
     '           for s in (FUTURES if futures is None else futures) if s.currency == "JPY"]',
     'the runtime warning only looks for one currency'),
    ('scripts/run_trend_paper.py',
     '        logging.warning("NON-USD trend contract(s): %s',
     '        logging.info("NON-USD trend contract(s): %s',
     'the runtime warning is logged below WARNING (never emailed)'),
]

if __name__ == "__main__":
    sys.exit(run("test_contract_currency.py", MUTATIONS))
