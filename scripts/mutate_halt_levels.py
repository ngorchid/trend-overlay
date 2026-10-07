"""Mutation-test `scripts/test_halt_levels.py`: the kill-switch levels and documented sizing (trend).

Seeds real faults into a TEMP COPY of the repo (the real files are never edited) and demands the
suite catches every one. Engine: _mutate_repo_core.py.

Run: python scripts/mutate_halt_levels.py
"""
from __future__ import annotations

import sys

from _mutate_repo_core import run

MUTATIONS = [
    ('risk_guard.py',
     '    for name, mode in ((r / "HALT_HARD", HALT_HARD), (r / "HALT_ALL", HALT_ALL), (r / "HALT", HALT_NEW)):',
     '    for name, mode in ((r / "HALT_ALL", HALT_ALL), (r / "HALT", HALT_NEW)):',
     'the HALT_HARD file is not recognised'),
    ('risk_guard.py',
     '    for name, mode in ((r / "HALT_HARD", HALT_HARD), (r / "HALT_ALL", HALT_ALL), (r / "HALT", HALT_NEW)):',
     '    for name, mode in ((r / "HALT_ALL", HALT_ALL), (r / "HALT_HARD", HALT_HARD), (r / "HALT", HALT_NEW)):',
     'HALT_ALL takes precedence over HALT_HARD'),
    ('risk_guard.py',
     '    if env in ("hard", "3"):\n        return HALT_HARD, "TRADING_HALT=hard"\n',
     '',
     'TRADING_HALT=hard not recognised'),
    ('scripts/run_trend_paper.py',
     '    if _halt == HALT_HARD:\n        logging.error("HALTED (hard)',
     '    if False:\n        logging.error("HALTED (hard)',
     'HALT_HARD connects and trades'),
    ('scripts/run_trend_paper.py',
     '    halt_all = _halt == HALT_ALL',
     '    halt_all = False',
     'HALT_ALL trades normally'),
    ('scripts/run_trend_paper.py',
     '            if not args.safety_only and not halt_all:   # HALT_ALL needs no prices',
     '            if not args.safety_only:   # HALT_ALL needs no prices',
     'HALT_ALL downloads prices'),
    ('scripts/run_trend_paper.py',
     '    rolls = plan_roll_orders(held_by_market(held, use_micro), held_left, front, BY_MARKET,',
     '    rolls = plan_roll_orders({m: 2 * q for m, q in held_by_market(held, use_micro).items()}, held_left, front, BY_MARKET,',
     'HALT_ALL rolls resize the position'),
    ('scripts/run_trend_paper.py',
     '    return [("SAFETY", safety), ("ROLL (HALT_ALL)", rolls)]',
     '    return [("ROLL (HALT_ALL)", rolls)]',
     'HALT_ALL drops the safety closes'),
    ('scripts/run_trend_paper.py',
     '            if halt_all and not args.safety_only:\n                batches = halt_all_batches(',
     '            if False:\n                batches = halt_all_batches(',
     'HALT_ALL never uses the safety-only plan'),
    ('scripts/run_trend_paper.py',
     '    om, osrc = documented_sizing(ROOT, "trend-overlay", key="overlay_mult", env_var="OVERLAY_MULT")',
     '    om, osrc = float(os.getenv("OVERLAY_MULT", "1.0")), "env"',
     'OVERLAY_MULT read from env only'),
    ('scripts/run_trend_paper.py',
     '        _base = cfg.budget              # the documented budget',
     '        _base = float(os.getenv("BUDGET", "100000"))  # the documented budget',
     'the breaker re-reads BUDGET from env'),
]

if __name__ == "__main__":
    sys.exit(run("test_halt_levels.py", MUTATIONS))
