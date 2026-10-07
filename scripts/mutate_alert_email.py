"""Mutation-test `scripts/test_alert_email.py`: alert emails on every path without a daily report (trend).

Seeds real faults into a TEMP COPY of the repo (the real files are never edited) and demands the
suite catches every one. Engine: _mutate_repo_core.py.

Run: python scripts/mutate_alert_email.py
"""
from __future__ import annotations

import sys

from _mutate_repo_core import run

MUTATIONS = [
    ("risk_guard.py", '    if _test_run() and getattr(_smtplib.SMTP_SSL, "__module__", "") == "smtplib":',
     "    if False:", "test runs mail real HALTED alerts again"),
    ("risk_guard.py", '    if _test_run() and getattr(_smtplib.SMTP_SSL, "__module__", "") == "smtplib":',
     "    if _test_run():", "test runs can no longer exercise the email send with a fake server"),
    ("risk_guard.py", "    if _test_run():\n", "    if False:\n",
     "test runs push real Pushbullet notes again"),
    ("risk_guard.py", '    return name.startswith(("test_", "mutate_"))', '    return name.startswith("test_")',
     "mutation runs push real Pushbullet notes"),
    ('risk_guard.py',
     '    if not recs:\n        return False\n    pick',
     '    if True:\n        return False\n    pick',
     'email_if_alerts never sends'),
    ('scripts/run_trend_paper.py',
     '        email_if_alerts(ALERTS, "Trend Overlay HALT_HARD", datetime.now().strftime("%Y-%m-%d"))\n',
     '',
     'HALT_HARD sends no email'),
    ('scripts/run_trend_paper.py',
     '        email_if_alerts(ALERTS, "Trend Overlay", today)\n        return',
     '        return',
     'a failed connection sends no email'),
]

if __name__ == "__main__":
    sys.exit(run("test_alert_email.py", MUTATIONS))
