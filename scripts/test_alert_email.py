"""Every trend-overlay path WITHOUT a daily report emails its alerts (2026-10-07): HALT_HARD and a
failed IB connection. Pushbullet is not configured on the live machine. Real alert collector, fake
SMTP. scripts/mutate_alert_email.py seeds the faults.

Run: python scripts/test_alert_email.py
"""
from __future__ import annotations

import os
import smtplib
import sys
from email import message_from_string
from email.header import decode_header, make_header
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

_fails: list[str] = []
_ran = 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global _ran
    _ran += 1
    if not cond:
        _fails.append(f"{label}  | {detail}")
    print(f"  [{'ok ' if cond else 'FAIL'}] {label}" + ("" if cond else f"   <- {detail}"))


def safe(fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception as e:  # noqa: BLE001 -- a crash is a FAILED check here, not a test crash
        return f"crashed: {type(e).__name__}: {e}"


SENT: list[tuple[str, str]] = []


class FakeSMTP:
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def login(self, u, p):
        pass

    def sendmail(self, frm, to, raw):
        m = message_from_string(raw)
        b = m.get_payload(decode=True)
        SENT.append((str(make_header(decode_header(m["Subject"]))), b.decode("utf-8", "replace") if b else ""))


smtplib.SMTP_SSL = FakeSMTP
os.environ.update(EMAIL_USER="u@x", EMAIL_PASS="p", TO_EMAIL="t@x", BOOK_LABEL="LIVE")
os.environ.pop("PUSHBULLET_API_KEY", None)

import tempfile  # noqa: E402

import risk_guard as rg  # noqa: E402
import run_trend_paper as runner  # noqa: E402

runner.STATE_FILE = Path(tempfile.mkdtemp()) / "state.json"
sys.argv = ["run_trend_paper.py", "--live", "--force"]

print("HALT_HARD")
runner.halt_state = lambda root: (rg.HALT_HARD, "test")
runner.make_broker = lambda **kw: (_ for _ in ()).throw(AssertionError("no broker under HALT_HARD"))
r = safe(runner.main)
check("HALT_HARD -> one alert email, subject names the level and the halt",
      r is None and len(SENT) == 1 and "Trend Overlay HALT_HARD LIVE" in SENT[0][0]
      and "HALTED (hard)" in SENT[0][0], str((r, [s for s, _ in SENT])))

print("\nIB CONNECT FAILURE")
SENT.clear()
runner.ALERTS.records.clear()
runner.halt_state = lambda root: (rg.HALT_NONE, "")


class NoConnect:
    def connect(self):
        return False


runner.make_broker = lambda **kw: NoConnect()
r = safe(runner.main)
check("a failed connection -> alert email (no report would otherwise go out)",
      r is None and len(SENT) == 1 and "IB connect failed" in SENT[0][0], str((r, [s for s, _ in SENT])))

print("\n" + "=" * 88)
if _fails:
    print(f"{len(_fails)} FAILURE(S) of {_ran}:")
    for x in _fails:
        print("   " + x)
    sys.exit(1)
print(f"all {_ran} alert-email checks behaved as expected")
