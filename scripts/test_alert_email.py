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


print("\nNO PHONE PUSH FROM A TEST RUN (the dev box's .env has a real key)")
import types as _types  # noqa: E402
_PUSHED: list[str] = []
_fake_pb = _types.ModuleType("pushbullet")
_fake_pb.Pushbullet = lambda key: _types.SimpleNamespace(push_note=lambda t, m: _PUSHED.append(t))
sys.modules["pushbullet"] = _fake_pb
import __main__ as _main  # noqa: E402
_argv0 = _main.__file__
_main.__file__ = "scripts/test_alert_email.py"
check("under a test script, push_alert with a key sends NOTHING",
      rg.push_alert("t", "m", api_key="k") is False and _PUSHED == [], str(_PUSHED))
_main.__file__ = "scripts/mutate_alert_email.py"
check("...nor under a mutation runner", rg.push_alert("t", "m", api_key="k") is False and _PUSHED == [],
      str(_PUSHED))
_main.__file__ = "scripts/run_live_entry.py"
check("a live entry point with a key DOES push (the guard is name-based, not a blanket off)",
      rg.push_alert("t", "m", api_key="k") is True and _PUSHED == ["t"], str(_PUSHED))
_main.__file__ = _argv0

print("\nNO REAL ALERT EMAIL FROM A TEST RUN")
_MAILED: list[str] = []


class _RealLooking:                       # stands in for smtplib.SMTP_SSL: same module name
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def login(self, u, p):
        pass

    def sendmail(self, frm, to, raw):
        _MAILED.append(raw)


_RealLooking.__module__ = "smtplib"
_saved_smtp = smtplib.SMTP_SSL
smtplib.SMTP_SSL = _RealLooking
os.environ.update(EMAIL_USER="u@x", EMAIL_PASS="p", TO_EMAIL="t@x")
_c = type("C", (), {"records": [("ERROR", "HALTED (hard): x")], "worst": "ERROR"})()
_main.__file__ = "scripts/test_alert_email.py"
check("under a test script, with the REAL smtplib class, email_if_alerts sends nothing",
      rg.email_if_alerts(_c, "T") is False and _MAILED == [], str(len(_MAILED)))
_main.__file__ = "scripts/run_live_entry.py"
sys.argv = ["run_options_paper.py", "--live"]
_main.__file__ = _argv0
check("a test that rewrites sys.argv to drive main() is STILL treated as a test (no real mail)",
      rg.email_if_alerts(_c, "T") is False and _MAILED == [], str(len(_MAILED)))
_main.__file__ = "scripts/run_live_entry.py"
check("a live entry point sends it (the guard is name-based)", rg.email_if_alerts(_c, "T") is True
      and len(_MAILED) == 1, str(len(_MAILED)))
_main.__file__ = _argv0
smtplib.SMTP_SSL = _saved_smtp

print("\n" + "=" * 88)
if _fails:
    print(f"{len(_fails)} FAILURE(S) of {_ran}:")
    for x in _fails:
        print("   " + x)
    sys.exit(1)
print(f"all {_ran} alert-email checks behaved as expected")
