"""The commit a run used is logged at startup AND printed in its emails (2026-10-07).

risk_guard.code_version() runs first in main(), above the kill switch, so even a halted run
records it; since 2026-10-07 it also remembers the note, and the daily report and the alert email
print it ("Code: running <sha> ..."). Uses a throw-away git repository, no network.
scripts/mutate_code_version.py seeds the faults.

Run: python scripts/test_code_version.py
"""
from __future__ import annotations

import os
import smtplib
import subprocess
import sys
import tempfile
from email import message_from_string
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

import risk_guard as rg  # noqa: E402
import run_trend_paper as runner  # noqa: E402
from trend_overlay.email_report import build_email_body  # noqa: E402
from trend_overlay.state import TrendState  # noqa: E402


def report() -> str:
    return build_email_body(TrendState(inception_date="2026-09-01"), [], [], 0.0, 0.0, "2026-10-07")


runner.STATE_FILE = Path(tempfile.mkdtemp()) / "state.json"
sys.argv = ["run_trend_paper.py", "--live", "--force"]

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


SENT: list[str] = []


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
        part = m.get_payload()[0] if m.is_multipart() else m
        SENT.append(part.get_payload(decode=True).decode("utf-8", "replace"))


smtplib.SMTP_SSL = FakeSMTP
os.environ.update(EMAIL_USER="u@x", EMAIL_PASS="p", TO_EMAIL="t@x")
os.environ.pop("PUSHBULLET_API_KEY", None)

repo = Path(tempfile.mkdtemp())
for cmd in (["init", "-q"], ["-c", "user.email=t@x", "-c", "user.name=t", "commit", "-q",
                             "--allow-empty", "-m", "x"]):
    subprocess.run(["git", "-C", str(repo), *cmd], check=True, capture_output=True)
subprocess.run(["git", "-C", str(repo), "update-ref", "refs/remotes/origin/master", "HEAD"], check=True)
SHA = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], capture_output=True,
                     text=True).stdout.strip()

print("BEFORE code_version() RUNS")
rg._CODE_VERSION_NOTE = ""
body = safe(report)
check("the report says 'version not recorded' (never a blank or a stale hash)",
      "Code: version not recorded" in str(body), str(body)[-400:])

print("\nAFTER code_version()")
note, _ = rg.code_version(repo, fetch=False)
check("code_version returns and REMEMBERS the note: hash and '0 commit(s) behind origin/master'",
      SHA in note and "0 commit(s) behind origin/master" in note and rg.code_version_note() == note, f"{note!r} vs {rg.code_version_note()!r}")
body = safe(report)
check("the daily report prints it: 'Code: running <sha>'", f"Code: running {SHA}" in str(body),
      str(body)[-400:])
SENT.clear()
rg.email_if_alerts(type("C", (), {"records": [("ERROR", "x")], "worst": "ERROR"})(), "T")
check("the alert email prints it too", bool(SENT) and f"Code: running {SHA}" in SENT[-1],
      SENT[-1][-300:] if SENT else "nothing sent")
rg.code_version(Path(tempfile.mkdtemp()), fetch=False)
check("a non-checkout overwrites it with 'unknown' (no stale hash carried over)",
      rg.code_version_note() == "code version unknown (not a git checkout)", rg.code_version_note())

print("\nIN main(): logged first, before the kill switch, and in a halted run's email")
order: list[str] = []
runner.code_version = lambda root, **k: (order.append("code_version"), rg.code_version(repo, fetch=False))[1]
runner.halt_state = lambda root: (order.append("halt_state"), (rg.HALT_HARD, "test"))[1]
runner.push_if_alerts = lambda *a, **k: None
SENT.clear()
r = safe(runner.main)
check("main() calls code_version BEFORE halt_state", order[:2] == ["code_version", "halt_state"],
      str((order, r)))
check("...and the HALT_HARD alert email carries the hash", bool(SENT) and f"Code: running {SHA}" in SENT[-1],
      SENT[-1][-300:] if SENT else f"nothing sent ({r})")

print("\n" + "=" * 88)
if _fails:
    print(f"{len(_fails)} FAILURE(S) of {_ran}:")
    for x in _fails:
        print("   " + x)
    sys.exit(1)
print(f"all {_ran} code-version checks behaved as expected")
