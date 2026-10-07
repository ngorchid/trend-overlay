"""Tests for the kill-switch levels (2026-10-07) and the documented sizing config, trend-overlay.

  HALT_HARD : exits before connecting -- nothing runs.
  HALT_ALL  : ONLY the SAFETY closes and same-size rolls; nothing opened, resized or closed for
              the signal, and no price feed is touched (the run is deterministic).
Drives the REAL main() with a fake broker. scripts/mutate_halt_levels.py seeds the faults.

Run: python scripts/test_halt_levels.py
"""
from __future__ import annotations

import inspect
import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

import risk_guard as rg  # noqa: E402
import run_trend_paper as runner  # noqa: E402
from trend_overlay.contracts import BY_MARKET  # noqa: E402
from trend_overlay.execution import HeldPosition, safety_closes  # noqa: E402

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


# =============================================================================================
print("LEVELS IN risk_guard (shared)")
tmp = Path(tempfile.mkdtemp())
for k in ("TRADING_HALT",):
    os.environ.pop(k, None)
check("no file -> none", rg.halt_state(tmp)[0] == rg.HALT_NONE, "")
(tmp / "HALT").write_text("x")
(tmp / "HALT_ALL").write_text("y")
check("HALT_ALL beats HALT", rg.halt_state(tmp)[0] == rg.HALT_ALL, "")
(tmp / "HALT_HARD").write_text("z")
check("HALT_HARD beats both (the deliberate 'touch nothing')", rg.halt_state(tmp)[0] == rg.HALT_HARD, "")
os.environ["TRADING_HALT"] = "hard"
check("TRADING_HALT=hard -> hard", rg.halt_state(Path(tempfile.mkdtemp()))[0] == rg.HALT_HARD, "")
os.environ.pop("TRADING_HALT")

# =============================================================================================
print("\nHALT_ALL PLAN — safety closes + SAME-SIZE rolls only")
today = datetime.now().strftime("%Y-%m-%d")
near = (datetime.now() + timedelta(days=10)).strftime("%Y%m%d")       # inside oil/fx buffers
front_x = (datetime.now() + timedelta(days=45)).strftime("%Y%m%d")
held = [HeldPosition("oil", "MCL", near, 2.0),            # long, about to be force-closed
        HeldPosition("fx_eur", "M6E", near, -1.0),        # short, about to be force-closed
        HeldPosition("fx_aud", "M6A", front_x, 3.0)]      # already in the front: untouched
front = {"oil": front_x, "fx_eur": front_x, "fx_aud": front_x, "gold": front_x}
safety = safety_closes(held, BY_MARKET, today)
done = {(o.ib_symbol, o.expiry) for o in safety}
held_left = [h for h in held if (h.ib_symbol, h.expiry) not in done]
batches = safe(runner.halt_all_batches, held, held_left, safety, front, True, today)
ok = isinstance(batches, list)
check("batches = [SAFETY, ROLL] only", ok and [b[0] for b in batches] == ["SAFETY", "ROLL (HALT_ALL)"],
      str(batches))
saf = batches[0][1] if ok else []
rol = batches[1][1] if ok and len(batches) > 1 else []
check("SAFETY closes the two near-expiry positions (SELL 2 MCL, BUY 1 M6E), flagged safety",
      sorted((o.ib_symbol, o.action, o.qty) for o in saf) == [("M6E", "BUY", 1), ("MCL", "SELL", 2)]
      and all(o.safety for o in saf), str(saf))
check("ROLL re-establishes EXACTLY the closed size in the front month (BUY 2 MCL, SELL 1 M6E)",
      sorted((o.ib_symbol, o.expiry, o.action, o.qty) for o in rol)
      == [("M6E", front_x, "SELL", 1), ("MCL", front_x, "BUY", 2)], str(rol))
check("nothing for a market already in the front (fx_aud) or not held (gold): no opening order",
      not any(o.ib_symbol in ("M6A", "MGC") for o in saf + rol), str(rol))
check("HALT_ALL needs no prices: the plan takes no price panel",
      "px" not in inspect.signature(runner.halt_all_batches).parameters, "")

# =============================================================================================
print("\nmain() UNDER EACH LEVEL (fake broker, no network)")


class FakeBroker:
    def __init__(self):
        self.batches, self.connected = [], False
        self.ib = NS(sleep=lambda s: None)

    def connect(self):
        self.connected = True
        return True

    def disconnect(self):
        pass

    def margin_cushion(self):
        return (25_000.0, 50_000.0)

    def portfolio_marks(self, futures):
        return []

    def front_expiries(self, futures, micro):
        return dict(front)

    def held_positions(self, futures):
        return list(held)

    def execute(self, batch, specs, micro, limits=None, held=None):
        self.batches.append(list(batch))
        return []


def boom(*a, **k):
    raise AssertionError("HALT_ALL must not download prices")


made: list[FakeBroker] = []
runner.make_broker = lambda **kw: made.append(FakeBroker()) or made[-1]
runner.download_ohlcv = boom
runner._spy_returns = lambda *a: (None, None, None)
runner.send_report = lambda *a, **k: None
runner.push_if_alerts = lambda *a, **k: None
runner.write_equity = lambda *a, **k: None
runner.book_drawdown = lambda *a, **k: (None, None, None, "")
runner.STATE_FILE = Path(tempfile.mkdtemp()) / "state.json"
LOG: list[str] = []
runner.logging.error = lambda f, *a: LOG.append(f % a if a else f)

runner.halt_state = lambda root: (rg.HALT_HARD, "test")
sys.argv = ["run_trend_paper.py", "--live", "--force"]
r = safe(runner.main)
check("HALT_HARD: returns before connecting (no broker made)", r is None and made == [], str((r, made)))
check("...and says the safety closes did NOT run", any("did NOT run" in x for x in LOG), str(LOG))

runner.halt_state = lambda root: (rg.HALT_ALL, "test")
r = safe(runner.main)
fb = made[-1] if made else None
sent = [o for b in (fb.batches if fb else []) for o in b]
check("HALT_ALL: connects and runs without downloading prices", r is None and fb is not None
      and fb.connected, str(r))
check("HALT_ALL: the ONLY orders are the safety closes and their same-size rolls",
      sorted((o.ib_symbol, o.action, o.qty) for o in sent)
      == [("M6E", "BUY", 1), ("M6E", "SELL", 1), ("MCL", "BUY", 2), ("MCL", "SELL", 2)], str(sent))

print("\nDOCUMENTED SIZING (single source: config/capital_bases.json)")
for k in ("BUDGET", "OVERLAY_MULT"):
    os.environ.pop(k, None)
WARN: list[str] = []
_ow = rg.logging.warning
rg.logging.warning = lambda f, *a: WARN.append(f % a if a else f)
cfg = runner._cfg(None)
check("budget and OVERLAY_MULT come from the config (50,000 x 1.5)",
      cfg.budget == 50_000 and cfg.overlay_multiple == 1.5 and WARN == [], str((cfg.budget, cfg.overlay_multiple, WARN)))
os.environ["OVERLAY_MULT"] = "1.0"
cfg2 = runner._cfg(None)
check("an env OVERLAY_MULT that differs is used but WARNED about", cfg2.overlay_multiple == 1.0
      and any("OVERLAY_MULT" in w and "DIFFERS" in w for w in WARN), str(WARN))
os.environ.pop("OVERLAY_MULT")
rg.logging.warning = _ow
check("the breaker measures equity against the documented budget (no second env read)",
      "_base = cfg.budget" in inspect.getsource(runner.main)
      and 'os.getenv("BUDGET"' not in inspect.getsource(runner.main), "")

print("\n" + "=" * 88)
if _fails:
    print(f"{len(_fails)} FAILURE(S) of {_ran}:")
    for x in _fails:
        print("   " + x)
    sys.exit(1)
print(f"all {_ran} halt-level checks behaved as expected")
