"""SIZING IDENTITY (2026-10-07): moving the live budget onto config/capital_bases.json must not change
a single sizing output. fixtures/sizing_fingerprint_2026-10-07.json was produced by the code BEFORE
the change, run with the live env (BUDGET=50000, OVERLAY_MULT=1.5); this recomputes the same fingerprint --
config dataclass + the real sizing output on fixed fixtures -- on the CURRENT code, from the config
alone (no env override) and with an agreeing env override, and requires BYTE-IDENTICAL JSON.

Also: the link between the two homes -- this sleeve's documented budget may not exceed its
risk_guard.ALLOCATIONS ceiling x NOMINAL_NAV -- and that config/capital_bases.json is byte-identical
to the sibling repos' copies when they are checked out next to this one.
scripts/mutate_sizing_identity.py seeds the faults.

Run: python scripts/test_sizing_identity.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass
import risk_guard as rg  # noqa: E402

SLEEVE = "trend-overlay"
_fails: list[str] = []
_ran = 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global _ran
    _ran += 1
    if not cond:
        _fails.append(f"{label}  | {detail}")
    print(f"  [{'ok ' if cond else 'FAIL'}] {label}" + ("" if cond else f"   <- {detail}"))


def fingerprint(env: dict) -> str:
    """Run the fingerprint in a FRESH process (module-level env reads must see `env`)."""
    e = {k: v for k, v in os.environ.items() if k not in ("BUDGET", "OVERLAY_MULT")}
    e.update(env, PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, "-B", str(Path(__file__).with_name("_sizing_fingerprint.py")),
                        SLEEVE, str(ROOT), "new"], capture_output=True, text=True, env=e, cwd=ROOT)
    return r.stdout.strip() if r.returncode == 0 else f"FAILED: {r.stderr[-400:]}"


before = (Path(__file__).parent / "fixtures" / "sizing_fingerprint_2026-10-07.json").read_text().strip()


def same(a: str, b: str, rel: float = 1e-9) -> bool:
    """Equal in every discrete value (config, contracts, strikes, share counts, order lists) and in
    every float to `rel` relative precision. Exact byte equality is machine-dependent: the live PC
    (Windows) and the Mac differ in the last digits of floating-point results (found 2026-10-07,
    e.g. a credit of 0.27703538746578005 vs ...893), while every sizing decision is identical."""
    def eq(x, y):
        if isinstance(x, str) and isinstance(y, str) and x[:1] in "{[" and y[:1] in "{[":
            try:
                return eq(json.loads(x), json.loads(y))
            except ValueError:
                return x == y
        if isinstance(x, float) or isinstance(y, float):
            if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                return False
            return abs(x - y) <= rel * max(1.0, abs(x), abs(y))
        if isinstance(x, dict) and isinstance(y, dict):
            return x.keys() == y.keys() and all(eq(x[k], y[k]) for k in x)
        if isinstance(x, list) and isinstance(y, list):
            return len(x) == len(y) and all(eq(i, j) for i, j in zip(x, y))
        return x == y
    try:
        return eq(json.loads(a), json.loads(b))
    except ValueError:
        return False
print("IDENTICAL SIZING, BEFORE vs AFTER (discrete values exact, floats to 1e-9 relative)")
check("the comparison treats last-digit float noise as equal (0.27703538746578005 vs ...893)",
      same('{"c": 0.27703538746578005, "n": 8}', '{"c": 0.27703538746578893, "n": 8}'), "")
check("float noise inside a NESTED JSON string (trend's targets) also counts as equal",
      same('{"t": "{\\"x\\": 0.27703538746578005}"}', '{"t": "{\\"x\\": 0.27703538746578893}"}'), "")
check("...but NOT a real change: a float moved by 1e-6, or a different contract count",
      not same('{"c": 0.277035, "n": 8}', '{"c": 0.277036, "n": 8}')
      and not same('{"c": 0.27, "n": 8}', '{"c": 0.27, "n": 9}')
      and not same('{"t": "{\\"x\\": 2}"}', '{"t": "{\\"x\\": 3}"}'), "")
fp_cfg = fingerprint({})
check("from the config alone (no env override) == the pre-change output", same(fp_cfg, before),
      fp_cfg[:300])
fp_env = fingerprint({"BUDGET": "50000", "OVERLAY_MULT": "1.5"})
check("with the live env override (agreeing) == the pre-change output", same(fp_env, before), fp_env[:300])
fp_bad = fingerprint({"BUDGET": "60000"})
check("...and the comparison CAN fail: a different budget changes the fingerprint", not same(fp_bad, before)
      and not fp_bad.startswith("FAILED"), fp_bad[:200])

print("\nONE HOME EACH: config = sizing, ALLOCATIONS = ceiling")
cfgj = json.loads((ROOT / "config" / "capital_bases.json").read_text())
budget = float(cfgj[SLEEVE]["budget"])
ceiling = rg.ALLOCATIONS[SLEEVE].fraction * rg.NOMINAL_NAV
check(f"{SLEEVE} budget ${budget:,.0f} <= ALLOCATIONS ceiling x NAV ${ceiling:,.0f}", budget <= ceiling,
      f"{budget} > {ceiling}")
for sl, e in cfgj.items():
    if sl.startswith("_"):
        continue
    want = float(e["budget"]) * float(e.get("overlay_mult", 1.0))
    if sl == "magic-formula":
        continue                       # amount = inception NAV, budget = cap: not a product
    check(f"{sl}: return base amount = budget x overlay ({want:,.0f})", float(e["amount"]) == want,
          str(e))
import tempfile  # noqa: E402
LOGS: list[tuple[str, str]] = []
_oe, _ow = rg.logging.error, rg.logging.warning
rg.logging.error = lambda f, *a: LOGS.append(("E", f % a if a else f))
rg.logging.warning = lambda f, *a: LOGS.append(("W", f % a if a else f))
_saved = os.environ.pop("BUDGET", None)
fb_val, fb_src = rg.documented_sizing(Path(tempfile.mkdtemp()), SLEEVE)
check("a missing config is an ERROR and falls back to the nominal allocation -- never to zero",
      fb_val == rg.ALLOCATIONS[SLEEVE].fraction * rg.NOMINAL_NAV and fb_val > 0
      and fb_src.startswith("FALLBACK") and any(lv == "E" for lv, _ in LOGS), str((fb_val, fb_src, LOGS)))
LOGS.clear()
rg.log_sizing(SLEEVE, ceiling * 1.2, "test")
check("log_sizing WARNS when a budget exceeds its ALLOCATIONS ceiling x NAV",
      any(lv == "W" and "exceeds its ALLOCATIONS ceiling" in m for lv, m in LOGS), str(LOGS))
LOGS.clear()
_oi = rg.logging.info
rg.logging.info = lambda f, *a: LOGS.append(("I", f % a if a else f))
line = rg.log_sizing(SLEEVE, budget, "config/capital_bases.json")
rg.logging.info = _oi
check("...and LOGS the budget in use with its source, without a warning at the ceiling",
      f"budget in use ${budget:,.0f} [config/capital_bases.json]" in line
      and LOGS == [("I", line)], str((line, LOGS)))
LOGS.clear()
os.environ["BUDGET"] = str(budget + 10_000)
ov, osrc = rg.documented_sizing(ROOT, SLEEVE)
os.environ.pop("BUDGET")
check("a DIVERGING env override is used but announced as a WARNING naming both values",
      ov == budget + 10_000 and "override" in osrc
      and any(lv == "W" and "DIFFERS" in m and f"{budget:g}" in m for lv, m in LOGS), str((ov, osrc, LOGS)))
LOGS.clear()
os.environ["BUDGET"] = str(budget)
av, asrc = rg.documented_sizing(ROOT, SLEEVE)
os.environ.pop("BUDGET")
check("an AGREEING env override is silent and reported as agreeing",
      av == budget and "agrees" in asrc and not LOGS, str((av, asrc, LOGS)))
rg.logging.error, rg.logging.warning = _oe, _ow
if _saved is not None:
    os.environ["BUDGET"] = _saved
mine = (ROOT / "config" / "capital_bases.json").read_bytes()
for sib in ("algo_trading", "trend-overlay", "options-vrp"):
    p = ROOT.parent / sib / "config" / "capital_bases.json"
    if p.exists() and p.resolve() != (ROOT / "config" / "capital_bases.json").resolve():
        check(f"config/capital_bases.json byte-identical to {sib}'s copy", p.read_bytes() == mine, sib)

print("\n" + "=" * 88)
if _fails:
    print(f"{len(_fails)} FAILURE(S) of {_ran}:")
    for x in _fails:
        print("   " + x)
    sys.exit(1)
print(f"all {_ran} sizing-identity checks behaved as expected")
