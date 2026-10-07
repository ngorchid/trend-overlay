"""Shared engine for mutation runners that need the whole repo (temp-copy style).

The repo's code (no .git, data, results, .env) is copied to a temp dir once; each mutant is written
there and the suite runs there, so an interrupted run can never leave a broken file for a scheduled
task. A mutant that does not compile is rejected (it would be "caught" by the SyntaxError alone).
Non-zero exit if any fault survives, a pattern does not match exactly once, or a mutant is invalid.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_BASE_IGNORE = shutil.ignore_patterns(".git", "results", "__pycache__", ".idea", ".claude",
                                      ".env", ".venv", "venv", "*.parquet", "*.pkl")


def IGNORE(dirpath, names):
    """Skip VCS, results and caches -- but a top-level `data` dir can be a Python PACKAGE
    (algo_trading's is, next to 14 GB of caches): keep its .py files, drop everything else."""
    out = set(_BASE_IGNORE(dirpath, names))
    if Path(dirpath).name == "data" and Path(dirpath).parent == ROOT:
        out |= {n for n in names if not n.endswith(".py")}
    return out


# NO BYTECODE CACHE in the suite processes. A .pyc is validated only by the source's size and
# mtime (1 s resolution), so a mutant of the SAME SIZE as the original (e.g. two names swapped),
# written within the same second, can run the stale cached code -- found 2026-10-07 when a caught
# mutation "survived" on its second application.
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")


def run(suite: str, mutations: list[tuple[str, str, str, str]]) -> int:
    cmd = [sys.executable, "-B", f"scripts/{suite}"]
    base = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=ENV)
    if base.returncode != 0:
        print(f"{suite} does not pass on the ORIGINAL code — fix that first")
        print(base.stdout[-2000:])
        return 1
    print("=" * 100)
    print(f"  {suite}: {len(mutations)} seeded faults; every one must be CAUGHT\n")
    results, crashes = [], []
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "repo"
        shutil.copytree(ROOT, work, ignore=IGNORE)
        # The suite must pass on the UNMUTATED copy, or every mutant would be "caught" by the copy
        # being incomplete (found 2026-10-07: algo_trading's `data` package was skipped, and all
        # 20 order-link faults were counted as caught on a ModuleNotFoundError).
        cb = subprocess.run(cmd, cwd=work, capture_output=True, text=True, env=ENV)
        if cb.returncode != 0:
            print(f"{suite} FAILS on the unmutated temp copy — the copy is incomplete; nothing "
                  f"was tested:\n{(cb.stderr or cb.stdout)[-1500:]}")
            return 1
        for rel, find, repl, why in mutations:
            f = work / rel
            orig = f.read_text(encoding="utf-8")
            n = orig.count(find)
            if n != 1:
                results.append((why, None))
                print(f"  [ ?? ] {why:80} PATTERN {'MISSING' if n == 0 else f'AMBIGUOUS x{n}'}")
                continue
            mutant = orig.replace(find, repl, 1)
            try:                    # only Python is compiled; config/docs mutants are text
                if rel.endswith(".py"):
                    compile(mutant, rel, "exec")
            except SyntaxError as e:
                results.append((why, None))
                print(f"  [ ?? ] {why:80} INVALID MUTANT ({e.msg})")
                continue
            f.write_text(mutant, encoding="utf-8")
            try:
                r = subprocess.run(cmd, cwd=work, capture_output=True, text=True, env=ENV)
            finally:
                f.write_text(orig, encoding="utf-8")
            caught = r.returncode != 0
            # A crash also exits non-zero, but it may be the TEST's own code failing (e.g. a bad
            # attribute in a failure message) rather than a check detecting the fault. Label it so
            # each crash-catch is reviewed instead of counted silently.
            crashed = caught and "FAILURE(S)" not in r.stdout
            results.append((why, caught))
            if crashed:
                crashes.append(why)
            tail = (r.stderr.strip().splitlines() or ["?"])[-1][:60] if crashed else ""
            print(f"  [{'ok  ' if caught else 'FAIL'}] {why:80} "
                  f"{('CAUGHT (crash: ' + tail + ')') if crashed else 'CAUGHT' if caught else '*** SURVIVED ***'}")
    survived = [w for w, c in results if c is False]
    missing = [w for w, c in results if c is None]
    print("\n" + "=" * 100)
    if missing:
        print(f"{len(missing)} mutation(s) not applied (pattern moved or mutant invalid) — update:")
        for w in missing:
            print("   " + w)
    if survived:
        print(f"{len(survived)} MUTATION(S) SURVIVED — those cases cannot fail and are decoration:")
        for w in survived:
            print("   " + w)
    if crashes:
        print(f"{len(crashes)} fault(s) caught by a CRASH, not a check — review each:")
        for w in crashes:
            print("   " + w)
    if survived or missing:
        return 1
    print(f"all {len(mutations)} seeded faults were caught; the real files were never modified")
    return 0
