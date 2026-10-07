"""Mutation-test `scripts/test_code_version.py`: the commit hash in the startup log AND the emails.

Seeds faults into a TEMP COPY of the repo (the real files are never edited) and demands the suite
catches every one. Engine: _mutate_repo_core.py.

Run: python scripts/mutate_code_version.py
"""
from __future__ import annotations

import sys

from _mutate_repo_core import run

REPORT = "trend_overlay/email_report.py"
RUNNER = "scripts/run_trend_paper.py"

MUTATIONS = [
    ("risk_guard.py", "    _CODE_VERSION_NOTE = note\n", "",
     "code_version does not remember the note"),
    ("risk_guard.py",
     '        _CODE_VERSION_NOTE = "code version unknown (not a git checkout)"\n        return _CODE_VERSION_NOTE, None',
     '        return "code version unknown (not a git checkout)", None',
     "a non-checkout leaves the previous hash in place (stale)"),
    ("risk_guard.py",
     "        f\"<p style='color:#64748b;font-size:11px'>Code: {_CODE_VERSION_NOTE or 'version not recorded'}</p>\"",
     '        ""',
     "the alert email carries no code version"),
    (REPORT, "        note = code_version_note()\n", "        note = \"\"\n",
     "the report never reads the code version"),
    (REPORT, "    {_code_line()}\n", "\n", "the report has no code-version line"),
    (RUNNER, "    code_version(ROOT)\n    warn_non_usd_contracts()\n    _halt, _hwhy = halt_state(ROOT)\n", "    warn_non_usd_contracts()\n    _halt, _hwhy = halt_state(ROOT)\n    code_version(ROOT)\n", "code_version runs AFTER the kill switch (a halted run logs no hash)"),
]

if __name__ == "__main__":
    sys.exit(run("test_code_version.py", MUTATIONS))
