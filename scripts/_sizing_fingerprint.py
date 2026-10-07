"""Print a JSON fingerprint of one sleeve's sizing (config + real sizing output on fixtures).
Usage: python fingerprint.py <sleeve> <repo_root> <mode: old|new>"""
import json, os, sys
from dataclasses import asdict
from math import erf, log, sqrt
from pathlib import Path
import numpy as np, pandas as pd
sleeve, root, mode = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
sys.path[:0] = [str(root), str(root / "scripts")]
os.chdir(root)
out = {}

def mkchain(spot=100.0, lo=60.0, step=1.0, oi=5000, iv=0.25, dte=35, r=0.04):
    ks = np.arange(lo, spot, step); T = dte / 365.0
    n = lambda x: 0.5 * (1.0 + erf(x / sqrt(2.0)))
    mid = []
    for k in ks:
        d1 = (log(spot / k) + (r + 0.5 * iv * iv) * T) / (iv * sqrt(T)); d2 = d1 - iv * sqrt(T)
        mid.append(k * np.exp(-r * T) * n(-d2) - spot * n(-d1))
    mid = np.array(mid)
    return pd.DataFrame({"strike": ks, "bid": mid * 0.97, "ask": mid * 1.03, "impliedVolatility": iv,
                         "openInterest": oi})

if sleeve == "options-vrp":
    import run_options_paper as r
    from options_vrp.strategy import build_spread
    cfg = r._cfg(None)
    out["cfg"] = {k: v for k, v in asdict(cfg).items()}
    out["spreads"] = []
    for spot, step, iv in ((100.0, 1.0, 0.25), (50.0, 0.5, 0.35), (400.0, 5.0, 0.18), (263.0, 1.0, 0.22)):
        sp = build_spread("T", mkchain(spot, spot * 0.6, step, iv=iv), spot, "2026-11-20", 35, iv, iv * 0.8, cfg)
        out["spreads"].append(asdict(sp) if sp else None)
elif sleeve == "trend-overlay":
    import run_trend_paper as r
    from trend_overlay.execution import compute_targets
    from trend_overlay.contracts import FUTURES
    cfg = r._cfg(None)
    out["cfg"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(cfg).items()}
    rng = np.random.default_rng(11)
    idx = pd.bdate_range(end="2026-10-06", periods=400)
    px = pd.DataFrame({s.proxy_etf: 100 * np.exp(np.cumsum(rng.standard_normal(len(idx)) * 0.012 + 0.0004))
                       for s in FUTURES}, index=idx)
    out["targets"] = compute_targets(px, cfg).round(10).to_json()
elif sleeve == "magic-formula":
    import run_paper as r
    from paper.orchestrator import PaperConfig, run_daily
    from paper.state import PortfolioState, Position
    cfg = (PaperConfig(budget=float(os.getenv("BUDGET", "100000"))) if mode == "old"
           else r.paper_config())
    out["cfg"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(cfg).items()}
    class B:
        dry_run = True
        def __init__(self): self.orders = []
        def order(self, t, a, n, wait=20.0):
            self.orders.append((a, t, n)); return {"ok": True, "status": "dryrun", "fill_price": None}
    rng = np.random.default_rng(7)
    idx = pd.bdate_range(end="2026-08-17", periods=400)
    names = [f"N{i}" for i in range(14)] + ["OLD"]
    px = pd.DataFrame({n: 100 * np.exp(np.cumsum(rng.standard_normal(len(idx)) * 0.01)) for n in names}, index=idx)
    rank = pd.Series(range(len(names)), index=names, dtype=float)
    for cash in (50_000.0, 80_000.0, 30_000.0):           # NAV below, above and at a loss vs budget
        st = PortfolioState(cash=cash, positions=[Position("OLD", 10, 100.0, "2026-07-01")])
        b = B()
        c2 = PaperConfig(**{**asdict(cfg), "max_new_buys_per_day": 5, "hold_n": 12, "top_n": 10})
        run_daily(st, rank, {"adj": px, "currency": {c: "USD" for c in names}}, {"USD": 1.0}, b, c2, "2026-08-17")
        out[f"orders_cash_{int(cash)}"] = b.orders
print(json.dumps(out, sort_keys=True, default=str))
