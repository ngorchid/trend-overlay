"""Trend-overlay paper runner — DAILY trade + DAILY email.

Modes:
  python scripts/run_trend_paper.py --selftest      # offline roll/safety logic check
  python scripts/run_trend_paper.py                 # offline dry-run: print target book (no IB)
  python scripts/run_trend_paper.py --live          # connect IB paper: trade + read marks + email
  python scripts/run_trend_paper.py --live --safety-only  # only run delivery safety-closes

Design: connects every weekday to compute targets → safety → roll → reconcile (DAILY rebalance,
"variant D") and send the daily P&L email. Daily rebalancing tracks the slow (6/12-month) signal
more tightly and reacts to reversals faster; because the signal is slow, turnover only rises
modestly (~30->45x/yr) while backtest Sharpe improves vs the old weekly cadence.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402
from trend_overlay.contracts import BY_MARKET, FUTURES, PROXY_ETFS  # noqa: E402
from trend_overlay.data import download_ohlcv  # noqa: E402
from trend_overlay.email_report import send_report  # noqa: E402
from trend_overlay.execution import (  # noqa: E402
    FuturesBroker, HeldPosition, TrendPaperConfig,
    compute_targets, plan_roll_orders, safety_closes,
)
from risk_guard import RiskLimits, effective_budget  # noqa: E402
from trend_overlay.state import TrendState  # noqa: E402

load_dotenv(ROOT / ".env")
logging.basicConfig(level=logging.INFO, format="%(message)s")
STATE_FILE = ROOT / "results" / "paper" / "state.json"


def _selftest() -> None:
    today = "2026-07-10"
    targets = {"equity_us": 4, "oil": 2, "gold": -1}
    held = [HeldPosition("equity_us", "MES", "20260619", 3),
            HeldPosition("oil", "MCL", "20260721", 2),
            HeldPosition("gold", "MGC", "20260828", -1)]
    front = {"equity_us": "20260918", "oil": "20260820", "gold": "20260828"}
    safety = safety_closes(held, BY_MARKET, today)
    done = {(o.ib_symbol, o.expiry) for o in safety}
    held_left = [h for h in held if (h.ib_symbol, h.expiry) not in done]
    rolls = plan_roll_orders(targets, held_left, front, BY_MARKET, True, today)
    print("SELF-TEST (today 2026-07-10) — safety then roll+reconcile, deduped")
    for o in safety: print(f"  [SAFETY] {o.action} {o.qty} {o.ib_symbol} {o.expiry}  <- {o.reason}")
    for o in rolls:  print(f"  [ROLL]   {o.action} {o.qty} {o.ib_symbol} {o.expiry}  <- {o.reason}")


def _spy_returns(inception):
    try:
        spy = yf.download("SPY", period="1y", auto_adjust=True, progress=False)["Close"].dropna()
        spy = spy.iloc[:, 0] if hasattr(spy, "columns") else spy
        day = float(spy.iloc[-1] / spy.iloc[-2] - 1)
        incep = float(spy.iloc[-1] / spy[spy.index >= inception].iloc[0] - 1) if inception and len(spy[spy.index >= inception]) else None
        return day, incep, float(spy.iloc[-1])
    except Exception as e:  # noqa: BLE001
        logging.warning("SPY fetch failed: %s", e)
        return None, None, None


def _cfg(state: TrendState | None = None, unrealized: float = 0.0) -> TrendPaperConfig:
    """Build the config, sizing off the strategy's OWN compounded equity.

    BUDGET is the BASE capital, not a fixed sizing number: the effective budget is
    base + this strategy's realised + unrealised P&L, so markets come online by themselves as
    the account grows and exposure shrinks after losses, with no manual edit and no restart.

    Deliberately NOT IB's NetLiquidation — three strategies share one account, so NetLiq would
    have each of them sizing as though it owned the whole thing.

    NB OVERLAY_MULT defaults to 1.0. It was 0.5, which silently ran the book at HALF its
    validated risk: the backtest (Sharpe 0.74, maxDD -20%) is at 1.0, so 0.5 realised ~5% vol
    against a 10% target, for roughly half the expected return.
    """
    base = float(os.getenv("BUDGET", "100000"))
    budget, note = base, ""
    if state is not None:
        budget, note = effective_budget(base, state.realized_pnl, unrealized,
                                        step=float(os.getenv("BUDGET_STEP", "0.10")))
        logging.info("budget: %s", note)
    return TrendPaperConfig(
        budget=budget,
        target_vol=float(os.getenv("TARGET_VOL", "0.10")),
        overlay_multiple=float(os.getenv("OVERLAY_MULT", "1.0")))


def _dry_book(cfg) -> None:
    print(f"[dry-run] proxy history for {len(PROXY_ETFS)} markets …")
    start = (pd.Timestamp.today() - pd.Timedelta(days=500)).strftime("%Y-%m-%d")
    px = download_ohlcv(PROXY_ETFS, start)["adj_close"]
    tgt = compute_targets(px, cfg)
    gross = float(tgt["notional_used"].abs().sum())
    print(f"\nTARGET BOOK  budget ${cfg.budget:,.0f} × {cfg.target_vol:.0%} vol × {cfg.overlay_multiple:g}")
    for m, r in tgt.iterrows():
        print(f"  {m:11s} {r['ib_symbol']:5s} signal{r['signal']:+.2f} vol{r['ann_vol']:6.1%} "
              f"-> {int(r['contracts']):+d}  (${r['notional_used']:+,.0f})")
    print(f"  gross ${gross:,.0f} = {gross/cfg.budget:.1f}x budget   (dry run — nothing sent)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--trade", action="store_true", help="force the trade leg today")
    ap.add_argument("--safety-only", action="store_true", help="only run delivery safety-closes")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--force", action="store_true", help="run on weekends too")
    ap.add_argument("--port", type=int, default=int(os.getenv("IB_PORT", "7497")))
    ap.add_argument("--client-id", type=int, default=int(os.getenv("IB_CLIENT_ID", "6")))
    args = ap.parse_args()
    # State is loaded here purely to size off compounded equity. Unrealised P&L needs live marks
    # and the config is built before connecting, so this is REALISED-only for now — which lags
    # gains and therefore UNDER-sizes after a run-up. Conservative, and the right way to be wrong.
    cfg = _cfg(TrendState.load(STATE_FILE))

    if args.selftest:
        _selftest(); return
    if not args.live:
        _dry_book(cfg); return

    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    if now.weekday() >= 5 and not args.force:
        logging.info("Weekend (%s) — skipping.", today); return
    # Variant D: rebalance every weekday (daily). --safety-only still restricts to the safety leg.
    is_trade_day = not args.safety_only

    broker = FuturesBroker(port=args.port, client_id=args.client_id, dry_run=False)
    if not broker.connect():
        logging.error("IB connect failed — aborting."); return
    state = TrendState.load(STATE_FILE); state.ensure_inception(today)
    todays_orders: list[dict] = []
    try:
        if is_trade_day:
            front = broker.front_expiries(FUTURES, cfg.use_micro)
            held = broker.held_positions(FUTURES)
            safety = safety_closes(held, BY_MARKET, today)
            done = {(o.ib_symbol, o.expiry) for o in safety}
            held_left = [h for h in held if (h.ib_symbol, h.expiry) not in done]
            if args.safety_only:
                batches = [("SAFETY", safety)]
            else:
                start = (pd.Timestamp.today() - pd.Timedelta(days=500)).strftime("%Y-%m-%d")
                px = download_ohlcv(PROXY_ETFS, start)["adj_close"]
                # Current holdings, so hysteresis can hold a position whose target sits inside
                # the band. Keyed by MARKET (held_left is per contract-month); summed because a
                # market can straddle two expiries mid-roll.
                held_by_mkt: dict[str, int] = {}
                for h in held_left:
                    m = next((s_.market for s_ in FUTURES
                              if s_.sym(cfg.use_micro) == h.ib_symbol), None)
                    if m:
                        held_by_mkt[m] = held_by_mkt.get(m, 0) + int(h.qty)
                tgt = compute_targets(px, cfg, held=held_by_mkt)
                targets = {m: int(r["contracts"]) for m, r in tgt.iterrows()}
                rolls = plan_roll_orders(targets, held_left, front, BY_MARKET, cfg.use_micro, today)
                batches = [("SAFETY", safety), ("ROLL+RECONCILE", rolls)]
            lim = RiskLimits.for_futures(cfg.budget)
            for label, batch in batches:
                fills = broker.execute(batch, BY_MARKET, cfg.use_micro, limits=lim)
                for f in fills:
                    signed = f["qty"] if f["action"] == "BUY" else -f["qty"]
                    if f["fill_price"]:
                        state.record_fill(f["market"], signed, f["fill_price"], f["mult"],
                                          today, f["symbol"], f["expiry"], f["reason"])
                        todays_orders.append(f)
                    else:
                        todays_orders.append({**f, "reason": f["reason"] + f" ({f['status']})"})

        positions = broker.portfolio_marks(FUTURES)
        unreal = sum(p.get("unrealized_pnl") or 0.0 for p in positions)
        spy_day, spy_incep, _ = _spy_returns(state.inception_date)
        state.record_snapshot(today, state.realized_pnl + unreal)
        state.save(STATE_FILE)
        send_report(state, positions, todays_orders, spy_day, spy_incep, today, dry_run=False)
    finally:
        broker.disconnect()
    logging.info("Done %s: %d trades, %d open positions.", today, len(todays_orders), len(positions))


if __name__ == "__main__":
    main()
