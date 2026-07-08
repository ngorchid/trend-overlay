"""Trend-overlay paper runner — weekly trade, DAILY email.

Modes:
  python scripts/run_trend_paper.py --selftest      # offline roll/safety logic check
  python scripts/run_trend_paper.py                 # offline dry-run: print target book (no IB)
  python scripts/run_trend_paper.py --live          # connect IB paper: read marks + email;
                                                     #   on the weekly trade day also trade
  python scripts/run_trend_paper.py --live --trade   # force the trade leg today
  python scripts/run_trend_paper.py --live --safety-only  # only run delivery safety-closes

Design: connects every weekday to mark positions and send the daily P&L email; the trade
leg (compute targets → safety → roll → reconcile) runs only on the weekly day (default Fri),
so a daily cron gives a daily email and a weekly rebalance.
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
from trend_overlay.state import TrendState  # noqa: E402

load_dotenv(ROOT / ".env")
logging.basicConfig(level=logging.INFO, format="%(message)s")
STATE_FILE = ROOT / "results" / "paper" / "state.json"
TRADE_WEEKDAY = 4  # Friday


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


def _cfg() -> TrendPaperConfig:
    return TrendPaperConfig(
        budget=float(os.getenv("BUDGET", "100000")),
        target_vol=float(os.getenv("TARGET_VOL", "0.10")),
        overlay_multiple=float(os.getenv("OVERLAY_MULT", "0.5")))


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
    cfg = _cfg()

    if args.selftest:
        _selftest(); return
    if not args.live:
        _dry_book(cfg); return

    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    if now.weekday() >= 5 and not args.force:
        logging.info("Weekend (%s) — skipping.", today); return
    is_trade_day = args.trade or args.safety_only or now.weekday() == TRADE_WEEKDAY

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
                tgt = compute_targets(px, cfg)
                targets = {m: int(r["contracts"]) for m, r in tgt.iterrows()}
                rolls = plan_roll_orders(targets, held_left, front, BY_MARKET, cfg.use_micro, today)
                batches = [("SAFETY", safety), ("ROLL+RECONCILE", rolls)]
            for label, batch in batches:
                fills = broker.execute(batch, BY_MARKET, cfg.use_micro)
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
