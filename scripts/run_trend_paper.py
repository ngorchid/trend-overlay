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
from risk_guard import (RiskLimits, effective_budget,  # noqa: E402
                        install_alert_collector, missed_runs, push_if_alerts,
                        reconcile, halt_state, HALT_ALL, HALT_NEW,
                        circuit_breaker, peak_equity, margin_check, MarginLimits,
                        data_fresh, write_equity, book_drawdown, BookLevels,
                        BreakerLevels, realised_vol)
from trend_overlay.state import TrendState  # noqa: E402

load_dotenv(ROOT / ".env")
logging.basicConfig(level=logging.INFO, format="%(message)s")
# WARNING+ collected so the daily email can carry it — the scheduled task's stdout goes to a log
# file nobody reads, so an unsurfaced guard rejection is indistinguishable from a clean run.
ALERTS = install_alert_collector()
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

    # KILL SWITCH. HALT_ALL exits before connecting; HALT freezes exposure but still ROLLS —
    # a physically-delivered contract (ZB, ZN, SIL) left past its notice date goes to DELIVERY,
    # so a halt that blocks rolls is more dangerous than the situation prompting it.
    _halt, _hwhy = halt_state(ROOT)
    if _halt == HALT_ALL:
        logging.error("HALTED (all): %s — exiting without trading. NOTE: delivery/roll safety "
                      "closes did NOT run.", _hwhy)
        push_if_alerts(ALERTS, "Trend Overlay")
        return

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
        # MARGIN CEILING — needs a live connection, so it sits here rather than at config time.
        # This is the strategy that actually CONSUMES the shared margin: the overlay runs
        # $311,900 of notional against ~$10,900 of margin, collateralised by the equity book.
        # The reading is ACCOUNT-WIDE, so trend can be blocked by another strategy's usage —
        # correct, since the constraint really is shared.
        _mu = broker.margin_usage()
        _mlvl, _mscale, _mwhy = margin_check(*(_mu if _mu else (float("nan"), 0.0)),
                                             limits=MarginLimits())
        if _mwhy:
            (logging.error if _mlvl in ("derisk", "halt") else logging.warning)(
                "margin: %s", _mwhy)
        if _mscale <= 0 and _mlvl != "unknown":
            _halt = HALT_NEW            # freeze exposure; rolls + safety closes still run
        elif 0 < _mscale < 1.0:
            cfg.overlay_multiple *= _mscale

        # CIRCUIT BREAKER — here, not at config time, so it can see UNREALISED P&L. Realised-only
        # is blind to exactly the drawdowns that matter: a futures book held for months can be
        # deep underwater with nothing booked. An extra portfolio read is cheap and idempotent;
        # the later read at reporting time reflects POST-trade state and must stay separate.
        _pre = broker.portfolio_marks(FUTURES)
        _unreal = sum(p.get("unrealized_pnl") or 0.0 for p in _pre)
        _base = float(os.getenv("BUDGET", "100000"))
        _eq = _base + state.realized_pnl + _unreal
        _peak = max(peak_equity(state.nav_history, _base, key="total_pnl"), _eq)
        write_equity(ROOT.parent, "trend-overlay", _eq, _peak)
        # Vol-scaled: 15/25/35 IS 1.2/2.0/2.8 sigma at trend's ~12.4% vol, so this leaves the
        # levels ~unchanged here while fixing the higher-vol equity book.
        _lv = BreakerLevels.from_vol(realised_vol(state.nav_history, _base, key="total_pnl"))
        _blvl, _bscale, _bwhy = circuit_breaker(_eq, _peak, _lv)
        if _bwhy:
            (logging.error if _blvl == "halt" else logging.warning)("circuit breaker: %s", _bwhy)
        # BOOK-level: three books each down 20% all sit under their own 25% threshold while the
        # total is down 20%. Takes the WORSE of own and book.
        _bdd, _beq, _bpk, _bnote = book_drawdown(ROOT.parent)
        if _bdd is not None:
            _lvl2, _sc2, _why2 = circuit_breaker(_beq, _bpk, BookLevels())
            if _why2:
                logging.warning("BOOK circuit breaker: %s | %s", _why2, _bnote)
            _bscale = min(_bscale, _sc2)
        if _bscale <= 0:
            _halt = HALT_NEW            # freeze exposure; rolls + safety closes continue
        elif _bscale < 1.0:
            cfg.overlay_multiple *= _bscale

        if is_trade_day:
            front = broker.front_expiries(FUTURES, cfg.use_micro)
            held = broker.held_positions(FUTURES)
            safety = safety_closes(held, BY_MARKET, today)
            done = {(o.ib_symbol, o.expiry) for o in safety}
            held_left = [h for h in held if (h.ib_symbol, h.expiry) not in done]
            # Target leg needs FRESH prices; SAFETY closes never do and must always run,
            # or a physically-delivered contract drifts toward delivery.
            _fresh = None
            if not args.safety_only:
                start_dt = (pd.Timestamp.today() - pd.Timedelta(days=500)).strftime("%Y-%m-%d")
                px = download_ohlcv(PROXY_ETFS, start_dt)["adj_close"]
                # A frozen feed still yields a signal, a vol estimate and a full target book —
                # all plausible, all wrong. price sanity cannot catch it: each price is valid,
                # just old.
                _fresh = data_fresh(px.index, pd.Timestamp(today),
                                    RiskLimits.for_futures(cfg.budget))
                if not _fresh:
                    logging.error("data staleness: %s — SAFETY closes only today", _fresh.reason)

            if args.safety_only or not _fresh:
                batches = [("SAFETY", safety)]
            else:
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
                if _halt == HALT_NEW:
                    # Freeze exposure at what is already held: plan_roll_orders will still roll
                    # out of near-expiry contracts into the front month at the SAME size, so
                    # nothing drifts toward delivery, but no position is opened, closed or
                    # resized. SAFETY closes are untouched and run regardless.
                    logging.warning("HALTED (new risk): holding current exposure, rolls only")
                    targets = dict(held_by_mkt)
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
        # RECONCILE the state LEDGER against IB. Positions themselves are re-read from IB every
        # run, so they cannot drift — but the ledger drives REALISED-P&L accounting, and if it
        # disagrees with the broker the P&L is wrong in a way nothing else surfaces. Report only.
        _exp = {m: float(l.qty) for m, l in state.ledger.items() if l.qty}
        _act = {p["market"]: float(p["contracts"]) for p in positions}
        _d, _rnote = reconcile(_exp, _act, label="futures ledger")
        if _rnote:
            logging.warning("%s", _rnote)
        unreal = sum(p.get("unrealized_pnl") or 0.0 for p in positions)
        spy_day, spy_incep, _ = _spy_returns(state.inception_date)
        state.record_snapshot(today, state.realized_pnl + unreal)
        state.save(STATE_FILE)
        _m, _l, _note = missed_runs(state.nav_history, today)
        if _note:
            logging.warning("heartbeat: %s", _note)
        send_report(state, positions, todays_orders, spy_day, spy_incep, today, dry_run=False,
                    alerts=ALERTS)
        push_if_alerts(ALERTS, "Trend Overlay")
    finally:
        broker.disconnect()
    logging.info("Done %s: %d trades, %d open positions.", today, len(todays_orders), len(positions))


if __name__ == "__main__":
    main()
