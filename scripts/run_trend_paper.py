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

import numpy as np
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
from risk_guard import (RiskLimits, documented_sizing, log_sizing, HALT_HARD,  # noqa: E402
                        code_version,
                        stale_columns,
                        check_allocations,
                        install_alert_collector, missed_runs, push_if_alerts,
                        reconcile, halt_state, HALT_ALL, HALT_NEW,
                        circuit_breaker, peak_equity, liquidity_check, MarginLimits,
                        data_fresh, write_equity, book_drawdown, book_vol,
                        BreakerLevels, blended_vol)
from trend_overlay.state import TrendState  # noqa: E402

load_dotenv(ROOT / ".env")
logging.basicConfig(level=logging.INFO, format="%(message)s")
# WARNING+ collected so the daily email can carry it — the scheduled task's stdout goes to a log
# file nobody reads, so an unsurfaced guard rejection is indistinguishable from a clean run.
ALERTS = install_alert_collector()
STATE_FILE = ROOT / "results" / "paper" / "state.json"

# One id per run, stamped on every order as orderRef "trend-overlay:<RUN_ID>" — so each fill in
# IB's executions and Flex statements traces back to this sleeve AND to this run's log section.
RUN_ID = datetime.now().strftime("%Y%m%d-%H%M%S")
ORDER_REF = f"trend-overlay:{RUN_ID}"


def make_broker(**kw) -> FuturesBroker:
    """The run's broker, tagged with this run's ORDER_REF. A factory rather than two inline lines
    so the tagging is testable: scripts/test_order_links.py asserts main() builds its broker here
    and that the tag is set, because a runner that forgets it sends every order out untagged
    while the broker-level tests stay green."""
    b = FuturesBroker(**kw)
    b.order_ref = ORDER_REF
    return b


def warn_non_usd_contracts(futures=None) -> list[str]:
    """Runtime twin of scripts/test_contract_currency.py (2026-10-07): a test only runs when
    someone runs it, so every run also WARNS if a contract is not USD-denominated. It never stops
    anything -- closes and rolls must stay unblockable -- it only says the IB-records attribution
    premise (foreign cash belongs to magic-formula) no longer holds."""
    from trend_overlay.contracts import FUTURES
    bad = [f"{s.market} ({s.symbol}) {s.currency}"
           for s in (FUTURES if futures is None else futures) if s.currency != "USD"]
    if bad:
        logging.warning("NON-USD trend contract(s): %s — its cash would be charged to magic-formula "
                        "by the IB-records attribution; re-decide that rule", ", ".join(bad))
    return bad


def held_by_market(held, use_micro: bool) -> dict[str, int]:
    """Signed contracts held per MARKET (held is per contract-month; summed because a market can
    straddle two expiries mid-roll). Built from ALL holdings, safety-closed ones included: a
    safety close is a ROLL, not an exit (the 2026-10-05 oil fix)."""
    out: dict[str, int] = {}
    for h in held:
        m = next((s_.market for s_ in FUTURES if s_.sym(use_micro) == h.ib_symbol), None)
        if m:
            out[m] = out.get(m, 0) + int(h.qty)
    return out


def halt_all_batches(held, held_left, safety, front, use_micro: bool, today: str) -> list:
    """HALT_ALL (2026-10-07): the SAFETY closes plus ROLLS at the SAME size -- targets are exactly
    what is held, so plan_roll_orders only moves a position out of a near-expiry contract into the
    front month. Deterministic (no prices, no signal), opens no new exposure, resizes nothing."""
    rolls = plan_roll_orders(held_by_market(held, use_micro), held_left, front, BY_MARKET,
                             use_micro, today)
    return [("SAFETY", safety), ("ROLL (HALT_ALL)", rolls)]


def book_fills(state: TrendState, fills: list[dict], today: str, todays_orders: list[dict]) -> None:
    """Book each FILLED order into the ledger with its link to IB's records, and list every order
    (filled or not) for the email. Booking records the tag the ORDER actually carried, not the
    run's ORDER_REF, so the ledger can never claim a tag IB did not receive."""
    for f in fills:
        signed = f["qty"] if f["action"] == "BUY" else -f["qty"]
        if f["fill_price"]:
            state.record_fill(f["market"], signed, f["fill_price"], f["mult"],
                              today, f["symbol"], f["expiry"], f["reason"],
                              order_ref=f.get("order_ref", ""), exec_ids=f.get("exec_ids"),
                              conid=f.get("conid") or 0, commission=f.get("commission"),
                              currency=f.get("currency") or "")
            todays_orders.append(f)
        else:
            todays_orders.append({**f, "reason": f["reason"] + f" ({f['status']})"})

# Annualised vol prior for the circuit-breaker levels, from the contract-level backtest
# (algo_trading/scripts/breaker_calibration_lab.py, live config at OVERLAY_MULT 1.0): 12.4%.
# ⚠ THIS IS THE POST-CHANGE FIGURE. OVERLAY_MULT went 0.5 -> 1.0 on 2026-08-13, doubling
# exposure and so vol; the recorded nav_history still describes the HALF-SIZED book and would
# understate it for months. Update this prior whenever OVERLAY_MULT, TARGET_VOL or the market
# set changes.
VOL_PRIOR = 0.124


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
    # Budget AND overlay multiple come from config/capital_bases.json, the single source for
    # sizing (owner's decision, 2026-10-07); an env BUDGET / OVERLAY_MULT still overrides for a
    # one-off but is WARNED about when it differs. risk_guard.ALLOCATIONS stays a set of guard
    # ceilings (checked below and in log_sizing), never a source of budgets.
    _alloc_ok = check_allocations()
    if not _alloc_ok:
        logging.error("ALLOCATION: %s", _alloc_ok.reason)
    budget, bsrc = documented_sizing(ROOT, "trend-overlay")
    om, osrc = documented_sizing(ROOT, "trend-overlay", key="overlay_mult", env_var="OVERLAY_MULT")
    log_sizing("trend-overlay", budget, bsrc, getattr(state, "last_net_liq", 0.0) or None,
               overlay=om, overlay_source=osrc)
    return TrendPaperConfig(
        budget=budget,
        target_vol=float(os.getenv("TARGET_VOL", "0.10")),
        overlay_multiple=om)


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

    # KILL SWITCH (levels split 2026-10-07). HALT_HARD exits before connecting. HALT_ALL runs
    # ONLY the SAFETY closes and same-size rolls (deterministic, no price feed, no new exposure).
    # HALT freezes exposure but still ROLLS —
    # a physically-delivered contract (ZB, ZN, SIL) left past its notice date goes to DELIVERY,
    # so a halt that blocks rolls is more dangerous than the situation prompting it.
    # Report WHICH COMMIT is running before anything else. Placed above the kill switch so it
    # is recorded even on a halted run: "the box is running week-old code" is exactly the kind of
    # thing you want to learn from a halted day's log, not discover a month later.
    code_version(ROOT)
    warn_non_usd_contracts()
    _halt, _hwhy = halt_state(ROOT)
    if _halt == HALT_HARD:
        logging.error("HALTED (hard): %s — exiting without connecting. NOTE: delivery/roll safety "
                      "closes did NOT run.", _hwhy)
        push_if_alerts(ALERTS, "Trend Overlay")
        return
    halt_all = _halt == HALT_ALL
    if halt_all:
        logging.error("HALTED (all): %s — SAFETY closes and same-size rolls ONLY; nothing opened, "
                      "resized or closed for the signal", _hwhy)

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

    broker = make_broker(port=args.port, client_id=args.client_id, dry_run=False)
    logging.info("run id %s — orders tagged orderRef=%s", RUN_ID, ORDER_REF)
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
        _mu = broker.margin_cushion()
        _mlvl, _mscale, _mwhy = liquidity_check(*(_mu if _mu else (float("nan"), 0.0)),
                                                limits=MarginLimits())
        # Same call already returns NetLiq; store it so the NEXT run sizes off the real account
        # rather than the nominal anchor.
        if _mu and _mu[1] and _mu[1] > 0:
            state.last_net_liq = float(_mu[1])
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
        _base = cfg.budget              # the documented budget (was a second, separate env read)
        _eq = _base + state.realized_pnl + _unreal
        _peak = max(peak_equity(state.nav_history, _base, key="total_pnl"), _eq)
        write_equity(ROOT.parent, "trend-overlay", _eq, _peak)
        # Vol-scaled: 15/25/35 IS 1.2/2.0/2.8 sigma at trend's ~12.4% vol, so this leaves the
        # levels ~unchanged here while fixing the higher-vol equity book.
        _lv = BreakerLevels.from_vol(blended_vol(state.nav_history, _base, VOL_PRIOR,
                                                 key="total_pnl"))
        _blvl, _bscale, _bwhy = circuit_breaker(_eq, _peak, _lv)
        if _bwhy:
            (logging.error if _blvl == "halt" else logging.warning)("circuit breaker: %s", _bwhy)
        # BOOK-level: three books each down 20% all sit under their own 25% threshold while the
        # total is down 20%. Takes the WORSE of own and book.
        _bdd, _beq, _bpk, _bnote = book_drawdown(ROOT.parent)
        if _bdd is not None:
            # Book levels scale to the BOOK's OWN vol, not a hard-coded tighter set. With one
            # strategy live the book curve IS that strategy's, so the levels come out identical
            # and this adds nothing — correct, since there is no diversification to reward.
            # Skipped entirely until the book has enough history to estimate its vol.
            _bvol = book_vol(ROOT.parent)
            _lvl2, _sc2, _why2 = (circuit_breaker(_beq, _bpk, BreakerLevels.from_vol(_bvol))
                                  if _bvol else ('ok', 1.0, ''))
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
            if not args.safety_only and not halt_all:   # HALT_ALL needs no prices
                start_dt = (pd.Timestamp.today() - pd.Timedelta(days=500)).strftime("%Y-%m-%d")
                px = download_ohlcv(PROXY_ETFS, start_dt)["adj_close"]
                # A frozen feed still yields a signal, a vol estimate and a full target book —
                # all plausible, all wrong. price sanity cannot catch it: each price is valid,
                # just old.
                _lim_f = RiskLimits.for_futures(cfg.budget * cfg.overlay_multiple)
                _fresh = data_fresh(px.index, pd.Timestamp(today), _lim_f)
                if not _fresh:
                    logging.error("data staleness: %s — SAFETY closes only today", _fresh.reason)
                # PER-COLUMN staleness. data_fresh sees only the INDEX, so one market's feed
                # dying or freezing leaves the panel looking current and reaches sizing as a
                # legitimate signal. NaN the offenders so compute_targets HOLDS them (a
                # non-finite target is the hold signal) instead of trading on a stale price.
                # Deliberately per-market rather than a global halt: one dead ticker should not
                # stop the other nine from being managed.
                _stale, _sc = stale_columns(px, pd.Timestamp(today), _lim_f)
                if _stale:
                    logging.error("per-column staleness: %s — those markets will be HELD, "
                                  "not resized", _sc.reason)
                    for _c in _stale:
                        px[_c] = np.nan

            if halt_all and not args.safety_only:
                batches = halt_all_batches(held, held_left, safety, front, cfg.use_micro, today)
            elif args.safety_only or not _fresh:
                batches = [("SAFETY", safety)]
            else:
                # Current holdings, so hysteresis can hold a position whose target sits inside
                # the band. Keyed by MARKET (held is per contract-month); summed because a
                # market can straddle two expiries mid-roll.
                # Built from `held`, NOT `held_left`: a safety close is a ROLL, not an exit. Using
                # held_left made a force-closed market look flat, so re-entry needed |target| >=
                # band instead of the hold threshold -- on 2026-10-05 oil (signal +1.00, target
                # 0.70 ct) was closed for expiry and never re-bought. Orders still come from
                # held_left below, so the front contract is bought back up to target.
                held_by_mkt = held_by_market(held, cfg.use_micro)
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
            # Size the guard against the EFFECTIVE budget (budget x overlay), the same base the
            # strategy's own per-market (0.40) and gross (3.0) caps use. for_futures' 0.45/3.30
            # fracs were chosen to backstop those WITH HEADROOM, which only holds if both use the
            # same base. At OVERLAY_MULT > 1.1 the raw-budget guard becomes TIGHTER than the
            # strategy and wrongly blocks legitimate orders: at 1.5 it rejected MES ($27.5k) and
            # fx_aud ($26.4k) — 55%/53% of the RAW $50k — that sit fine under the strategy's
            # $30k (0.40 x 75k) per-market cap. It still catches a real sizing bug (5x too big
            # blows past 0.45 x effective too), so independence is preserved.
            lim = RiskLimits.for_futures(cfg.budget * cfg.overlay_multiple)
            for label, batch in batches:
                fills = broker.execute(batch, BY_MARKET, cfg.use_micro, limits=lim, held=held)
                book_fills(state, fills, today, todays_orders)

        # Let IB's portfolio/position feed catch up with THIS run's fills before we read it. The
        # feed lags a fill by a second or two, so reading immediately can show a just-traded
        # contract at its PRE-trade size and report a FALSE orphan/phantom in the reconcile
        # (fx_aud, 2026-08-13: sold 3 M6A, ledger already 0, feed still showed 3).
        broker.ib.sleep(3)
        positions = broker.portfolio_marks(FUTURES)
        # RECONCILE the state LEDGER against IB. Positions themselves are re-read from IB every
        # run, so they cannot drift — but the ledger drives REALISED-P&L accounting, and if it
        # disagrees with the broker the P&L is wrong in a way nothing else surfaces. Report only.
        _exp = {m: float(l.qty) for m, l in state.ledger.items() if l.qty}
        _act = {p["market"]: float(p["contracts"]) for p in positions}
        _d, _rnote = reconcile(_exp, _act, label="futures ledger")
        if _rnote:
            logging.warning("%s", _rnote)
            # AUTO-CORRECT the drift the reconcile just reported. IB is the source of truth for
            # positions, so snap the P&L ledger to it instead of emailing the same phantom every
            # day (rates_10y drifted for weeks this way). Bookkeeping only — never trades.
            _actful = {p["market"]: (float(p["contracts"]), float(p["avg_price"])) for p in positions}
            _mults = {s.market: s.multiplier for s in FUTURES}
            _fixed = state.resync_to_broker(_actful, today, _mults)
            if _fixed:
                logging.warning("reconcile: auto-corrected the P&L ledger to IB -> %s",
                                "; ".join(_fixed))
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
