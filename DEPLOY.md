# Trend Overlay — deployment

Cross-asset trend-following futures overlay (7 markets). **Weekly** rebalance, **daily** P&L
email. Runs on the **same IB paper account** as the magic formula but **clientId 6** (magic
formula = 5) so both connect at once. Signals come from yfinance ETF proxies; IB is used only
to trade and to mark positions. Bitbucket-only (private) — not published.

## The strategy in one paragraph
3/6/12-month time-series-momentum (sign) → inverse-vol risk-parity → 10% portfolio vol-target,
scaled by `OVERLAY_MULT` (start 0.5). Budget `$100k` (tied to the magic formula). Fixed 7-market
basket (ES/MES, ZN, GC/MGC, HG/MHG, CL/MCL, 6E/M6E, 6A/M6A) — never rotated, only long/short/resized.
Physical-delivery safety: every non-cash market is force-closed inside a per-market notice buffer
(all wider than the weekly gap), so a missed roll can't become a delivery obligation.

## Windows setup

```bat
cd C:\trading
git clone git@bitbucket-picard:picard_capital/trend-overlay.git
cd trend-overlay
python -m venv .venv && call .venv\Scripts\activate.bat
pip install -r requirements.txt
copy .env.example .env & notepad .env
```

Fill `.env`: `IB_PORT` (the paper Gateway port), `IB_CLIENT_ID=6`, `BUDGET=100000`,
`OVERLAY_MULT=0.5`, `TARGET_VOL=0.10`, and `EMAIL_USER/PASS/TO_EMAIL` (Gmail App Password).

## IB Gateway
Same paper account/Gateway as the magic formula. **Deactivate API order precautions** (else
orders hold at PendingSubmit — same gotcha as the magic formula). Error 162 is harmless (signals
come from yfinance; IB is only trade + position marks). Futures trade ~23h so timing is flexible.

## Test before scheduling
```bat
python scripts\run_trend_paper.py --selftest          # offline roll/safety logic
python scripts\run_trend_paper.py                      # offline target-book preview
python scripts\run_trend_paper.py --live --trade       # force a live trade + email (Gateway up)
```

## Schedule
A **daily** run gives a daily email; the trade leg only fires on the weekly day (Friday).

⚠ 18:00 CET is load-bearing, not arbitrary: margin in the shared IB account is claimed
FIRST-COME-FIRST-SERVED, so run order IS the priority order. It must stay between
magic-formula (16:30) and options-vrp (21:30) — Sharpe 0.96 > 0.74 > 0.52.

```bat
schtasks /Create /TN "TrendOverlayPaper" ^
  /TR "C:\trading\trend-overlay\scripts\run_trend_paper.py --live" ^
  /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 18:00 /F
```
(Wrap in a .bat that activates the venv, like the magic formula, if preferred.) Optional extra
insurance: a **daily** `--live --safety-only` run (only ever force-closes near-delivery contracts).

## Notes
- **State** (`results\paper\state.json`) lives on this box — realized P&L, trade log, inception.
  Current positions/unrealized always come live from IB, so a lost state file only loses the
  realized-P&L history, not the book. Not in git.
- Start at `OVERLAY_MULT=0.5`; raise toward 1.0 once comfortable.
- Realized P&L is booked from confirmed fills; if a weekly order ever doesn't fill within the
  wait window it's not booked (positions still reconcile against IB next run).
