"""Daily HTML email for the trend overlay — magic-formula style.

Positions + marks come from IB (source of truth, passed in); realized P&L + inception come
from TrendState. Shows: positions table (market/side/contracts/avg/mark/unrealized), today's
trades, realized + unrealized + total P&L since inception, and SPY for context (uncorrelated
diversifier, so this is context not a benchmark).
"""
from __future__ import annotations

import logging
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from .state import TrendState



def _code_line() -> str:
    """The commit this run used (risk_guard.code_version), as the report's last line."""
    try:
        from risk_guard import code_version_note
        note = code_version_note()
    except Exception:  # noqa: BLE001 -- reporting must never break the run
        note = ""
    return (f"<p style='color:#64748b;font-size:11px'>Code: "
            f"{note or 'version not recorded'}</p>")

def _pct(x) -> str:
    return "—" if x is None else f"{x*100:+.2f}%"


def _table(headers, rows) -> str:
    th = "".join(f"<th style='text-align:left;padding:4px 10px;border-bottom:2px solid #1a3c5e'>{h}</th>" for h in headers)
    trs = "".join("<tr>" + "".join(f"<td style='padding:3px 10px;border-bottom:1px solid #e2e8f0'>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table style='border-collapse:collapse;font-family:monospace;font-size:13px'><tr>{th}</tr>{trs}</table>"


def _daily_pnl(state: TrendState, total_today: float, today: str) -> float | None:
    """Change in TOTAL P&L (realized + unrealized) since the last recorded run. None on the very
    first run (no prior snapshot). nav_history is kept date-sorted, and record_snapshot may have
    already written today's row, so the baseline is the most recent row that is NOT today's."""
    prev = [h for h in state.nav_history if h.get("date") != today and h.get("total_pnl") is not None]
    return (total_today - float(prev[-1]["total_pnl"])) if prev else None


def build_email_body(state: TrendState, positions: list[dict], todays_orders: list[dict],
                     spy_day, spy_incep, today: str) -> str:
    """positions: [{market, symbol, contracts, avg_price, mark, unrealized_pnl}]
    todays_orders: [{action, qty, symbol, expiry, reason}]"""
    unreal = sum(p.get("unrealized_pnl") or 0.0 for p in positions)
    total = state.realized_pnl + unreal
    daily = _daily_pnl(state, total, today)
    label = os.getenv("BOOK_LABEL", "Paper")
    _dcolor = "#1a7f37" if (daily or 0) >= 0 else "#b91c1c"
    _dcell = (f"<span style='color:{_dcolor}'>${daily:+,.0f}</span>" if daily is not None
              else "<span style='color:#64748b'>— (first run)</span>")

    pos_rows = []
    for p in sorted(positions, key=lambda x: x["market"]):
        c = p["contracts"]
        side = "LONG" if c > 0 else "SHORT" if c < 0 else "flat"
        u = p.get("unrealized_pnl")
        color = "#1a7f37" if (u or 0) >= 0 else "#b91c1c"
        pos_rows.append((p["market"], p.get("symbol", ""), side, f"{c:+g}",
                         f"{p.get('avg_price', 0):,.2f}", f"{p.get('mark', 0):,.2f}",
                         f"<span style='color:{color}'>${(u or 0):+,.0f}</span>"))
    pos_tbl = _table(["Market", "Sym", "Side", "Contracts", "Avg", "Mark", "Unrealized"],
                     pos_rows) if pos_rows else "<i>no open positions</i>"

    if todays_orders:
        tr_rows = [(o["action"], f"{o['qty']:g}", o.get("symbol", ""), o.get("expiry", ""), o.get("reason", "")) for o in todays_orders]
        trades_tbl = _table(["Action", "Qty", "Sym", "Expiry", "Reason"], tr_rows)
    else:
        trades_tbl = "<i>no trades today</i>"

    summary = f"""
    <table style='font-family:monospace;font-size:13px;margin:8px 0'>
      <tr><td style='padding:2px 14px 2px 0'>Daily P&amp;L (since last run)</td><td><b>{_dcell}</b></td></tr>
      <tr><td>Total P&amp;L (since {state.inception_date})</td><td><b>${total:+,.0f}</b></td></tr>
      <tr><td>Realized</td><td>${state.realized_pnl:+,.0f}</td></tr>
      <tr><td>Unrealized</td><td>${unreal:+,.0f}</td></tr>
      <tr><td>Open positions</td><td>{sum(1 for p in positions if p['contracts'])}</td></tr>
      <tr><td style='padding-top:8px'>SPY (context)</td><td style='padding-top:8px'>today {_pct(spy_day)} &nbsp; since inception {_pct(spy_incep)}</td></tr>
    </table>"""

    _foot = "Real capital." if label.upper() == "LIVE" else "Paper trading."
    return f"""<html><body style='font-family:sans-serif;color:#1e293b'>
    <h2 style='color:#1a3c5e'>Trend Overlay {label} — {today}</h2>
    {summary}
    <h3 style='color:#1a3c5e'>Positions</h3>
    {pos_tbl}
    <h3 style='color:#1a3c5e'>Today's trades</h3>
    {trades_tbl}
    <p style='color:#64748b;font-size:11px;margin-top:14px'>Cross-asset trend (7 futures markets), weekly rebalance, inverse-vol risk parity, {os.getenv('TARGET_VOL','0.10')} vol-target × {os.getenv('OVERLAY_MULT','0.5')}. Uncorrelated diversifier — SPY shown for context, not as a benchmark. {_foot}</p>
    {_code_line()}
    </body></html>"""


def send_report(state: TrendState, positions: list[dict], todays_orders: list[dict],
                spy_day, spy_incep, today: str, dry_run: bool = False, alerts=None) -> str:
    body = build_email_body(state, positions, todays_orders, spy_day, spy_incep, today)
    unreal = sum(p.get("unrealized_pnl") or 0.0 for p in positions)
    total = state.realized_pnl + unreal
    if alerts is not None and getattr(alerts, 'records', None):
        body = alerts.html() + body
    _mark = (f'[{alerts.worst} x{len(alerts.records)}] '
             if alerts is not None and getattr(alerts, 'worst', None) else '')
    _label = os.getenv("BOOK_LABEL", "Paper")
    _daily = _daily_pnl(state, total, today)
    _dstr = f"day ${_daily:+,.0f}, " if _daily is not None else ""
    subject = _mark + (f"Trend Overlay {_label} — {today}: {_dstr}total ${total:+,.0f} "
                       f"({len(todays_orders)} trades)")
    user, pw, to = os.getenv("EMAIL_USER"), os.getenv("EMAIL_PASS"), os.getenv("TO_EMAIL")
    if dry_run or not all([user, pw, to]):
        if not dry_run:
            logging.warning("Email creds not set — skipping send.")
        return body
    msg = MIMEMultipart("alternative")
    msg["From"], msg["To"], msg["Subject"] = user, to, subject
    msg.attach(MIMEText(body, "html"))
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(user, pw)
            s.send_message(msg)
        logging.info("Report sent to %s", to)
    except Exception as e:  # noqa: BLE001
        logging.error("Email failed: %s", e)
    return body
