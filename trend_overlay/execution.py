"""Trend-overlay execution: signal → target contracts, and an IB futures broker.

`compute_targets` is pure/offline (unit-testable, no IB): it turns the ETF-proxy price
history into a target number of (signed) contracts per market, inverse-vol risk-parity sized
and vol-targeted to the configured budget. `FuturesBroker` mirrors paper/broker.py for
futures — front-month resolution, positions (signed contract counts), market orders, and a
roll helper — with a dry_run mode so the whole loop runs offline (no orders, no connection).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .contracts import FUTURES, FutureSpec


@dataclass
class TrendPaperConfig:
    budget: float = 200_000.0          # $ risk capital allocated to the overlay
    target_vol: float = 0.10           # annualised portfolio vol target
    lookbacks: tuple[int, ...] = (63, 126, 252)
    vol_window: int = 60
    use_micro: bool = True             # micros for sizing granularity
    overlay_multiple: float = 1.0      # scale the whole book (0.5x, 1.0x, ...)


def compute_targets(proxy_prices: pd.DataFrame, cfg: TrendPaperConfig,
                    specs: list[FutureSpec] = FUTURES) -> pd.DataFrame:
    """ETF-proxy price panel -> target signed contract counts per market.

    Returns a frame indexed by market with columns: signal, ann_vol, dollar_exposure,
    contracts, notional. Uses the *latest* row of the (blended TSMOM) signal and 60d vol.
    """
    specs = [s for s in specs if s.proxy_etf in proxy_prices.columns]
    px = proxy_prices[[s.proxy_etf for s in specs]]
    rets = px.pct_change(fill_method=None)

    signal = sum(np.sign(px / px.shift(lb) - 1.0) for lb in cfg.lookbacks) / len(cfg.lookbacks)
    vol = rets.rolling(cfg.vol_window).std() * np.sqrt(252)
    sig, v = signal.iloc[-1], vol.iloc[-1]

    N = len(specs)
    rows = []
    for s in specs:
        sg, vv = float(sig[s.proxy_etf]), float(v[s.proxy_etf])
        # inverse-vol risk parity: each market ~ (budget*target_vol/sqrt(N)) of risk
        expo = 0.0 if (vv == 0 or vv != vv) else sg * (cfg.budget * cfg.target_vol / np.sqrt(N)) / vv
        rows.append((s.market, sg, vv, expo, s.notional(cfg.use_micro), s.sym(cfg.use_micro)))
    df = pd.DataFrame(rows, columns=["market", "signal", "ann_vol", "dollar_exposure", "notional", "ib_symbol"]).set_index("market")

    # aggregate vol-target (assume ~uncorrelated): scale so est. book vol ≈ target_vol
    per_mkt_vol = (df["dollar_exposure"] / cfg.budget) * df["ann_vol"]
    est_vol = float(np.sqrt((per_mkt_vol ** 2).sum()))
    scale = (cfg.target_vol / est_vol) if est_vol > 0 else 0.0
    df["dollar_exposure"] *= scale * cfg.overlay_multiple
    df["contracts"] = (df["dollar_exposure"] / df["notional"]).round().astype(int)
    df["notional_used"] = df["contracts"] * df["notional"]
    return df


def reconcile(targets: pd.Series, positions: dict[str, float]) -> list[tuple[str, str, int]]:
    """(ib_symbol -> target contracts) vs current positions -> [(symbol, BUY/SELL, qty)]."""
    orders = []
    for sym, tgt in targets.items():
        cur = positions.get(sym, 0.0)
        delta = int(round(tgt - cur))
        if delta != 0:
            orders.append((sym, "BUY" if delta > 0 else "SELL", abs(delta)))
    return orders


# --- Roll handling (pure / offline-testable) -------------------------------
@dataclass
class HeldPosition:
    market: str
    ib_symbol: str
    expiry: str          # 'YYYYMMDD' (or 'YYYYMM')
    qty: float           # signed contract count


@dataclass
class RollOrder:
    ib_symbol: str
    expiry: str          # trade THIS specific contract month
    action: str          # BUY | SELL
    qty: int
    reason: str


def days_to_expiry(expiry: str, today) -> int:
    e = str(expiry)
    ts = pd.Timestamp(e if len(e) == 8 else e + "01")
    return int((ts - pd.Timestamp(today).normalize()).days)


def safety_closes(held: list[HeldPosition], specs_by_market: dict, today) -> list[RollOrder]:
    """HARD RULE: force-close any position within its market's notice buffer of last-trade,
    regardless of signal — so a missed roll can never become a physical-delivery obligation.
    Safe to run daily; it only ever *closes*."""
    out = []
    for h in held:
        spec = specs_by_market.get(h.market)
        if spec is None or h.qty == 0:
            continue
        d = days_to_expiry(h.expiry, today)
        if d <= spec.notice_buffer_days:
            out.append(RollOrder(h.ib_symbol, h.expiry, "SELL" if h.qty > 0 else "BUY",
                                 abs(int(h.qty)),
                                 f"SAFETY force-close: {d}d to last-trade <= {spec.notice_buffer_days}d buffer"))
    return out


def plan_roll_orders(targets: dict[str, int], held: list[HeldPosition],
                     front_expiry: dict[str, str], specs_by_market: dict,
                     use_micro: bool, today) -> list[RollOrder]:
    """Roll + reconcile. For each market: close any holding NOT in the (safe) front contract,
    then adjust the front contract to the target. Because the front is always chosen beyond
    the notice buffer, this simultaneously rolls out of any near-expiry contract."""
    from collections import defaultdict
    by_mkt: dict[str, list[HeldPosition]] = defaultdict(list)
    for h in held:
        by_mkt[h.market].append(h)

    orders: list[RollOrder] = []
    for m in set(targets) | set(by_mkt):
        spec = specs_by_market.get(m)
        if spec is None:
            continue
        front = front_expiry.get(m)
        sym = spec.sym(use_micro)
        in_front = 0
        for h in by_mkt.get(m, []):
            if front is not None and str(h.expiry) == str(front):
                in_front += int(h.qty)
            else:  # holding in a non-front (older / near-expiry) contract -> roll out
                d = days_to_expiry(h.expiry, today)
                orders.append(RollOrder(h.ib_symbol, h.expiry, "SELL" if h.qty > 0 else "BUY",
                                        abs(int(h.qty)), f"roll out of {h.expiry} ({d}d to expiry)"))
        if front is not None:
            delta = int(targets.get(m, 0)) - in_front
            if delta != 0:
                orders.append(RollOrder(sym, front, "BUY" if delta > 0 else "SELL",
                                        abs(delta), f"reconcile front {front} to target"))
    return orders


class FuturesBroker:
    """ib_insync futures bridge (front-month resolution, positions, orders, roll)."""

    def __init__(self, host="127.0.0.1", port=7497, client_id=6, dry_run=False):
        self.host, self.port, self.client_id, self.dry_run = host, port, client_id, dry_run
        self.ib = None
        self._front: dict[str, object] = {}

    def connect(self, timeout: int = 15) -> bool:
        from ib_insync import IB
        self.ib = IB()
        try:
            self.ib.connect(self.host, self.port, clientId=self.client_id, timeout=timeout)
            logging.info("IB futures connected (clientId %s, port %s)", self.client_id, self.port)
            return True
        except Exception as e:  # noqa: BLE001
            logging.error("IB connect failed: %s", e)
            return False

    def disconnect(self):
        if self.ib and self.ib.isConnected():
            self.ib.disconnect()

    def front_future(self, spec: FutureSpec, use_micro: bool):
        """Qualify the nearest-expiry contract that is safely BEYOND the market's notice buffer,
        so the 'front' we trade is always clear of the delivery/first-notice window."""
        key = spec.sym(use_micro)
        if key in self._front:
            return self._front[key]
        from ib_insync import Future
        base = Future(symbol=key, exchange=spec.exchange, currency=spec.currency)
        try:
            details = self.ib.reqContractDetails(base)
            cutoff = pd.Timestamp.today() + pd.Timedelta(days=spec.notice_buffer_days)
            cands = sorted((d.contract for d in details), key=lambda c: c.lastTradeDateOrContractMonth)
            front = next((c for c in cands
                          if pd.Timestamp(c.lastTradeDateOrContractMonth) >= cutoff), None)
            self._front[key] = front
        except Exception as e:  # noqa: BLE001
            logging.warning("front_future failed for %s: %s", key, e)
            self._front[key] = None
        return self._front[key]

    def front_expiries(self, specs, use_micro: bool) -> dict[str, str]:
        """{market -> front contract's lastTradeDateOrContractMonth} for roll planning."""
        out = {}
        for s in specs:
            c = self.front_future(s, use_micro)
            if c is not None:
                out[s.market] = c.lastTradeDateOrContractMonth
        return out

    def portfolio_marks(self, specs) -> list[dict]:
        """Per-market position marks from IB's portfolio feed (works without a market-data
        subscription): {market, symbol, contracts, avg_price(points), mark(points), unrealized_pnl}.
        IB's averageCost for a future = price × multiplier, so avg_price = averageCost/multiplier."""
        sym_to = {}
        for s in specs:
            sym_to[s.symbol] = (s.market, s.multiplier)
            if s.micro_symbol:
                sym_to[s.micro_symbol] = (s.market, s.micro_multiplier or s.multiplier)
        out = []
        if self.dry_run or self.ib is None:
            return out
        for it in self.ib.portfolio():
            c = it.contract
            if c.secType in ("FUT", "CONTFUT") and c.symbol in sym_to and it.position:
                market, mult = sym_to[c.symbol]
                out.append({"market": market, "symbol": c.symbol, "contracts": it.position,
                            "avg_price": (it.averageCost / mult) if mult else it.averageCost,
                            "mark": it.marketPrice, "unrealized_pnl": it.unrealizedPNL})
        return out

    def held_positions(self, specs) -> list[HeldPosition]:
        """Current futures positions WITH their contract month (needed to detect rolls)."""
        sym_to_mkt = {s.symbol: s.market for s in specs}
        sym_to_mkt.update({s.micro_symbol: s.market for s in specs if s.micro_symbol})
        out: list[HeldPosition] = []
        if self.dry_run or self.ib is None:
            return out
        for p in self.ib.positions():
            c = p.contract
            if c.secType in ("FUT", "CONTFUT") and c.symbol in sym_to_mkt and p.position:
                out.append(HeldPosition(sym_to_mkt[c.symbol], c.localSymbol or c.symbol,
                                        c.lastTradeDateOrContractMonth, p.position))
        return out

    def execute(self, orders: list[RollOrder], specs_by_market: dict, use_micro: bool, wait: float = 4.0) -> list[dict]:
        """Place a list of RollOrders, each on its specific (symbol, expiry) contract.
        Returns fills for state accounting:
            {market, symbol, expiry, action, qty, mult, fill_price, status, reason}
        (fill_price None if not filled within `wait` — e.g. a queued/off-hours order)."""
        from ib_insync import Future, MarketOrder
        fills: list[dict] = []
        for o in orders:
            spec = next((s for s in specs_by_market.values()
                         if o.ib_symbol in (s.symbol, s.micro_symbol)), None)
            mult = spec.mult(use_micro) if spec else 1.0
            base = {"market": spec.market if spec else "?", "symbol": o.ib_symbol,
                    "expiry": o.expiry, "action": o.action, "qty": o.qty, "mult": mult,
                    "reason": o.reason}
            if self.dry_run:
                logging.info("[DRY RUN] %s %d %s %s  (%s)", o.action, o.qty, o.ib_symbol, o.expiry, o.reason)
                fills.append({**base, "fill_price": None, "status": "dryrun"})
                continue
            if spec is None:
                logging.warning("no spec for %s — skipped", o.ib_symbol); continue
            c = Future(symbol=o.ib_symbol, lastTradeDateOrContractMonth=o.expiry,
                       exchange=spec.exchange, currency=spec.currency)
            try:
                q = self.ib.qualifyContracts(c)
                if not q:
                    logging.warning("could not qualify %s %s — skipped", o.ib_symbol, o.expiry); continue
                trade = self.ib.placeOrder(q[0], MarketOrder(o.action, o.qty))
                self.ib.sleep(wait)
                st = trade.orderStatus.status
                fp = trade.orderStatus.avgFillPrice or None
                logging.info("%s %d %s %s -> %s%s  (%s)", o.action, o.qty, o.ib_symbol, o.expiry,
                             st, f" @ {fp}" if fp else "", o.reason)
                fills.append({**base, "fill_price": float(fp) if fp else None, "status": st})
            except Exception as e:  # noqa: BLE001
                logging.error("order failed %s %s %s: %s", o.action, o.ib_symbol, o.expiry, e)
                fills.append({**base, "fill_price": None, "status": "error"})
        return fills
