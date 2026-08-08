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
    # 6/12-month TSMOM. The 3-month (63d) leg was dropped 2026-07-14: it whipsaws in chop
    # (standalone Sharpe 0.24, worst DD) and dragged the blend; (126,252) lifts backtest
    # Sharpe 0.57->0.67 on its own and is the "variant D" config (with daily rebalance).
    lookbacks: tuple[int, ...] = (126, 252)
    vol_window: int = 60
    use_micro: bool = True             # micros for sizing granularity
    overlay_multiple: float = 1.0      # scale the whole book (0.5x, 1.0x, ...)
    # --- exposure guards (added 2026-08-08) -------------------------------------------------
    # Sizing is inverse-vol, so expo ~ 1/vol and notional EXPLODES as vol vanishes. The
    # portfolio vol-target cannot catch it, because it is circular:
    #     per_mkt_vol_i = (expo_i/budget)*vol_i = target_vol/sqrt(N)   <- vol_i CANCELS
    #     est_vol = sqrt(sum per_mkt_vol^2) = target_vol, so scale = 1.0, a NO-OP.
    # The model sizes with the same collapsed estimate it uses to MEASURE risk. Measured on the
    # live 10-market basket: gross ran 2.8x budget at the median and hit 4.8x, and IEF at its
    # 2.8% vol floor implies a $225k position on a $200k book — in ONE market. There was NO
    # leverage cap here at all (the backtest's max_leverage=5.0 was never carried across).
    # Three layered guards; each fixes something the others do not. Backtested in algo_trading
    # scripts/trend_exposure_lab.py: worst-case gross 4.66x -> 3.00x and single-market 1.12x ->
    # 0.40x, for 0.02 of Sharpe, with drawdown and skew both slightly BETTER.
    vol_floor_pct: float = 0.20        # floor each market's vol at this percentile of its OWN
                                       #   history (adaptive: IEF ~5.6% vs USO ~31% normal vol,
                                       #   so an absolute floor cannot work). Also leans against
                                       #   the estimate — vol at its 1st pct mean-reverts UP.
    per_market_cap: float = 0.40       # max |notional| per market, as a fraction of budget.
                                       #   The only guard that limits CONCENTRATION.
    gross_cap: float = 3.0             # max total notional, as a multiple of budget. Backstop.
                                       #   Do NOT tighten to 2.5: it buys no drawdown and takes
                                       #   skew -0.17 -> -0.26, and skew protection is the whole
                                       #   reason the vol-target is there.


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

    # GUARD 1 — floor each market's vol at a percentile of its own trailing history. Expanding,
    # so no lookahead. Falls back to the raw vol until there is enough history to form it.
    vol_used = vol
    if cfg.vol_floor_pct and cfg.vol_floor_pct > 0:
        floor = vol.expanding(min_periods=252).quantile(cfg.vol_floor_pct)
        vol_used = vol.where(floor.isna(), np.maximum(vol, floor))

    sig, v_raw, v = signal.iloc[-1], vol.iloc[-1], vol_used.iloc[-1]

    N = len(specs)
    rows = []
    for s in specs:
        sg = float(sig[s.proxy_etf])
        vv, vv_raw = float(v[s.proxy_etf]), float(v_raw[s.proxy_etf])
        # inverse-vol risk parity: each market ~ (budget*target_vol/sqrt(N)) of risk
        expo = 0.0 if (vv == 0 or vv != vv) else sg * (cfg.budget * cfg.target_vol / np.sqrt(N)) / vv
        # GUARD 2 — cap any single market's notional. The gross cap alone does NOT stop one
        # low-vol market dominating the book.
        if cfg.per_market_cap and cfg.per_market_cap > 0:
            lim = cfg.per_market_cap * cfg.budget
            if abs(expo) > lim:
                logging.info("cap %s: $%s -> $%s (%.0f%% of budget; vol %.1f%%)",
                             s.market, f"{expo:,.0f}", f"{np.sign(expo)*lim:,.0f}",
                             100 * cfg.per_market_cap, 100 * vv_raw)
                expo = float(np.sign(expo) * lim)
        rows.append((s.market, sg, vv_raw, expo, s.notional(cfg.use_micro), s.sym(cfg.use_micro)))
    df = pd.DataFrame(rows, columns=["market", "signal", "ann_vol", "dollar_exposure", "notional", "ib_symbol"]).set_index("market")

    # aggregate vol-target (assume ~uncorrelated): scale so est. book vol ≈ target_vol.
    # NB this is ~a no-op by construction once signals are +/-1 (vol cancels, see the config
    # comment); it only bites when signals are fractional. The real protection is the guards.
    per_mkt_vol = (df["dollar_exposure"] / cfg.budget) * df["ann_vol"]
    est_vol = float(np.sqrt((per_mkt_vol ** 2).sum()))
    scale = (cfg.target_vol / est_vol) if est_vol > 0 else 0.0
    df["dollar_exposure"] *= scale * cfg.overlay_multiple

    # GUARD 3 — gross notional backstop, applied AFTER the vol-target and overlay multiple.
    if cfg.gross_cap and cfg.gross_cap > 0:
        gross = float(df["dollar_exposure"].abs().sum())
        lim = cfg.gross_cap * cfg.budget * cfg.overlay_multiple
        if gross > lim:
            logging.warning("gross cap: $%s -> $%s (%.1fx budget)",
                            f"{gross:,.0f}", f"{lim:,.0f}", cfg.gross_cap)
            df["dollar_exposure"] *= lim / gross

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
                # Store the ROOT symbol (e.g. "MHG"), NOT the localSymbol ("MHGU6"): execute()
                # matches ib_symbol against spec roots and rebuilds the contract from
                # (symbol, expiry). Using localSymbol made every close/roll/safety order fail
                # its spec lookup and get silently skipped ("no spec for MHGU6 — skipped"),
                # so old contracts never rolled out (over-target + broken delivery safety).
                out.append(HeldPosition(sym_to_mkt[c.symbol], c.symbol,
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
                order = MarketOrder(o.action, o.qty)
                order.tif = "DAY"          # explicit — trims the preset TIF cancel/resubmit (Error 10349)
                trade = self.ib.placeOrder(q[0], order)
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
