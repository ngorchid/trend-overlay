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

# risk_guard.py lives at the REPO ROOT, not in this package — one identical copy per repo, kept
# in sync by hand. The runner puts ROOT on sys.path, so this resolves when invoked via scripts/.
try:
    from risk_guard import check_order
except ImportError:  # guard unavailable -> execute() must still work, just unguarded
    check_order = None


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
    # --- covariance vol-target (added 2026-08-08) --------------------------------------------
    # The old estimator assumed markets were UNCORRELATED, so est_vol = sqrt(sum (w_i*vol_i)^2).
    # But rates correlate with rates (ZN/ZB), FX with FX (6E/6A/6J), and in risk-off everything
    # correlates — so it understated portfolio vol and the book ran 12-14% against a 10% target.
    # Now est_vol = sqrt(w' Sigma w) with Sigma = D R D, D = diag(vol_used), R = sample
    # correlation shrunk toward the identity by (1 - corr_weight).
    # Backtested (live 10 basket, variant D, guards on): realised vol 13.9% -> 10.6%, Sharpe
    # 0.61 -> 0.74, maxDD -32.5% -> -20.2%, skew -0.21 -> -0.09, with return UNCHANGED (+7.8%
    # vs +7.6%). Monotone in corr_weight and holds in BOTH sub-periods, so not a fit.
    # NB Sigma is only ever used as a QUADRATIC FORM (w'Sigma w) and never inverted, so
    # estimation error matters far less here than it would in an optimiser — which is why the
    # near-raw sample correlation beats heavy shrinkage, contrary to the usual intuition.
    corr_window: int = 252             # correlations are more stable than vols -> longer window
    corr_weight: float = 0.90          # weight on the sample correlation; the rest goes to the
                                       #   CONSTANT-CORRELATION target (every pair = the sample
                                       #   mean), NOT the identity. Shrinking toward I would
                                       #   shrink the off-diagonals toward zero, LOWERING
                                       #   est_vol and RAISING leverage — anti-conservative when
                                       #   correlations are mostly positive (mean |rho| ~0.35
                                       #   here). Empirically it barely matters (Sharpe 0.72-0.74
                                       #   across every target/weight tested); the big win was
                                       #   going from independence to ANY correlation-aware
                                       #   estimate. This just makes the shrinkage err the safe
                                       #   way.
    scale_clip: tuple[float, float] = (0.2, 1.5)   # bound the vol-target multiplier
    # --- hysteresis on the contract count (added 2026-08-11) --------------------------------
    # round() puts a knife-edge at exactly 0.5 contracts, so a market whose target sits near the
    # boundary flips 0->1->0 on tiny changes in vol or signal, each flip a full-contract round
    # trip. At $200k, rates_10y flipped 12.4x/yr at $112,000 a time = $1.39M of notional churned
    # by noise. It does NOT shrink with budget, it MOVES: every budget parks some market on its
    # own boundary. Hold n unless |target - n| >= band, then move to round(target); reversing
    # then needs a 0.4-contract move rather than an arbitrarily small one.
    # NB band 0.5 is EXACTLY today's behaviour (round() changes precisely when |target-n|>=0.5),
    # which the lab uses as a null check and reproduces to the digit.
    # Backtested contract-level in algo_trading/scripts/trend_hysteresis_lab.py, net of $0.85/ct
    # + 1bp: Sharpe 0.655->0.801 at $100k and 0.619->0.715 at $200k, maxDD -18.8%->-15.0% and
    # -26.4%->-25.1%. Chosen on SUB-PERIOD ROBUSTNESS, not on the full-sample peak: 0.7 beats
    # the baseline in ALL FOUR cells (2 budgets x 2 halves) and sits mid-plateau (0.6/0.7/0.8 all
    # survive), so it is not the argmax of any single cell. 0.9-1.0 look better in one cell and
    # worse in another -- the classic fitted spike.
    hysteresis_band: float = 0.70


def _portfolio_vol(w: np.ndarray, vols: np.ndarray, rets: pd.DataFrame,
                   cfg: TrendPaperConfig) -> float:
    """Annualised book vol from the covariance: sqrt(w' Sigma w), Sigma = D R D.

    Falls back to the independence assumption when there is not enough history to form a
    correlation matrix (early in the sample, or if a market has just been added).
    """
    n = len(w)
    indep = float(np.sqrt(np.nansum((w * vols) ** 2)))
    if not cfg.corr_weight or len(rets) < cfg.corr_window:
        return indep
    R = rets.tail(cfg.corr_window).corr().values
    if R.shape != (n, n) or not np.isfinite(R).all():
        return indep
    off = ~np.eye(n, dtype=bool)
    target = np.full((n, n), float(R[off].mean()))       # constant-correlation target
    np.fill_diagonal(target, 1.0)
    R = cfg.corr_weight * R + (1.0 - cfg.corr_weight) * target
    D = np.diag(np.nan_to_num(vols))
    var = float(w @ (D @ R @ D) @ w)
    return float(np.sqrt(var)) if var > 0 else indep


def compute_targets(proxy_prices: pd.DataFrame, cfg: TrendPaperConfig,
                    specs: list[FutureSpec] = FUTURES,
                    held: dict[str, int] | None = None) -> pd.DataFrame:
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

    # aggregate vol-target: scale so estimated book vol ≈ target_vol, using the FULL covariance
    # (see the config comment for why assuming independence understated vol by 20-30%).
    w = (df["dollar_exposure"] / cfg.budget).values          # fraction of budget per market
    sig_vec = v[[s.proxy_etf for s in specs]].values         # floored vols, aligned to df
    est_vol = _portfolio_vol(w, sig_vec, rets[[s.proxy_etf for s in specs]], cfg)
    lo, hi = cfg.scale_clip
    scale = float(np.clip(cfg.target_vol / est_vol, lo, hi)) if est_vol > 0 else 0.0
    logging.info("vol-target: est book vol %.2f%% -> scale %.2f (target %.1f%%)",
                 100 * est_vol, scale, 100 * cfg.target_vol)
    df["dollar_exposure"] *= scale * cfg.overlay_multiple

    # GUARD 2b — RE-APPLY the per-market cap AFTER scaling. Guard 2 above runs before the
    # vol-target multiplier, so any scale>1 silently undid it: measured 2026-08-11 with scale
    # 1.16, "capped" markets sat at 46-49% of budget against a 40% cap, and scale_clip permits
    # up to 1.5 (=60%). Capping only once, before a multiplier, is not a cap.
    if cfg.per_market_cap and cfg.per_market_cap > 0:
        lim2 = cfg.per_market_cap * cfg.budget * cfg.overlay_multiple
        over = df["dollar_exposure"].abs() > lim2
        if over.any():
            logging.info("post-scale cap: %s -> %.0f%% of budget",
                         ", ".join(df.index[over]), 100 * cfg.per_market_cap)
            df.loc[over, "dollar_exposure"] = (np.sign(df.loc[over, "dollar_exposure"]) * lim2)

    # GUARD 3 — gross notional backstop, applied AFTER the vol-target and overlay multiple.
    if cfg.gross_cap and cfg.gross_cap > 0:
        gross = float(df["dollar_exposure"].abs().sum())
        lim = cfg.gross_cap * cfg.budget * cfg.overlay_multiple
        if gross > lim:
            logging.warning("gross cap: $%s -> $%s (%.1fx budget)",
                            f"{gross:,.0f}", f"{lim:,.0f}", cfg.gross_cap)
            df["dollar_exposure"] *= lim / gross

    # GUARD 2c — the cap bounds TARGET DOLLARS, but the position we actually hold is a WHOLE
    # number of contracts, and one contract can exceed the cap on its own. Measured 2026-08-11:
    # rates_30y targeted -$71,910 against an $80,000 cap, yet one ZB is $115,000 = 57.5% of a
    # $200k budget. Where a single contract breaches the cap, hold ZERO rather than blow through
    # it -- the market is simply too chunky for this budget, which is a sizing fact, not a
    # rounding preference.
    # Hysteresis needs the CURRENT holding, so `held` (market -> signed contracts) must be
    # passed in from the broker. Without it the band cannot apply and we fall back to round().
    raw = df["dollar_exposure"] / df["notional"]
    if cfg.hysteresis_band and cfg.hysteresis_band > 0 and held is not None:
        band = float(cfg.hysteresis_band)
        ct = []
        for m, x in raw.items():
            n = int(held.get(m, 0))
            # A non-finite target means missing data, NOT a signal to flatten: HOLD. Writing
            # this as `not isfinite(x) or ...` sends NaN into int(round(nan)), which raises.
            if not np.isfinite(x):
                ct.append(n)
            else:
                ct.append(int(np.round(x)) if abs(x - n) >= band else n)
        df["contracts"] = pd.Series(ct, index=df.index).astype(int)
    else:
        df["contracts"] = raw.round().astype(int)
    if cfg.per_market_cap and cfg.per_market_cap > 0:
        lim3 = cfg.per_market_cap * cfg.budget * cfg.overlay_multiple
        too_big = (df["contracts"].abs() * df["notional"]) > lim3 * 1.001
        if too_big.any():
            for m in df.index[too_big]:
                n_ct = int(df.at[m, "contracts"])
                held = abs(n_ct) * df.at[m, "notional"]
                fits = int(np.floor(lim3 / df.at[m, "notional"]))
                logging.warning("rounded position over cap: %s %dct = $%s > cap $%s -> %dct%s",
                                m, n_ct, f"{held:,.0f}", f"{lim3:,.0f}", fits,
                                "  (one contract alone exceeds the cap)" if fits == 0 else "")
            df.loc[too_big, "contracts"] = ((np.sign(df.loc[too_big, "dollar_exposure"]) *
                                             np.floor(lim3 / df.loc[too_big, "notional"]))
                                            .astype(int))
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

    def margin_usage(self) -> tuple[float, float] | None:
        """(maintenance margin used, net liquidation) for the WHOLE account, or None.

        Account-wide on purpose: three strategies share this account, so the margin constraint
        genuinely is shared and one strategy can legitimately be blocked by another's usage.
        Being blocked beats being liquidated. MaintMarginReq rather than FullInitMarginReq
        because maintenance is what an actual liquidation is measured against.

        Returns None on any failure, so the caller reports "unknown" rather than assuming healthy.
        """
        if self.dry_run or self.ib is None:
            return None
        try:
            rows = {r.tag: r for r in self.ib.accountSummary()}
            mm = rows.get("MaintMarginReq") or rows.get("FullMaintMarginReq")
            nl = rows.get("NetLiquidation")
            if not mm or not nl:
                return None
            return float(mm.value), float(nl.value)
        except Exception as e:  # noqa: BLE001
            logging.warning("margin_usage failed: %s", e)
            return None

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

    def execute(self, orders: list[RollOrder], specs_by_market: dict, use_micro: bool,
                wait: float = 20.0, limits=None) -> list[dict]:
        """Place a list of RollOrders, each on its specific (symbol, expiry) contract.
        Returns fills for state accounting:
            {market, symbol, expiry, action, qty, mult, fill_price, status, reason}
        (fill_price None if not filled within `wait` — e.g. a queued/off-hours order).

        `limits` (a RiskLimits) enables the independent pre-trade guard. It re-derives each
        order's notional from the budget rather than trusting compute_targets, because a guard
        reusing the strategy's arithmetic cannot catch the strategy's own bug — and this file had
        exactly such a bug (a per-market cap that was applied before a multiplier and so never
        bound). SAFETY/delivery closes are exempt: blocking a forced close would leave a
        physically-delivered contract to go to delivery, which is far worse than an oversized
        position.
        """
        from ib_insync import Future, MarketOrder
        fills: list[dict] = []
        gross_seen = 0.0
        for o in orders:
            spec = next((s for s in specs_by_market.values()
                         if o.ib_symbol in (s.symbol, s.micro_symbol)), None)
            mult = spec.mult(use_micro) if spec else 1.0
            base = {"market": spec.market if spec else "?", "symbol": o.ib_symbol,
                    "expiry": o.expiry, "action": o.action, "qty": o.qty, "mult": mult,
                    "reason": o.reason}
            if limits is not None and "SAFETY" not in (o.reason or "").upper():
                notl = spec.notional(use_micro) if spec else 0.0
                px = notl / mult if mult else 0.0
                chk = check_order(o.ib_symbol, o.action, abs(o.qty), px, mult, limits,
                                  gross_notional=gross_seen,
                                  max_gross_frac=limits.max_gross_frac)
                if not chk:
                    logging.warning("RISK REJECT %s", chk.reason)
                    fills.append({**base, "fill_price": None, "status": "RiskRejected"})
                    continue
                gross_seen += abs(o.qty) * notl
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
                # Poll up to `wait`s, returning as soon as the order reaches a terminal state. A
                # single fixed sleep read the status while still PreSubmitted, so the email showed
                # unfilled orders that had actually filled a second later. Liquid names return in ~1s.
                waited = 0.0
                while waited < wait:
                    self.ib.sleep(1.0)
                    waited += 1.0
                    if trade.orderStatus.status in ("Filled", "Cancelled", "ApiCancelled", "Inactive", "Rejected"):
                        break
                st = trade.orderStatus.status
                fp = trade.orderStatus.avgFillPrice or None
                logging.info("%s %d %s %s -> %s%s  (%s)", o.action, o.qty, o.ib_symbol, o.expiry,
                             st, f" @ {fp}" if fp else "", o.reason)
                fills.append({**base, "fill_price": float(fp) if fp else None, "status": st})
            except Exception as e:  # noqa: BLE001
                logging.error("order failed %s %s %s: %s", o.action, o.ib_symbol, o.expiry, e)
                fills.append({**base, "fill_price": None, "status": "error"})
        return fills
