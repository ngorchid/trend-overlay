"""Persistent P&L state for the trend overlay (the strategy's memory between runs).

The overlay is a futures P&L overlay on margin (no dedicated cash), so state tracks realized
P&L via average-cost accounting rather than a NAV/cash balance. Current positions and
unrealized P&L come live from IB; this file owns realized P&L, the trade log, and inception.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class MarketLedger:
    qty: float = 0.0          # signed contracts currently held
    avg_price: float = 0.0    # average entry price (index points)
    multiplier: float = 1.0   # $ per point


@dataclass
class TrendState:
    inception_date: str | None = None
    realized_pnl: float = 0.0
    ledger: dict[str, MarketLedger] = field(default_factory=dict)
    trade_log: list[dict] = field(default_factory=list)
    nav_history: list[dict] = field(default_factory=list)   # [{date, total_pnl, spy}]
    # Account-wide NetLiquidation as of the LAST run. The budget is a fraction of it, but the
    # config is built before the broker connects, so it is read from here and refreshed after.
    # One day stale is immaterial: the budget is quantised to 10% steps. 0.0 = never seen, and
    # `allocated_budget` then falls back to the nominal share and says so in the log.
    last_net_liq: float = 0.0

    # ---- persistence ----
    @classmethod
    def load(cls, path) -> "TrendState":
        p = Path(path)
        if not p.exists():
            return cls()
        d = json.loads(p.read_text())
        d["ledger"] = {m: MarketLedger(**v) for m, v in d.get("ledger", {}).items()}
        return cls(**d)

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2, default=str))

    def ensure_inception(self, today: str) -> None:
        if self.inception_date is None:
            self.inception_date = today

    # ---- fills: average-cost realized-P&L accounting ----
    def record_fill(self, market: str, signed_qty: int, price: float, multiplier: float,
                    date: str, symbol: str = "", expiry: str = "", reason: str = "") -> float:
        """Record a fill (signed_qty: + buy, - sell). Books realized P&L when the trade
        reduces or flips the existing position. Returns the realized P&L of this fill."""
        led = self.ledger.get(market) or MarketLedger(multiplier=multiplier)
        old_qty, old_avg = led.qty, led.avg_price
        new_qty = old_qty + signed_qty
        realized = 0.0

        same_dir = old_qty == 0 or (old_qty > 0) == (signed_qty > 0)
        if same_dir:
            # adding (or opening from flat): volume-weighted average entry
            if new_qty != 0:
                led.avg_price = (old_avg * abs(old_qty) + price * abs(signed_qty)) / abs(new_qty)
        else:
            # reducing / closing / flipping: book realized on the closed portion
            closed = min(abs(signed_qty), abs(old_qty))
            realized = (price - old_avg) * closed * multiplier * (1 if old_qty > 0 else -1)
            self.realized_pnl += realized
            if abs(signed_qty) > abs(old_qty):     # flipped past flat -> residual opens at price
                led.avg_price = price

        led.qty = new_qty
        led.multiplier = multiplier
        if new_qty == 0:
            led.avg_price = 0.0
        self.ledger[market] = led

        self.trade_log.append({"date": date, "market": market, "symbol": symbol, "expiry": expiry,
                               "signed_qty": signed_qty, "price": price, "realized_pnl": round(realized, 2),
                               "reason": reason})
        return realized

    def record_snapshot(self, date: str, total_pnl: float, spy: float | None = None) -> None:
        self.nav_history = [h for h in self.nav_history if h["date"] != date]
        self.nav_history.append({"date": date, "total_pnl": round(total_pnl, 2), "spy": spy})
        self.nav_history.sort(key=lambda h: h["date"])
