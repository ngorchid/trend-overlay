"""Tests for contract values from IB prices (2026-10-14, Appendix E #23).

WHY THIS EXISTS. The sizer and the order guard valued every contract at a stored approximation in
contracts.py ("offline sizing"), and those had drifted 40-80% below real: MES ~$38k against $27.5k
stored, the gold micro ~$43k against $30k, the silver micro ~$62k against $35k. A dollar target
became too many contracts, and the $30k per-market cap held only in stored dollars: one MES was
really ~$38k. Now each contract is valued from IB's price, the sizer and the guard use the SAME
values, and a market IB cannot price falls back to the stored value with a WARNING.

Every case drives the real code: checked_values, contract_value, compute_targets,
FuturesBroker.execute (dry run, which still runs the guard) and main() with a fake broker.
scripts/mutate_contract_values.py seeds the faults.

Run: python3 scripts/test_contract_values.py
"""
from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

import risk_guard as rg  # noqa: E402
import run_trend_paper as runner  # noqa: E402
from risk_guard import RiskLimits  # noqa: E402
from trend_overlay.contracts import BY_MARKET, FUTURES, PROXY_ETFS  # noqa: E402
from trend_overlay.execution import (FuturesBroker, RollOrder, TrendPaperConfig,  # noqa: E402
                                     VALUE_SANITY, checked_values, compute_targets,
                                     contract_value)

logging.disable(logging.CRITICAL)
_fails: list[str] = []
_ran = 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global _ran
    _ran += 1
    if not cond:
        _fails.append(f"{label}  | {detail}")
    print(f"  [{'ok ' if cond else 'FAIL'}] {label}" + ("" if cond else f"   <- {detail}"))


def safe(fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception as e:  # noqa: BLE001 -- a crash is a FAILED check here, not a test crash
        return f"crashed: {type(e).__name__}: {e}"


ES = BY_MARKET["equity_us"]
STORED_MES = ES.notional(True)              # 27,500
REAL_MES = 38_000.0

# =============================================================================================
print("checked_values: WHICH PRICE-DERIVED VALUES ARE USED")
raw = {"equity_us": REAL_MES, "gold": 43_000.0, "oil": 7_000.0 * 100,      # oil: a 100x unit error
       "fx_aud": float("nan"), "copper": 0.0}
vals, notes = checked_values(raw, FUTURES, True)
check("a plausible value is kept (MES $38k, gold $43k)",
      vals.get("equity_us") == REAL_MES and vals.get("gold") == 43_000.0, str(vals))
check("a value 100x off (a unit error) is refused, with a note naming it",
      "oil" not in vals and any("oil" in n and "refused" in n for n in notes), str(notes))
check("NaN and zero are refused as 'no price' (not as an implausible price)",
      "fx_aud" not in vals and "copper" not in vals
      and any(n.startswith("fx_aud") and "no price" in n for n in notes)
      and any(n.startswith("copper") and "no price" in n for n in notes), str(notes))
check("a market IB did not price gets a note saying it is sized on the stored value",
      any(n.startswith("rates_10y") and "stored value" in n for n in notes), str(notes))
edge_in = {"equity_us": STORED_MES * VALUE_SANITY * 0.999}
edge_out = {"equity_us": STORED_MES * VALUE_SANITY * 1.001}
check(f"the sanity band is {VALUE_SANITY:g}x: just inside kept, just outside refused",
      "equity_us" in checked_values(edge_in, FUTURES, True)[0]
      and "equity_us" not in checked_values(edge_out, FUTURES, True)[0], "")
low = {"equity_us": STORED_MES / VALUE_SANITY * 0.999}
check("...on the low side too", "equity_us" not in checked_values(low, FUTURES, True)[0], "")

print("\ncontract_value: ONE CONTRACT'S $ VALUE")
check("a given value wins", contract_value(ES, True, {"equity_us": REAL_MES}) == REAL_MES, "")
check("no value -> the stored approximation", contract_value(ES, True, {}) == STORED_MES
      and contract_value(ES, True, None) == STORED_MES, "")
check("a NaN value -> the stored approximation",
      contract_value(ES, True, {"equity_us": float("nan")}) == STORED_MES, "")

# =============================================================================================
print("\ncompute_targets: THE SIZER USES THE VALUES")
IDX = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=700)


def panel(seed=3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = {}
    for i, etf in enumerate(PROXY_ETFS):
        r = 0.0006 + rng.standard_normal(len(IDX)) * (0.14 / np.sqrt(252))
        out[etf] = 100 * np.exp(np.cumsum(r))
    return pd.DataFrame(out, index=IDX)


PX = panel()
CFG = TrendPaperConfig(budget=50_000, overlay_multiple=1.5)
CAP = CFG.per_market_cap * CFG.budget * CFG.overlay_multiple          # $30,000
base = compute_targets(PX, CFG)
T = float(base.at["equity_us", "dollar_exposure"])
check("precondition: the equity target buys one MES at the stored value (target >= half a contract)",
      abs(T) >= 0.5 * STORED_MES and int(base.at["equity_us", "contracts"]) != 0,
      f"target {T:,.0f}")
real = compute_targets(PX, CFG, values={"equity_us": REAL_MES})
check("with the real MES value ($38k > the $30k cap) the book holds NO MES",
      int(real.at["equity_us", "contracts"]) == 0, str(real.loc["equity_us"].to_dict()))
check("the frame carries the value used, and notional_used is in real dollars",
      real.at["equity_us", "notional"] == REAL_MES and base.at["equity_us", "notional"] == STORED_MES, "")
check("the dollar target itself does not depend on the contract value",
      np.allclose(real["dollar_exposure"].to_numpy(), base["dollar_exposure"].to_numpy(), equal_nan=True), "")
small = abs(T) / 0.4                       # raw = 0.4 of a contract -> rounds to 0
fit = min(abs(T) / 0.8, CAP * 0.99)        # raw >= 0.8 of a contract and within the cap -> 1
check("rounding uses the value: target/value 0.4 -> 0 contracts, >= 0.8 within the cap -> 1",
      int(compute_targets(PX, CFG, values={"equity_us": small}).at["equity_us", "contracts"]) == 0
      and abs(int(compute_targets(PX, CFG, values={"equity_us": fit}).at["equity_us", "contracts"])) >= 1,
      f"T={T:,.0f} small={small:,.0f} fit={fit:,.0f}")
allreal = {m: contract_value(BY_MARKET[m], True) * 1.5 for m in BY_MARKET}
r2 = compute_targets(PX, CFG, values=allreal)
check("with every value raised 1.5x, no market exceeds the cap in those (real) dollars",
      bool((r2["contracts"].abs() * r2["notional"] <= CAP * 1.001).all()), str(r2["notional_used"].to_dict()))

# =============================================================================================
print("\nFuturesBroker.execute: THE ORDER GUARD USES THE SAME VALUES")
fb = FuturesBroker(dry_run=True)
lim = RiskLimits.for_futures(CFG.budget * CFG.overlay_multiple)        # 0.50 x $75k = $37.5k / position
order = [RollOrder("MES", "20261218", "BUY", 1, "target")]
stored_fill = safe(fb.execute, order, BY_MARKET, True, limits=lim)
real_fill = safe(fb.execute, order, BY_MARKET, True, limits=lim, values={"equity_us": REAL_MES})
check("at the stored $27.5k one MES passes the guard (dry run)",
      isinstance(stored_fill, list) and stored_fill and stored_fill[0]["status"] == "dryrun", str(stored_fill))
check("at the real $38k the guard rejects it (over the $37.5k position limit)",
      isinstance(real_fill, list) and real_fill and real_fill[0]["status"] == "RiskRejected", str(real_fill))

# =============================================================================================
print("\nFuturesBroker.contract_prices: WHERE THE PRICES COME FROM (fake IB)")


class FakeTicker:
    def __init__(self, mp, close=float("nan")):
        self._mp, self.close = mp, close

    def marketPrice(self):
        return self._mp


class FakeIB:
    def __init__(self, portfolio, quotes, fail_quotes=False):
        self._pf, self._q, self.fail, self.types, self.asked = portfolio, quotes, fail_quotes, [], []

    def portfolio(self):
        return self._pf

    def reqMarketDataType(self, t):
        self.types.append(t)

    def reqTickers(self, *contracts):
        if self.fail:
            raise RuntimeError("no market data permission")
        self.asked += [c.symbol for c in contracts]
        return [self._q.get(c.symbol, FakeTicker(float("nan"))) for c in contracts]


def pf_item(symbol, mult, price, sec="FUT"):
    return NS(contract=NS(symbol=symbol, multiplier=str(mult), secType=sec), marketPrice=price)


def broker_with(ib):
    b = FuturesBroker(dry_run=False)
    b.ib = ib
    b.front_future = lambda spec, micro: NS(symbol=spec.sym(micro))
    return b


ib = FakeIB([pf_item("MES", 5, 7600.0), pf_item("SI", 5000, 31.0)],          # held MES; a FULL SI
            {"MGC": FakeTicker(4300.0), "MCL": FakeTicker(float("nan"), close=70.0)})
got = safe(broker_with(ib).contract_prices, FUTURES, True)
check("a held contract is priced from IB's portfolio mark, without asking for a quote",
      isinstance(got, dict) and got.get("equity_us") == 7600.0 and "MES" not in ib.asked, str(got))
check("a full-size SI in the portfolio is NOT taken for the 1,000oz micro (multiplier differs)",
      isinstance(got, dict) and "silver" not in got, str(got))
check("unheld markets get a quote, live or else delayed / frozen (market data type 4)",
      isinstance(got, dict) and got.get("gold") == 4300.0 and ib.types == [4], str((got, ib.types)))
check("with no market price the quote's close is used", isinstance(got, dict) and got.get("oil") == 70.0, str(got))
check("a market with no price at all is simply absent (stored value later, with a warning)",
      isinstance(got, dict) and "fx_eur" not in got, str(got))
ib2 = FakeIB([pf_item("MES", 5, 7600.0)], {}, fail_quotes=True)
got2 = safe(broker_with(ib2).contract_prices, FUTURES, True)
check("a failed quote request does not crash: the portfolio prices survive",
      isinstance(got2, dict) and got2 == {"equity_us": 7600.0}, str(got2))
check("a dry-run broker asks IB for nothing", FuturesBroker(dry_run=True).contract_prices(FUTURES, True) == {}, "")

# =============================================================================================
print("\nmain(): IB PRICES REACH THE SIZER AND THE GUARD")


class FakeBroker:
    def __init__(self):
        self.calls, self.connected = [], False
        self.ib = NS(sleep=lambda s: None)

    def connect(self):
        self.connected = True
        return True

    def disconnect(self):
        pass

    def margin_cushion(self):
        return (25_000.0, 50_000.0)

    def portfolio_marks(self, futures):
        return []

    def front_expiries(self, futures, micro):
        return {s.market: "20261218" for s in futures}

    def held_positions(self, futures):
        return []

    def contract_prices(self, futures, micro):
        return {"equity_us": REAL_MES / ES.mult(True), "oil": 70.0}     # points: MES 7,600; oil $70 x 100

    def execute(self, batch, specs, micro, limits=None, held=None, values=None):
        self.calls.append((list(batch), values))
        return []


made: list[FakeBroker] = []
runner.make_broker = lambda **kw: made.append(FakeBroker()) or made[-1]
SIZER_VALUES: list = []
_real_ct = runner.compute_targets


def spy_compute_targets(*a, **k):
    SIZER_VALUES.append(k.get("values"))
    return _real_ct(*a, **k)


runner.compute_targets = spy_compute_targets
runner.download_ohlcv = lambda *a, **k: {"adj_close": PX.copy()}
runner._spy_returns = lambda *a: (None, None, None)
runner.send_report = lambda *a, **k: None
runner.push_if_alerts = lambda *a, **k: None
runner.write_equity = lambda *a, **k: None
runner.book_drawdown = lambda *a, **k: (None, None, None, "")
runner.halt_state = lambda root: (rg.HALT_NONE, "")
runner.STATE_FILE = Path(tempfile.mkdtemp()) / "state.json"
WARN: list[str] = []
runner.logging.warning = lambda f, *a: WARN.append(f % a if a else f)
sys.argv = ["run_trend_paper.py", "--live", "--force"]
r = safe(runner.main)
fbk = made[-1] if made else None
calls = fbk.calls if fbk else []
check("main() runs to the end with the fake broker", r is None and fbk is not None and calls, str(r))
vals_seen = [v for _, v in calls]
check("every execute call receives the IB-derived values (MES $38k, oil $7k)",
      bool(vals_seen) and all(v and v.get("equity_us") == REAL_MES and v.get("oil") == 7_000.0
                               for v in vals_seen), str(vals_seen))
check("the sizer receives the same IB-derived values",
      bool(SIZER_VALUES) and all(v and v.get("equity_us") == REAL_MES for v in SIZER_VALUES),
      str(SIZER_VALUES))
sent = [o for b, _ in calls for o in b]
check("no MES is bought: at its real value one contract exceeds the $30k cap",
      not any(o.ib_symbol == "MES" and o.action == "BUY" for o in sent), str(sent))
check("markets IB did not price are named in a WARNING (sized on the stored value)",
      any("contract value" in w and "gold" in w and "stored value" in w for w in WARN), str(WARN)[:300])

print("\n" + "=" * 88)
if _fails:
    print(f"{len(_fails)} FAILURE(S) of {_ran}:")
    for x in _fails:
        print("   " + x)
    sys.exit(1)
print(f"all {_ran} contract-value checks behaved as expected")
