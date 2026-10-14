"""Mutation-test `scripts/test_contract_values.py`: contract values from IB prices (Appendix E #23).

Seeds real faults into a TEMP COPY of the repo (the real files are never edited) and demands the
suite catches every one. Engine: _mutate_repo_core.py.

Run: python3 scripts/mutate_contract_values.py
"""
from __future__ import annotations

import sys

from _mutate_repo_core import run

E = "trend_overlay/execution.py"
R = "scripts/run_trend_paper.py"
MUTATIONS = [
    (E, "        rows.append((s.market, sg, vv_raw, expo, contract_value(s, cfg.use_micro, values),",
     "        rows.append((s.market, sg, vv_raw, expo, s.notional(cfg.use_micro),",
     "the sizer ignores the values (the original bug)"),
    (E, "                notl = contract_value(spec, use_micro, values) if spec else 0.0",
     "                notl = spec.notional(use_micro) if spec else 0.0",
     "the order guard ignores the values (sizer and guard disagree)"),
    (E, "    if v is not None and np.isfinite(v) and v > 0:\n        return float(v)\n    return spec.notional(use_micro)",
     "    return float(v) if v is not None else spec.notional(use_micro)",
     "a NaN value is used as the contract value"),
    (E, "        if not stored / VALUE_SANITY <= v <= stored * VALUE_SANITY:",
     "        if False:", "no sanity band: a 100x unit error is used for sizing"),
    (E, "        if not stored / VALUE_SANITY <= v <= stored * VALUE_SANITY:",
     "        if not v <= stored * VALUE_SANITY:", "the sanity band has no lower side"),
    (E, "        if v is None or not np.isfinite(v) or v <= 0:\n            notes.append(",
     "        if v is None:\n            notes.append(", "NaN or zero values pass the check"),
    (E, "                    key = (c.symbol, int(float(c.multiplier or 0)))",
     "                    key = next((k for k in want if k[0] == c.symbol), (c.symbol, 0))",
     "a portfolio contract is matched by symbol only (full SI taken for the micro)"),
    (E, "                    out.setdefault(want[key], float(p))",
     "                    pass", "held contracts are never priced from the portfolio"),
    (E, "                self.ib.reqMarketDataType(4)\n", "",
     "no market data type set: quotes need a live subscription"),
    (E, "                    if p is None or p != p or p <= 0:\n                        p = t.close\n", "",
     "the quote's close is not used when there is no market price"),
    (E, "            except Exception as e:  # noqa: BLE001\n                logging.warning(\"contract prices: quote request failed: %s\", e)",
     "            except ImportError as e:\n                logging.warning(\"contract prices: quote request failed: %s\", e)",
     "a failed quote request crashes the run"),
    (R, "                tgt = compute_targets(px, cfg, held=held_by_mkt, values=values)",
     "                tgt = compute_targets(px, cfg, held=held_by_mkt)",
     "main() does not pass the values to the sizer"),
    (R, "                fills = broker.execute(batch, BY_MARKET, cfg.use_micro, limits=lim, held=held,\n                                       values=values)",
     "                fills = broker.execute(batch, BY_MARKET, cfg.use_micro, limits=lim, held=held)",
     "main() does not pass the values to the order guard"),
    (R, "                for _n in _vnotes:\n                    logging.warning(\"contract value: %s\", _n)\n", "",
     "markets sized on the stored value are not reported"),
    ("trend_overlay/contracts.py",
     "        if use_micro and self.micro_symbol and self.micro_notice_buffer_days is not None:\n            return self.micro_notice_buffer_days\n",
     "", "the micro's own buffer is ignored (1OZ held to within 30 days like GC)"),
    ("trend_overlay/contracts.py", "               notice_buffer_days=30, micro_notice_buffer_days=5),",
     "               notice_buffer_days=30, micro_notice_buffer_days=30),", "1OZ given GC's 30-day buffer"),
    ("trend_overlay/contracts.py", '"GC", "COMEX", "USD", 100,   411_000, "1OZ", 1,    4_100,',
     '"GC", "COMEX", "USD", 100,   411_000, "MGC", 10,   41_100,', "gold back on the 10oz micro"),
    (E, "        buf = spec.buffer_days(h.ib_symbol == spec.micro_symbol and spec.micro_symbol != spec.symbol)",
     "        buf = spec.notice_buffer_days", "the forced close ignores the held contract's own buffer"),
    (E, "            cutoff = pd.Timestamp.today() + pd.Timedelta(days=spec.buffer_days(use_micro))",
     "            cutoff = pd.Timestamp.today() + pd.Timedelta(days=spec.notice_buffer_days)",
     "the front contract is chosen with the standard contract's buffer"),
]

if __name__ == "__main__":
    sys.exit(run("test_contract_values.py", MUTATIONS))
