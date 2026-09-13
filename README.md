# ml-data

Read side of the malatium data store. One `load_*` per table that [ml-data-pipelines](https://github.com/Atium-Research/ml-data-pipelines) writes, over a [bear-lake](https://github.com/andrewhall1124/bear-lake) database, in the shape of `at-research`'s data helpers.

```bash
pip install ml-data
```

```python
import datetime as dt

import ml_data

db = ml_data.connect()  # ML_DATA_STORE, or connect(path)
start, end = dt.date(2018, 7, 2), dt.date(2025, 6, 30)

reference_df = ml_data.load_reference_returns(db, start, end)
scores_df = ml_data.load_signals(db, "vrp", start, end)
loadings_df = ml_data.load_factor_loadings(db, start, end)
chain_df = ml_data.load_option_greeks(db, "AAPL", dt.date(2025, 1, 1), dt.date(2025, 3, 31))
```

Every loader is `load_x(db, start=None, end=None)` and returns a collected DataFrame for the inclusive window; the three per-symbol chain loaders take the symbol first. Nothing is screened, joined or derived: the store holds canonical tables (`symbol` is the option root, `right` is `C`/`P`, `iv` is null where the vendor's inversion failed, `vega` is per vol point).

The frames slot straight into malatium:

```python
from malatium.providers import PanelProvider, TradingCalendar

calendar = TradingCalendar(ml_data.load_sessions(db, start, end))
reference = PanelProvider(reference_df)
scores = PanelProvider(scores_df)
```

## Loaders

| loader | table |
| --- | --- |
| `load_calendar`, `load_sessions` | exchange sessions, as a frame or a list of dates |
| `load_universe` | point-in-time S&P 500 membership, `ticker` and `symbol` |
| `load_sectors` | GICS sector snapshot |
| `load_indices`, `load_yields`, `load_rates` | index levels (2024 on), the CBOE yield curve, SOFR |
| `load_corporate_actions`, `load_earnings` | splits and dividends; announcement dates with a session |
| `load_underlying` | EOD stock OHLCV, 2023-06 on |
| `load_option_greeks(db, symbol, ...)`, `load_index_greeks`, `load_open_interest` | one name's chain, index chain, open interest |
| `load_symbology_check`, `usable_symbol_years` | which symbol-years are the right company |
| `load_reference_returns` | the reference straddle's per-vega P&L, `SPX` included |
| `load_factor_returns`, `load_factor_loadings`, `load_factor_covariances`, `load_idio_vol` | the vol risk model |
| `load_surface`, `load_realized_vol`, `load_forecast`, `load_stock_features` | the derived panels |
| `load_signals(db, name, ...)` | one signal's `(date, symbol, score)` |
| `in_universe` | semi-join a `(date, symbol)` panel to membership |

`ml_data.describe()` lists what the store holds. `ml_data.scan(name, years, symbols)` is the lazy scan under every loader; it opens only the partitions asked for, using bear-lake's `<table>/<year>/<symbol>.parquet` layout.

## Things to know before trusting a number

- **Eighteen symbol-years are another company's chain.** Semi-join against `usable_symbol_years(db)` for any multi-year study.
- **Open interest is one day stale by construction** and joins on the same `date`.
- **`iv` is null, not wrong**, on the ~3% of contract-days that failed to invert.
- **Repaired index sessions** (mostly 2020-21) carry null gamma and a 15:59 underlying.

## Development

```bash
uv sync
uv run pytest        # builds a synthetic store in a temp dir
uv run ruff check . && uv run ruff format .
```
