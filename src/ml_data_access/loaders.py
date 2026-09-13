"""One loader per table. Every loader is `load_x(db, start=None, end=None)`.

The per-symbol chain loaders take the symbol first, since it selects the
partition rather than filtering rows. `db` is the connected bear-lake
database; it is passed for the same reason at-research passes it, so a
study can hold two stores open.
"""

import datetime as dt

import bear_lake as bl
import polars as pl

from ml_data_access.store import in_window, scan, years_in


def load_window(
    db: bl.Database, name: str, start: dt.date | None, end: dt.date | None, sort: list[str]
) -> pl.DataFrame:
    frame = scan(name, years=years_in(start, end, name))
    return db.query(in_window(frame, start, end).sort(sort))


# --- raw ---------------------------------------------------------------------


def load_calendar(
    db: bl.Database, start: dt.date | None = None, end: dt.date | None = None
) -> pl.DataFrame:
    """Exchange sessions, one `date` per row."""
    return load_window(db, "calendar", start, end, ["date"])


def load_sessions(
    db: bl.Database, start: dt.date | None = None, end: dt.date | None = None
) -> list[dt.date]:
    """The calendar as a list of dates, the shape `malatium.providers.TradingCalendar` takes."""
    return load_calendar(db, start, end)["date"].to_list()


def load_universe(
    db: bl.Database, start: dt.date | None = None, end: dt.date | None = None
) -> pl.DataFrame:
    """Point-in-time membership: `date`, `ticker` (Wikipedia), `symbol` (option root)."""
    return load_window(db, "universe", start, end, ["date", "ticker"])


def load_sectors(db: bl.Database) -> pl.DataFrame:
    """`ticker`, `symbol`, `sector`, `sub_industry`; a snapshot of today's constituents."""
    return db.query(scan("sectors").sort("ticker"))


def load_indices(
    db: bl.Database, start: dt.date | None = None, end: dt.date | None = None
) -> pl.DataFrame:
    """EOD levels for SPX, RUT, OEX, XSP and the VIX complex, 2024 on."""
    return load_window(db, "indices", start, end, ["date", "symbol"])


def load_yields(
    db: bl.Database, start: dt.date | None = None, end: dt.date | None = None
) -> pl.DataFrame:
    """CBOE treasury yield indices (`13w`, `5y`, `10y`, `30y`) as decimals."""
    return load_window(db, "yields", start, end, ["date", "tenor"])


def load_rates(
    db: bl.Database, start: dt.date | None = None, end: dt.date | None = None
) -> pl.DataFrame:
    """Overnight SOFR as a decimal, 2024 on."""
    return load_window(db, "rates", start, end, ["date", "symbol"])


def load_corporate_actions(
    db: bl.Database, start: dt.date | None = None, end: dt.date | None = None
) -> pl.DataFrame:
    """Splits (ex-date ratios) and dividends, one row per (symbol, date, action)."""
    return load_window(db, "corporate_actions", start, end, ["symbol", "date", "action"])


def load_earnings(
    db: bl.Database, start: dt.date | None = None, end: dt.date | None = None
) -> pl.DataFrame:
    """Announcement dates with a `bmo` / `amc` / `unknown` session."""
    return load_window(db, "earnings", start, end, ["symbol", "date"])


def load_underlying(
    db: bl.Database, start: dt.date | None = None, end: dt.date | None = None
) -> pl.DataFrame:
    """EOD stock OHLCV, 2023-06 on, `symbol` in option-root spelling."""
    return load_window(db, "underlying", start, end, ["date", "symbol"])


def load_chain(
    db: bl.Database, name: str, symbol: str, start: dt.date | None, end: dt.date | None
) -> pl.DataFrame:
    frame = scan(name, years=years_in(start, end, name), symbols=[symbol.upper()])
    return db.query(in_window(frame, start, end).sort("date", "expiration", "strike", "right"))


def load_option_greeks(
    db: bl.Database, symbol: str, start: dt.date | None = None, end: dt.date | None = None
) -> pl.DataFrame:
    """One name's EOD chain: quotes, `iv` (null where the inversion failed), greeks with
    `vega` per vol point, and `underlying`, the spot the greeks were struck against."""
    return load_chain(db, "option_greeks", symbol, start, end)


def load_index_greeks(
    db: bl.Database, symbol: str, start: dt.date | None = None, end: dt.date | None = None
) -> pl.DataFrame:
    """The same for an index root (SPX, SPXW, XSP, VIX); repaired sessions carry null gamma."""
    return load_chain(db, "index_greeks", symbol, start, end)


def load_open_interest(
    db: bl.Database, symbol: str, start: dt.date | None = None, end: dt.date | None = None
) -> pl.DataFrame:
    """One name's EOD open interest; stamped pre-open, so it joins the same session's chain."""
    return load_chain(db, "open_interest", symbol, start, end)


def load_symbology_check(db: bl.Database) -> pl.DataFrame:
    """Per symbol-year: is the stored chain the company the universe names?"""
    return db.query(scan("symbology_check").sort("year", "symbol"))


# --- derived -------------------------------------------------------------------


def load_reference_returns(
    db: bl.Database,
    start: dt.date | None = None,
    end: dt.date | None = None,
    symbols: list[str] | None = None,
) -> pl.DataFrame:
    """The reference straddle's per-vega P&L per (date, symbol); `SPX` is the market."""
    frame = scan(
        "reference_returns", years=years_in(start, end, "reference_returns"), symbols=symbols
    )
    return db.query(in_window(frame, start, end).sort("date", "symbol"))


def load_factor_returns(
    db: bl.Database, start: dt.date | None = None, end: dt.date | None = None
) -> pl.DataFrame:
    """`(date, factor, ret)`: `market` and one GICS sector factor each."""
    return load_window(db, "factor_returns", start, end, ["date", "factor"])


def load_factor_loadings(
    db: bl.Database, start: dt.date | None = None, end: dt.date | None = None
) -> pl.DataFrame:
    """`(date, symbol, factor, loading)`, trailing 250 sessions."""
    return load_window(db, "factor_loadings", start, end, ["date", "symbol", "factor"])


def load_factor_covariances(
    db: bl.Database, start: dt.date | None = None, end: dt.date | None = None
) -> pl.DataFrame:
    """`(date, factor_1, factor_2, covariance)`, Ledoit-Wolf shrunk."""
    return load_window(db, "factor_covariances", start, end, ["date", "factor_1", "factor_2"])


def load_idio_vol(
    db: bl.Database, start: dt.date | None = None, end: dt.date | None = None
) -> pl.DataFrame:
    """`(date, symbol, idio_vol)`, daily residual std per dollar of vega."""
    return load_window(db, "idio_vol", start, end, ["date", "symbol"])


def load_surface(
    db: bl.Database, start: dt.date | None = None, end: dt.date | None = None
) -> pl.DataFrame:
    """Constant-maturity ATM IV per (date, symbol): `iv30`, `iv60`, `iv90`."""
    return load_window(db, "surface", start, end, ["date", "symbol"])


def load_realized_vol(
    db: bl.Database, start: dt.date | None = None, end: dt.date | None = None
) -> pl.DataFrame:
    """Close-to-close realized vol: trailing `rv_1`, `rv_5`, `rv_22` and the forward `rv_fwd`."""
    return load_window(db, "realized_vol", start, end, ["date", "symbol"])


def load_forecast(
    db: bl.Database, start: dt.date | None = None, end: dt.date | None = None
) -> pl.DataFrame:
    """The HAR forecast of the forward 60-session vol, `rv_fcst`."""
    return load_window(db, "forecast", start, end, ["date", "symbol"])


def load_stock_features(
    db: bl.Database, start: dt.date | None = None, end: dt.date | None = None
) -> pl.DataFrame:
    """`ret`, `beta`, `idio_vol`, `gap_freq` per (date, symbol)."""
    return load_window(db, "stock_features", start, end, ["date", "symbol"])


def load_signals(
    db: bl.Database,
    signal: str,
    start: dt.date | None = None,
    end: dt.date | None = None,
) -> pl.DataFrame:
    """`(date, symbol, score)` for one named signal (`vrp`, `iv_zscore`, `momentum`)."""
    frame = scan("signals", years=years_in(start, end, "signals")).filter(
        pl.col("signal") == signal
    )
    return db.query(
        in_window(frame, start, end).select("date", "symbol", "score").sort("date", "symbol")
    )


# --- screens -------------------------------------------------------------------


def usable_symbol_years(db: bl.Database) -> pl.DataFrame:
    """`(symbol, year)` pairs whose chain is not known to be another company's.

    Only `wrong_instrument` is excluded. `thin_overlap` means the check could
    not run, and it falls on names later delisted or acquired, so excluding
    it would be a survivorship filter. Semi-join a panel against this.
    """
    return db.query(
        scan("symbology_check")
        .filter(pl.col("status") != "wrong_instrument")
        .select("symbol", "year")
        .unique()
        .sort("symbol", "year")
    )


def in_universe(db: bl.Database, panel_df: pl.DataFrame) -> pl.DataFrame:
    """Keep the (date, symbol) rows that were index members that day."""
    start, end = panel_df["date"].min(), panel_df["date"].max()
    members_df = load_universe(db, start, end).select("date", "symbol").unique()
    return panel_df.join(members_df, on=["date", "symbol"], how="semi")
