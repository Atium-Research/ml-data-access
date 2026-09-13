"""Read side of the malatium data store.

    import ml_data_access

    db = ml_data_access.connect()
    reference_df = ml_data_access.load_reference_returns(db, start, end)
    chain_df = ml_data_access.load_option_greeks(db, "AAPL", start, end)

One `load_*` per table `ml-data-pipelines` writes, every one taking the
database and an inclusive `[start, end]` window and returning a collected
DataFrame. Nothing is screened, joined or derived: the store already holds
canonical tables, and the two screens a multi-year study needs are named
functions, `usable_symbol_years` and `in_universe`.
"""

from ml_data_access.loaders import (
    in_universe,
    load_calendar,
    load_corporate_actions,
    load_earnings,
    load_factor_covariances,
    load_factor_loadings,
    load_factor_returns,
    load_forecast,
    load_idio_vol,
    load_index_greeks,
    load_indices,
    load_open_interest,
    load_option_greeks,
    load_rates,
    load_realized_vol,
    load_reference_returns,
    load_sectors,
    load_sessions,
    load_signals,
    load_stock_features,
    load_surface,
    load_symbology_check,
    load_underlying,
    load_universe,
    load_yields,
    usable_symbol_years,
)
from ml_data_access.store import (
    available_symbols,
    available_years,
    connect,
    describe,
    scan,
)

__all__ = [
    "available_symbols",
    "available_years",
    "connect",
    "describe",
    "in_universe",
    "load_calendar",
    "load_corporate_actions",
    "load_earnings",
    "load_factor_covariances",
    "load_factor_loadings",
    "load_factor_returns",
    "load_forecast",
    "load_idio_vol",
    "load_index_greeks",
    "load_indices",
    "load_open_interest",
    "load_option_greeks",
    "load_rates",
    "load_realized_vol",
    "load_reference_returns",
    "load_sectors",
    "load_sessions",
    "load_signals",
    "load_stock_features",
    "load_surface",
    "load_symbology_check",
    "load_underlying",
    "load_universe",
    "load_yields",
    "scan",
    "usable_symbol_years",
]
