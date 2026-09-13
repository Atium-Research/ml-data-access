import datetime as dt

import bear_lake as bl
import polars as pl
import pytest

import ml_data_access


def business_days(start: dt.date, n: int) -> list[dt.date]:
    days, day = [], start
    while len(days) < n:
        if day.weekday() < 5:
            days.append(day)
        day += dt.timedelta(days=1)
    return days


@pytest.fixture
def db(tmp_path):
    """A small store in the layout ml-data-pipelines writes."""
    database = bl.connect(str(tmp_path / "store"))
    dates = business_days(dt.date(2024, 12, 27), 6)  # straddles the year end
    database.create("calendar", {"date": pl.Date}, None, ["date"])
    database.insert("calendar", pl.DataFrame({"date": dates}))

    universe_schema = {"date": pl.Date, "ticker": pl.String, "symbol": pl.String, "year": pl.Int32}
    database.create("universe", universe_schema, ["year"], ["date", "ticker"])
    universe_df = pl.DataFrame(
        {
            "date": dates * 2,
            "ticker": ["AAPL"] * 6 + ["BRK.B"] * 6,
            "symbol": ["AAPL"] * 6 + ["BRKB"] * 6,
        }
    ).with_columns(pl.col("date").dt.year().cast(pl.Int32).alias("year"))
    database.insert("universe", universe_df)

    chain_schema = {
        "date": pl.Date,
        "symbol": pl.String,
        "expiration": pl.Date,
        "strike": pl.Float64,
        "right": pl.String,
        "mid": pl.Float64,
        "year": pl.Int32,
    }
    database.create(
        "option_greeks",
        chain_schema,
        ["year", "symbol"],
        ["date", "symbol", "expiration", "strike", "right"],
    )
    rows = []
    for symbol in ("AAPL", "BRKB"):
        for date_ in dates:
            rows.append(
                {
                    "date": date_,
                    "symbol": symbol,
                    "expiration": dt.date(2025, 3, 21),
                    "strike": 100.0,
                    "right": "C",
                    "mid": 1.0,
                }
            )
    database.insert(
        "option_greeks",
        pl.DataFrame(rows).with_columns(pl.col("date").dt.year().cast(pl.Int32).alias("year")),
    )

    check_schema = {"year": pl.Int32, "symbol": pl.String, "status": pl.String}
    database.create("symbology_check", check_schema, None, ["year", "symbol"])
    database.insert(
        "symbology_check",
        pl.DataFrame(
            {
                "year": [2024, 2024, 2025],
                "symbol": ["AAPL", "FI", "AAPL"],
                "status": ["ok", "wrong_instrument", "thin_overlap"],
            }
        ).cast({"year": pl.Int32}),
    )

    signals_schema = {
        "date": pl.Date,
        "symbol": pl.String,
        "signal": pl.String,
        "score": pl.Float64,
        "year": pl.Int32,
    }
    database.create("signals", signals_schema, ["year"], ["date", "symbol", "signal"])
    database.insert(
        "signals",
        pl.DataFrame(
            {
                "date": dates * 2,
                "symbol": ["AAPL"] * 12,
                "signal": ["vrp"] * 6 + ["momentum"] * 6,
                "score": [0.5] * 6 + [-0.5] * 6,
            }
        ).with_columns(pl.col("date").dt.year().cast(pl.Int32).alias("year")),
    )
    return database


def test_windowed_loaders_respect_the_window(db):
    sessions = ml_data_access.load_sessions(db, dt.date(2025, 1, 1), None)
    assert all(day >= dt.date(2025, 1, 1) for day in sessions) and len(sessions) == 3
    universe_df = ml_data_access.load_universe(db, dt.date(2024, 12, 30), dt.date(2024, 12, 31))
    assert universe_df["date"].n_unique() == 2 and set(universe_df["symbol"]) == {"AAPL", "BRKB"}
    assert ml_data_access.available_years("universe") == [2024, 2025]


def test_chain_loader_opens_one_symbol_and_filters_the_window(db):
    chain_df = ml_data_access.load_option_greeks(
        db, "aapl", dt.date(2025, 1, 1), dt.date(2025, 1, 2)
    )
    assert set(chain_df["symbol"]) == {"AAPL"}
    assert chain_df["date"].min() >= dt.date(2025, 1, 1)
    assert ml_data_access.available_symbols("option_greeks", 2025) == ["AAPL", "BRKB"]
    with pytest.raises(FileNotFoundError):
        ml_data_access.load_option_greeks(db, "MSFT")
    with pytest.raises(FileNotFoundError):
        ml_data_access.load_open_interest(db, "AAPL")


def test_screens(db):
    usable_df = ml_data_access.usable_symbol_years(db)
    assert usable_df.rows() == [("AAPL", 2024), ("AAPL", 2025)]  # thin_overlap kept, wrong dropped
    panel_df = pl.DataFrame(
        {"date": [dt.date(2024, 12, 30), dt.date(2024, 12, 30)], "symbol": ["AAPL", "MSFT"]}
    )
    assert ml_data_access.in_universe(db, panel_df)["symbol"].to_list() == ["AAPL"]


def test_signals_loader_picks_one_signal(db):
    scores_df = ml_data_access.load_signals(db, "momentum")
    assert scores_df.columns == ["date", "symbol", "score"]
    assert (scores_df["score"] == -0.5).all() and scores_df.height == 6


def test_describe_lists_tables(db):
    text = ml_data_access.describe()
    assert "option_greeks" in text and "calendar" in text
