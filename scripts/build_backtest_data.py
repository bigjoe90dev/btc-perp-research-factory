"""
Build cleaned ticks and candle tables for backtesting.

Run:
  .venv/bin/python scripts/build_backtest_data.py
"""
from backtest.config import load_backtest_settings
from backtest.data_pipeline import build_backtest_tables
from shared.config import load_settings
from scripts import init_db


def main():
    print("Initializing DB schema...")
    init_db.main()
    settings = load_settings()
    bt_cfg = load_backtest_settings()
    print("Building cleaned ticks + candles...")
    counts = build_backtest_tables(
        db_path=str(settings.db_path),
        symbol_id=settings.ctrader_symbol_id or 1,
        cfg=bt_cfg,
        pip_size=0.0001,
        force=True,
    )
    print("Backtest tables built:")
    for k, v in counts.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
