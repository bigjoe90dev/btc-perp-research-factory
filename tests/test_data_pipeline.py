import unittest
import pandas as pd

from backtest.config import BacktestSettings
from backtest.data_pipeline import clean_ticks, build_candles


class TestDataPipeline(unittest.TestCase):
    def test_clean_and_candles(self):
        df = pd.DataFrame({
            "ts_utc": [
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:00:01+00:00",
                "2026-01-01T00:00:02+00:00",
            ],
            "bid": [1.1000, 1.1001, 1.1002],
            "ask": [1.1002, 1.1003, 1.1004],
        })
        cfg = BacktestSettings()
        cleaned = clean_ticks(df, cfg, pip_size=0.0001)
        self.assertFalse(cleaned.empty)
        candles = build_candles(cleaned, "1min", pip_size=0.0001)
        self.assertIn("mid_c", candles.columns)
        self.assertIn("spread_c_pips", candles.columns)


if __name__ == "__main__":
    unittest.main()
