import unittest
import pandas as pd

from dashboard import app as dashboard_app


class TestChartDataCleaning(unittest.TestCase):
    def test_mixed_ts_and_scaling(self):
        df = pd.DataFrame({
            "ts_utc": [
                "2026-01-17T20:14:27.048792+00:00",
                "2026-01-17T20:15:27+00:00",
            ],
            "bid": [115980.0, 1.0425],
            "ask": [115994.0, 1.0427],
        })
        out = dashboard_app._clean_quotes_for_chart(df, resample=None)
        self.assertEqual(len(out), 2)
        self.assertTrue((out["open"] < 10).all())

    def test_outlier_filter(self):
        df = pd.DataFrame({
            "ts_utc": [
                "2026-01-17T20:14:27+00:00",
                "2026-01-17T20:15:27+00:00",
                "2026-01-17T20:16:27+00:00",
            ],
            "bid": [1.0425, 1.0427, 150.0],
            "ask": [1.0427, 1.0429, 150.2],
        })
        out = dashboard_app._clean_quotes_for_chart(df, resample=None)
        # Outlier should be removed by median-based filter
        self.assertEqual(len(out), 2)


if __name__ == "__main__":
    unittest.main()
