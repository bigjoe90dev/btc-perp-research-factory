import unittest
from datetime import datetime, timezone, timedelta

from shared.glsb_strategy import CandleAggregator, GLSBStrategy, GLSBConfig, Bar


class TestGLSBStrategy(unittest.TestCase):
    def test_candle_aggregator_rollover(self):
        agg = CandleAggregator(5)
        t0 = datetime(2026, 2, 6, 8, 0, tzinfo=timezone.utc)
        self.assertIsNone(agg.update(t0, 1.0))
        self.assertIsNone(agg.update(t0 + timedelta(minutes=4), 1.2))
        closed = agg.update(t0 + timedelta(minutes=5), 1.1)
        self.assertIsNotNone(closed)
        self.assertAlmostEqual(closed.open, 1.0)
        self.assertAlmostEqual(closed.high, 1.2)
        self.assertAlmostEqual(closed.low, 1.0)
        self.assertAlmostEqual(closed.close, 1.2)

    def test_atr_computation(self):
        cfg = GLSBConfig()
        strat = GLSBStrategy(cfg)
        start = datetime(2026, 2, 6, 0, 0, tzinfo=timezone.utc)
        # Build 16 bars with constant range of 1.0
        for i in range(16):
            b = Bar(
                start=start + timedelta(minutes=5 * i),
                open=10.0,
                high=11.0,
                low=10.0,
                close=10.5,
            )
            strat.bars_5m.append(b)
        atr = strat._atr()
        self.assertIsNotNone(atr)
        self.assertAlmostEqual(atr, 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
