import unittest

from backtest.eurusd_strategies import PivotDetector


class TestPivotDetector(unittest.TestCase):
    def test_pivot_detection(self):
        pd = PivotDetector(wing=2)
        highs = [1, 2, 5, 3, 2, 4, 6, 3, 2]
        lows = [0.5, 1, 2, 1.5, 1, 1.2, 2, 1.4, 1.0]
        pivots_high = []
        pivots_low = []
        for h, l in zip(highs, lows):
            ph, pl = pd.update(h, l)
            if ph:
                pivots_high.append(ph)
            if pl:
                pivots_low.append(pl)
        self.assertTrue(len(pivots_high) >= 1)
        self.assertTrue(len(pivots_low) >= 1)


if __name__ == "__main__":
    unittest.main()
