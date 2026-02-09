import unittest
from datetime import datetime, timezone

from backtest.session import classify_session, default_session_rules


class TestSessionClassification(unittest.TestCase):
    def test_london_session(self):
        # 2026-01-15 08:30 UTC == 08:30 London (winter time)
        ts = datetime(2026, 1, 15, 8, 30, tzinfo=timezone.utc)
        sess = classify_session(ts, default_session_rules())
        self.assertEqual(sess, "London")

    def test_newyork_session(self):
        # 2026-01-15 14:00 UTC == 09:00 New York (EST)
        ts = datetime(2026, 1, 15, 14, 0, tzinfo=timezone.utc)
        sess = classify_session(ts, default_session_rules())
        self.assertEqual(sess, "NewYork")


if __name__ == "__main__":
    unittest.main()
