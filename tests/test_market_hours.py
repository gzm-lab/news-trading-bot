"""Tests for market hours helpers (_market_phase, _seconds_until_active)."""

from datetime import datetime
from zoneinfo import ZoneInfo

from src.main import _market_phase, _seconds_until_active, _seconds_until_market

_ET = ZoneInfo("America/New_York")


class TestMarketPhase:
    def test_premarket_start(self):
        t = datetime(2026, 4, 6, 4, 0, tzinfo=_ET)   # Mon 04:00
        assert _market_phase(t) == "premarket"

    def test_premarket_mid(self):
        t = datetime(2026, 4, 6, 7, 30, tzinfo=_ET)  # Mon 07:30
        assert _market_phase(t) == "premarket"

    def test_premarket_ends_at_open(self):
        t = datetime(2026, 4, 6, 9, 29, tzinfo=_ET)  # Mon 09:29 — still pre
        assert _market_phase(t) == "premarket"

    def test_open_at_930(self):
        t = datetime(2026, 4, 6, 9, 30, tzinfo=_ET)  # Mon 09:30
        assert _market_phase(t) == "open"

    def test_open_midday(self):
        t = datetime(2026, 4, 6, 12, 0, tzinfo=_ET)
        assert _market_phase(t) == "open"

    def test_open_at_close(self):
        # 16:00 is closed (>= close_t)
        t = datetime(2026, 4, 6, 16, 0, tzinfo=_ET)
        assert _market_phase(t) == "closed"

    def test_closed_before_premarket(self):
        t = datetime(2026, 4, 6, 3, 59, tzinfo=_ET)  # Mon 03:59
        assert _market_phase(t) == "closed"

    def test_closed_after_hours(self):
        t = datetime(2026, 4, 6, 18, 0, tzinfo=_ET)
        assert _market_phase(t) == "closed"

    def test_closed_saturday(self):
        t = datetime(2026, 4, 4, 10, 0, tzinfo=_ET)
        assert _market_phase(t) == "closed"

    def test_closed_sunday(self):
        t = datetime(2026, 4, 5, 10, 0, tzinfo=_ET)
        assert _market_phase(t) == "closed"


class TestSecondsUntilActive:
    def test_premarket_returns_zero(self):
        t = datetime(2026, 4, 6, 6, 0, tzinfo=_ET)   # Mon 06:00
        assert _seconds_until_active(t) == 0

    def test_open_returns_zero(self):
        t = datetime(2026, 4, 6, 10, 0, tzinfo=_ET)  # Mon 10:00
        assert _seconds_until_active(t) == 0

    def test_saturday_waits_until_monday_4am(self):
        t = datetime(2026, 4, 4, 12, 0, tzinfo=_ET)  # Sat 12:00
        wait = _seconds_until_active(t)
        # Sat 12:00 → Mon 04:00 = 40h = 144000s
        assert 143800 < wait < 144200

    def test_sunday_waits_until_monday_4am(self):
        t = datetime(2026, 4, 5, 20, 0, tzinfo=_ET)  # Sun 20:00
        wait = _seconds_until_active(t)
        # Sun 20:00 → Mon 04:00 = 8h = 28800s
        assert 28600 < wait < 29000

    def test_friday_after_close_waits_monday_4am(self):
        t = datetime(2026, 4, 3, 17, 0, tzinfo=_ET)  # Fri 17:00
        wait = _seconds_until_active(t)
        # Fri 17:00 → Mon 04:00 = 59h = 212400s
        assert 212200 < wait < 212600

    def test_weekday_before_premarket_waits_today(self):
        t = datetime(2026, 4, 6, 2, 0, tzinfo=_ET)   # Mon 02:00
        wait = _seconds_until_active(t)
        # Mon 02:00 → Mon 04:00 = 2h = 7200s
        assert 7000 < wait < 7400

    def test_weekday_after_close_waits_next_day(self):
        t = datetime(2026, 4, 6, 17, 0, tzinfo=_ET)  # Mon 17:00
        wait = _seconds_until_active(t)
        # Mon 17:00 → Tue 04:00 = 11h = 39600s
        assert 39400 < wait < 39800


class TestBackwardCompat:
    """_seconds_until_market alias still works for existing tests."""

    def test_weekday_during_hours_returns_zero(self):
        t = datetime(2026, 4, 6, 10, 0, tzinfo=_ET)
        assert _seconds_until_market(t) == 0

    def test_premarket_returns_zero(self):
        t = datetime(2026, 4, 6, 5, 0, tzinfo=_ET)
        assert _seconds_until_market(t) == 0

    def test_saturday_returns_nonzero(self):
        t = datetime(2026, 4, 4, 12, 0, tzinfo=_ET)
        assert _seconds_until_market(t) > 0
