"""Tests for market hours helper."""

from datetime import datetime
from zoneinfo import ZoneInfo

from src.main import _seconds_until_market

_ET = ZoneInfo("America/New_York")


class TestSecondsUntilMarket:
    def test_weekday_during_hours_returns_zero(self):
        # Monday 10:00 ET
        t = datetime(2026, 4, 6, 10, 0, tzinfo=_ET)
        assert _seconds_until_market(t) == 0

    def test_weekday_within_buffer_returns_zero(self):
        # Monday 09:16 ET (within 15-min pre-market buffer)
        t = datetime(2026, 4, 6, 9, 16, tzinfo=_ET)
        assert _seconds_until_market(t) == 0

    def test_weekday_before_buffer_waits(self):
        # Monday 08:00 ET → should wait until 09:15 ET
        t = datetime(2026, 4, 6, 8, 0, tzinfo=_ET)
        wait = _seconds_until_market(t)
        assert 4400 < wait < 4600  # ~75 min = 4500s

    def test_weekday_after_close_waits_next_day(self):
        # Monday 17:00 ET → should wait until Tue 09:15 ET
        t = datetime(2026, 4, 6, 17, 0, tzinfo=_ET)
        wait = _seconds_until_market(t)
        assert 57000 < wait < 59000  # ~16.25h

    def test_friday_after_close_waits_monday(self):
        # Friday 17:00 ET → should wait until Mon 09:15 ET
        t = datetime(2026, 4, 3, 17, 0, tzinfo=_ET)
        wait = _seconds_until_market(t)
        assert 230000 < wait < 232000  # ~64.25h

    def test_saturday_waits_monday(self):
        t = datetime(2026, 4, 4, 12, 0, tzinfo=_ET)
        wait = _seconds_until_market(t)
        # Sat 12:00 → Mon 09:15 = 45.25h
        assert 162000 < wait < 164000

    def test_sunday_waits_monday(self):
        t = datetime(2026, 4, 5, 12, 0, tzinfo=_ET)
        wait = _seconds_until_market(t)
        # Sun 12:00 → Mon 09:15 = 21.25h
        assert 75000 < wait < 77000

    def test_at_close_exactly_returns_zero(self):
        # Monday 16:00 ET — market closes at 16:00, should still be in window
        t = datetime(2026, 4, 6, 16, 0, tzinfo=_ET)
        assert _seconds_until_market(t) == 0

    def test_one_minute_past_close_waits(self):
        # Monday 16:01 ET
        t = datetime(2026, 4, 6, 16, 1, tzinfo=_ET)
        assert _seconds_until_market(t) > 0
