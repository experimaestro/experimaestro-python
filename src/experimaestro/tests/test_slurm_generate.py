"""Tests for SLURM CLI parsing functions."""

from experimaestro.launchers.slurm.cli import (
    parse_gres,
    parse_time_to_seconds,
)


# =============================================================================
# Unit tests for parsing functions
# =============================================================================


class TestParseTimeToSeconds:
    def test_days_hours_minutes_seconds(self):
        assert parse_time_to_seconds("4-04:00:00") == 4 * 86400 + 4 * 3600

    def test_hours_minutes_seconds(self):
        assert parse_time_to_seconds("20:00:00") == 20 * 3600

    def test_minutes_seconds(self):
        assert parse_time_to_seconds("30:00") == 30 * 60

    def test_seconds_only(self):
        assert parse_time_to_seconds("120") == 120

    def test_unlimited(self):
        assert parse_time_to_seconds("UNLIMITED") is None

    def test_none(self):
        assert parse_time_to_seconds(None) is None

    def test_two_hours(self):
        assert parse_time_to_seconds("02:00:00") == 2 * 3600


class TestParseGres:
    def test_gpu_count_only(self):
        count, gpu_type = parse_gres("gpu:4(S:0-1)")
        assert count == 4
        assert gpu_type is None

    def test_gpu_with_type(self):
        count, gpu_type = parse_gres("gpu:tesla:4(S:0-1)")
        assert count == 4
        assert gpu_type == "tesla"

    def test_gpu_8(self):
        count, gpu_type = parse_gres("gpu:8(S:0-1)")
        assert count == 8
        assert gpu_type is None

    def test_null(self):
        count, gpu_type = parse_gres("(null)")
        assert count == 0
        assert gpu_type is None

    def test_empty(self):
        count, gpu_type = parse_gres("")
        assert count == 0
        assert gpu_type is None
