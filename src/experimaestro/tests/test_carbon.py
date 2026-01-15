"""Tests for carbon tracking functionality."""

import json
import tempfile
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import patch


from experimaestro.carbon.base import CarbonMetrics, NullCarbonTracker
from experimaestro.carbon.utils import (
    format_co2,
    format_co2_kg,
    format_energy,
    format_energy_kwh,
    format_power,
    format_carbon_summary,
)
from experimaestro.carbon.storage import (
    CarbonRecord,
    CarbonStorage,
)
from experimaestro.carbon.region import get_cached_region_info, RegionInfo
from experimaestro.carbon import create_tracker
from experimaestro.scheduler.state_status import CarbonMetricsEvent
from experimaestro.scheduler.state_provider import MockJob, CarbonMetricsData
from experimaestro.settings import CarbonSettings


class TestCarbonMetrics:
    """Tests for CarbonMetrics dataclass."""

    def test_default_values(self):
        """Test default values for CarbonMetrics."""
        metrics = CarbonMetrics()
        assert metrics.co2_kg == 0.0
        assert metrics.energy_kwh == 0.0
        assert metrics.cpu_power_w == 0.0
        assert metrics.gpu_power_w == 0.0
        assert metrics.ram_power_w == 0.0
        assert metrics.duration_s == 0.0
        assert metrics.region == ""
        assert metrics.timestamp > 0

    def test_custom_values(self):
        """Test CarbonMetrics with custom values."""
        metrics = CarbonMetrics(
            co2_kg=0.5,
            energy_kwh=1.2,
            cpu_power_w=45.0,
            gpu_power_w=120.0,
            ram_power_w=10.0,
            duration_s=3600.0,
            region="FRA",
            timestamp=1234567890.0,
        )
        assert metrics.co2_kg == 0.5
        assert metrics.energy_kwh == 1.2
        assert metrics.cpu_power_w == 45.0
        assert metrics.gpu_power_w == 120.0
        assert metrics.ram_power_w == 10.0
        assert metrics.duration_s == 3600.0
        assert metrics.region == "FRA"
        assert metrics.timestamp == 1234567890.0


class TestNullCarbonTracker:
    """Tests for NullCarbonTracker (no-op implementation)."""

    def test_start_does_nothing(self):
        """Test that start() doesn't raise."""
        tracker = NullCarbonTracker()
        tracker.start()  # Should not raise

    def test_stop_returns_empty_metrics(self):
        """Test that stop() returns zero metrics."""
        tracker = NullCarbonTracker()
        tracker.start()
        metrics = tracker.stop()
        assert metrics.co2_kg == 0.0
        assert metrics.energy_kwh == 0.0

    def test_get_current_metrics_returns_empty(self):
        """Test that get_current_metrics() returns zero metrics."""
        tracker = NullCarbonTracker()
        tracker.start()
        metrics = tracker.get_current_metrics()
        assert metrics.co2_kg == 0.0
        assert metrics.energy_kwh == 0.0


class TestFormatUtils:
    """Tests for formatting utilities."""

    def test_format_co2_grams(self):
        """Test CO2 formatting for gram values."""
        assert format_co2(0.001) == "0.001g"
        assert format_co2(0.5) == "0.500g"
        assert format_co2(1.5) == "1.50g"
        assert format_co2(50.5) == "50.5g"
        assert format_co2(999.9) == "999.9g"

    def test_format_co2_kilograms(self):
        """Test CO2 formatting for kilogram values."""
        assert format_co2(1000) == "1.00kg"
        assert format_co2(1500) == "1.50kg"
        assert format_co2(10000) == "10.00kg"

    def test_format_co2_negative(self):
        """Test CO2 formatting for negative values."""
        assert format_co2(-1) == "-"

    def test_format_co2_kg(self):
        """Test CO2 formatting from kg input."""
        assert format_co2_kg(0.001) == "1.00g"
        assert format_co2_kg(0.5) == "500.0g"
        assert format_co2_kg(1.5) == "1.50kg"

    def test_format_energy_wh(self):
        """Test energy formatting for Wh values."""
        assert format_energy(0.001) == "0.001Wh"
        assert format_energy(0.5) == "0.500Wh"
        assert format_energy(1.5) == "1.50Wh"
        assert format_energy(50.5) == "50.5Wh"
        assert format_energy(999.9) == "999.9Wh"

    def test_format_energy_kwh(self):
        """Test energy formatting for kWh values."""
        assert format_energy(1000) == "1.00kWh"
        assert format_energy(1500) == "1.50kWh"
        assert format_energy(10000) == "10.00kWh"

    def test_format_energy_negative(self):
        """Test energy formatting for negative values."""
        assert format_energy(-1) == "-"

    def test_format_energy_kwh_func(self):
        """Test energy formatting from kWh input."""
        assert format_energy_kwh(0.001) == "1.00Wh"
        assert format_energy_kwh(0.5) == "500.0Wh"
        assert format_energy_kwh(1.5) == "1.50kWh"

    def test_format_power(self):
        """Test power formatting."""
        assert format_power(0.001) == "0.001W"
        assert format_power(0.5) == "0.500W"
        assert format_power(1.5) == "1.50W"
        assert format_power(50.5) == "50.5W"
        assert format_power(-1) == "-"

    def test_format_carbon_summary(self):
        """Test carbon summary formatting."""
        summary = format_carbon_summary(0.5, 1.2, 3600)
        assert "500.0g" in summary
        assert "1.20kWh" in summary
        assert "1.0h" in summary


class TestCarbonRecord:
    """Tests for CarbonRecord dataclass."""

    def test_to_dict(self):
        """Test serialization to dictionary."""
        record = CarbonRecord(
            job_id="job123",
            task_id="module.Task",
            started_at="2025-01-15T10:00:00",
            ended_at="2025-01-15T11:00:00",
            co2_kg=0.5,
            energy_kwh=1.2,
            cpu_power_w=45.0,
            gpu_power_w=120.0,
            ram_power_w=10.0,
            duration_s=3600.0,
            region="FRA",
        )
        d = record.to_dict()
        assert d["job_id"] == "job123"
        assert d["task_id"] == "module.Task"
        assert d["co2_kg"] == 0.5
        assert d["energy_kwh"] == 1.2

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        d = {
            "job_id": "job456",
            "task_id": "module.Task2",
            "started_at": "2025-01-15T10:00:00",
            "ended_at": "2025-01-15T11:00:00",
            "co2_kg": 0.75,
            "energy_kwh": 1.5,
            "cpu_power_w": 50.0,
            "gpu_power_w": 100.0,
            "ram_power_w": 8.0,
            "duration_s": 3600.0,
            "region": "USA",
        }
        record = CarbonRecord.from_dict(d)
        assert record.job_id == "job456"
        assert record.co2_kg == 0.75

    def test_to_json_line(self):
        """Test JSON line serialization."""
        record = CarbonRecord(
            job_id="job789",
            task_id="module.Task3",
            started_at="2025-01-15T10:00:00",
            ended_at="2025-01-15T11:00:00",
            co2_kg=0.25,
            energy_kwh=0.8,
            cpu_power_w=30.0,
            gpu_power_w=0.0,
            ram_power_w=5.0,
            duration_s=1800.0,
            region="DEU",
        )
        line = record.to_json_line()
        parsed = json.loads(line)
        assert parsed["job_id"] == "job789"
        assert parsed["co2_kg"] == 0.25


class TestCarbonStorage:
    """Tests for carbon storage."""

    def test_write_and_read_record(self):
        """Test writing and reading a single record."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            storage = CarbonStorage(workspace)

            record = CarbonRecord(
                job_id="test_job_1",
                task_id="test.Task",
                started_at="2025-01-15T10:00:00",
                ended_at="2025-01-15T11:00:00",
                co2_kg=0.1,
                energy_kwh=0.5,
                cpu_power_w=20.0,
                gpu_power_w=0.0,
                ram_power_w=5.0,
                duration_s=3600.0,
                region="FRA",
            )
            storage.write_record(record)

            # Read back
            records = list(storage.read_records())
            assert len(records) == 1
            assert records[0].job_id == "test_job_1"
            assert records[0].co2_kg == 0.1

    def test_get_job_records(self):
        """Test getting records for a specific job."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            storage = CarbonStorage(workspace)

            # Write multiple records for different jobs
            for i in range(3):
                record = CarbonRecord(
                    job_id=f"job_{i}",
                    task_id="test.Task",
                    started_at="2025-01-15T10:00:00",
                    ended_at="2025-01-15T11:00:00",
                    co2_kg=0.1 * (i + 1),
                    energy_kwh=0.5,
                    cpu_power_w=20.0,
                    gpu_power_w=0.0,
                    ram_power_w=5.0,
                    duration_s=3600.0,
                    region="FRA",
                )
                storage.write_record(record)

            # Get records for job_1 only
            records = storage.get_job_records("job_1")
            assert len(records) == 1
            assert records[0].job_id == "job_1"
            assert records[0].co2_kg == 0.2

    def test_aggregate_for_jobs(self):
        """Test aggregating carbon metrics for multiple jobs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            storage = CarbonStorage(workspace)

            # Write records
            for i in range(3):
                record = CarbonRecord(
                    job_id=f"job_{i}",
                    task_id="test.Task",
                    started_at="2025-01-15T10:00:00",
                    ended_at="2025-01-15T11:00:00",
                    co2_kg=0.1,
                    energy_kwh=0.5,
                    cpu_power_w=20.0,
                    gpu_power_w=0.0,
                    ram_power_w=5.0,
                    duration_s=1000.0,
                    region="FRA",
                )
                storage.write_record(record)

            # Aggregate for all jobs
            result = storage.aggregate_for_jobs({"job_0", "job_1", "job_2"})
            assert result["job_count"] == 3
            assert abs(result["co2_kg"] - 0.3) < 0.001
            assert abs(result["energy_kwh"] - 1.5) < 0.001
            assert abs(result["duration_s"] - 3000.0) < 0.001

    def test_get_stats(self):
        """Test getting storage statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            storage = CarbonStorage(workspace)

            # Initially empty
            stats = storage.get_stats()
            assert stats["total_records"] == 0
            assert stats["file_count"] == 0

            # Write some records
            for i in range(5):
                record = CarbonRecord(
                    job_id=f"job_{i}",
                    task_id="test.Task",
                    started_at="2025-01-15T10:00:00",
                    ended_at="2025-01-15T11:00:00",
                    co2_kg=0.1,
                    energy_kwh=0.5,
                    cpu_power_w=20.0,
                    gpu_power_w=0.0,
                    ram_power_w=5.0,
                    duration_s=3600.0,
                    region="FRA",
                )
                storage.write_record(record)

            stats = storage.get_stats()
            assert stats["total_records"] == 5
            assert stats["file_count"] == 1
            assert stats["size_bytes"] > 0


class TestRegionDetection:
    """Tests for region detection and caching."""

    def test_region_info_typeddict(self):
        """Test RegionInfo TypedDict."""
        info: RegionInfo = {
            "country_iso_code": "FRA",
            "country_name": "France",
            "region": "europe-west",
            "detected_at": time.time(),
        }
        assert info["country_iso_code"] == "FRA"
        assert info["region"] == "europe-west"

    def test_get_cached_region_with_detection(self):
        """Test that region detection works."""
        # Mock the _detect_region function to avoid actual detection
        with patch("experimaestro.carbon.region._detect_region") as mock_detect:
            mock_detect.return_value = {
                "country_iso_code": "DEU",
                "country_name": "Germany",
                "region": "europe-west",
                "detected_at": time.time(),
            }
            # Clear any existing cache
            with patch("experimaestro.carbon.region._load_cache", return_value=None):
                with patch("experimaestro.carbon.region._save_cache"):
                    result = get_cached_region_info()

                    assert result is not None
                    assert result["country_iso_code"] == "DEU"

    def test_get_cached_region_uses_cache(self):
        """Test that cached region is used when available."""
        cached_data: RegionInfo = {
            "country_iso_code": "GBR",
            "country_name": "United Kingdom",
            "region": "europe-west",
            "detected_at": time.time(),
        }

        # Mock _load_cache to return cached data
        with patch("experimaestro.carbon.region._load_cache", return_value=cached_data):
            with patch("experimaestro.carbon.region._detect_region") as mock_detect:
                result = get_cached_region_info()

                # Detection should not be called
                mock_detect.assert_not_called()

                assert result is not None
                assert result["country_iso_code"] == "GBR"


class TestCarbonSettings:
    """Tests for CarbonSettings dataclass."""

    def test_default_settings(self):
        """Test default carbon settings."""
        settings = CarbonSettings()
        assert settings.enabled is True
        assert settings.provider == "codecarbon"
        assert settings.country_iso_code is None
        assert settings.region is None
        assert settings.report_interval_s == 60.0
        assert settings.warn_if_unavailable is True

    def test_custom_settings(self):
        """Test custom carbon settings."""
        settings = CarbonSettings(
            enabled=False,
            provider="custom",
            country_iso_code="FRA",
            region="europe-west",
            report_interval_s=30.0,
            warn_if_unavailable=False,
        )
        assert settings.enabled is False
        assert settings.country_iso_code == "FRA"
        assert settings.report_interval_s == 30.0


class TestCarbonMetricsEvent:
    """Tests for CarbonMetricsEvent."""

    def test_event_creation(self):
        """Test creating a CarbonMetricsEvent."""
        event = CarbonMetricsEvent(
            job_id="test_job",
            co2_kg=0.5,
            energy_kwh=1.2,
            cpu_power_w=45.0,
            gpu_power_w=120.0,
            ram_power_w=10.0,
            duration_s=3600.0,
            region="FRA",
            is_final=True,
        )
        assert event.job_id == "test_job"
        assert event.co2_kg == 0.5
        assert event.is_final is True


class TestMockJobCarbonMetrics:
    """Tests for MockJob carbon metrics handling."""

    def test_apply_carbon_metrics_event(self):
        """Test applying CarbonMetricsEvent to MockJob."""
        job = MockJob(
            identifier="test_job_123",
            task_id="module.TestTask",
            path=Path("/tmp/test"),
            state="running",
            submittime=datetime.now(),
            starttime=datetime.now(),
            endtime=None,
            progress=[],
            updated_at="2025-01-15T10:00:00",
        )

        # Initially no carbon metrics
        assert job.carbon_metrics is None

        # Apply carbon metrics event
        event = CarbonMetricsEvent(
            job_id="test_job_123",
            co2_kg=0.25,
            energy_kwh=0.8,
            cpu_power_w=35.0,
            gpu_power_w=80.0,
            ram_power_w=8.0,
            duration_s=1800.0,
            region="FRA",
            is_final=False,
        )
        job.apply_event(event)

        # Check carbon metrics updated
        assert job.carbon_metrics is not None
        assert job.carbon_metrics.co2_kg == 0.25
        assert job.carbon_metrics.energy_kwh == 0.8
        assert job.carbon_metrics.cpu_power_w == 35.0
        assert job.carbon_metrics.gpu_power_w == 80.0
        assert job.carbon_metrics.region == "FRA"
        assert job.carbon_metrics.is_final is False

    def test_carbon_metrics_data_init(self):
        """Test CarbonMetricsData initialization."""
        data = CarbonMetricsData(
            co2_kg=0.5,
            energy_kwh=1.0,
            cpu_power_w=50.0,
            gpu_power_w=100.0,
            ram_power_w=10.0,
            duration_s=3600.0,
            region="DEU",
            is_final=True,
        )
        assert data.co2_kg == 0.5
        assert data.is_final is True


class TestCarbonAvailability:
    """Tests for carbon tracking availability."""

    def test_is_available_without_codecarbon(self):
        """Test is_available returns False when codecarbon not installed."""
        with patch.dict("sys.modules", {"codecarbon": None}):
            # Force re-evaluation by patching the import
            with patch("experimaestro.carbon.is_available") as mock_available:
                mock_available.return_value = False
                assert mock_available() is False

    def test_create_tracker_returns_null_when_unavailable(self):
        """Test create_tracker returns NullCarbonTracker when unavailable."""
        # Mock both platform-specific checks to return False
        with (
            patch("experimaestro.carbon._is_macos_apple_silicon", return_value=False),
            patch("experimaestro.carbon._is_codecarbon_available", return_value=False),
        ):
            tracker = create_tracker(show_warning=False)
            assert isinstance(tracker, NullCarbonTracker)
