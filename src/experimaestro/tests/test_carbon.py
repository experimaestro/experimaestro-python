"""Tests for carbon tracking functionality."""

import json
import tempfile
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from experimaestro.carbon.base import CarbonMetrics, NullCarbonTracker
from experimaestro.carbon.utils import (
    format_co2,
    format_co2_kg,
    format_energy,
    format_energy_kwh,
    format_power,
    format_carbon_summary,
    to_float,
)
from experimaestro.carbon.storage import (
    CarbonRecord,
    CarbonStorage,
)
from experimaestro.carbon.region import get_cached_region_info, RegionInfo
from experimaestro.carbon import create_tracker
from experimaestro.scheduler.state_status import CarbonMetricsEvent
from experimaestro.scheduler.state_provider import MockJob, CarbonMetricsData
from experimaestro.scheduler.interfaces import JobFailureStatus
from experimaestro.settings import CarbonSettings

# Mark all tests in this module as carbon tests
pytestmark = [pytest.mark.carbon]


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


class TestToFloat:
    """Tests for to_float conversion utility."""

    def test_none_returns_zero(self):
        """Test None input returns 0.0."""
        assert to_float(None) == 0.0

    def test_int_conversion(self):
        """Test int input is converted to float."""
        assert to_float(42) == 42.0
        assert to_float(0) == 0.0
        assert to_float(-5) == -5.0

    def test_float_passthrough(self):
        """Test float input is returned as-is."""
        assert to_float(3.14) == 3.14
        assert to_float(0.0) == 0.0

    def test_dict_with_kwh_key(self):
        """Test dict with 'kWh' key extracts the value."""
        assert to_float({"kWh": 0.006442117743102216}) == 0.006442117743102216
        assert to_float({"kWh": 1.5}) == 1.5

    def test_empty_dict_returns_zero(self):
        """Test empty dict returns 0.0."""
        assert to_float({}) == 0.0

    def test_dict_without_kwh_returns_zero(self):
        """Test dict without 'kWh' key returns 0.0."""
        assert to_float({"other": 1.5}) == 0.0

    def test_invalid_string_returns_zero(self):
        """Test invalid string returns 0.0."""
        assert to_float("invalid") == 0.0

    def test_numeric_string_converts(self):
        """Test numeric string is converted."""
        assert to_float("3.14") == 3.14
        assert to_float("42") == 42.0

    def test_object_with_kwh_attribute(self):
        """Test object with kWh attribute extracts the value."""

        class Energy:
            def __init__(self, kwh):
                self.kWh = kwh

        assert to_float(Energy(1.5)) == 1.5
        assert to_float(Energy(0.0)) == 0.0


class TestCarbonMetricsWrittenField:
    """Tests for the 'written' field in carbon metrics."""

    def test_event_written_field_default(self):
        """Test CarbonMetricsEvent has written=False by default."""
        event = CarbonMetricsEvent(
            job_id="test_job",
            co2_kg=0.5,
            energy_kwh=1.2,
        )
        assert event.written is False

    def test_event_written_field_explicit(self):
        """Test CarbonMetricsEvent can set written=True."""
        event = CarbonMetricsEvent(
            job_id="test_job",
            co2_kg=0.5,
            energy_kwh=1.2,
            written=True,
        )
        assert event.written is True

    def test_carbon_metrics_data_written_field(self):
        """Test CarbonMetricsData has written field."""
        data = CarbonMetricsData(
            co2_kg=0.5,
            energy_kwh=1.0,
            written=True,
        )
        assert data.written is True

        data_default = CarbonMetricsData(co2_kg=0.5, energy_kwh=1.0)
        assert data_default.written is False

    def test_apply_event_copies_written_field(self):
        """Test applying CarbonMetricsEvent copies the written field."""
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

        # Apply event with written=True
        event = CarbonMetricsEvent(
            job_id="test_job_123",
            co2_kg=0.25,
            energy_kwh=0.8,
            is_final=True,
            written=True,
        )
        job.apply_event(event)

        assert job.carbon_metrics is not None
        assert job.carbon_metrics.written is True

        # Apply event with written=False
        event_not_written = CarbonMetricsEvent(
            job_id="test_job_123",
            co2_kg=0.3,
            energy_kwh=0.9,
            is_final=True,
            written=False,
        )
        job.apply_event(event_not_written)

        assert job.carbon_metrics.written is False


class TestCarbonAccumulationAcrossRetries:
    """Tests for carbon tracking accumulation across job retries.

    This tests the scenario where:
    - A resumable job runs and emits carbon metrics
    - The job is killed (hard timeout) or exits gracefully (GracefulTimeout)
    - The job restarts and runs again
    - Carbon metrics should be accumulated across all runs
    """

    def test_finalize_status_writes_unwritten_carbon_record(self):
        """Test that finalize_status writes carbon record if not written."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            # Create job directory structure: workspace/jobs/task_id/job_id
            task_id = "test.CarbonTask"
            job_id = "carbon_job_123"
            job_path = workspace / "jobs" / task_id / job_id
            job_path.mkdir(parents=True)
            xpm_dir = job_path / ".experimaestro"
            xpm_dir.mkdir()

            # Create a MockJob with unwritten carbon metrics
            job = MockJob(
                identifier=job_id,
                task_id=task_id,
                path=job_path,
                state="running",
                submittime=datetime.now(),
                starttime=datetime.now(),
                endtime=None,
                progress=[],
                updated_at="2025-01-15T10:00:00",
            )

            # Set carbon metrics with written=False (simulating hard timeout)
            job.carbon_metrics = CarbonMetricsData(
                co2_kg=0.5,
                energy_kwh=1.2,
                cpu_power_w=45.0,
                gpu_power_w=120.0,
                ram_power_w=10.0,
                duration_s=3600.0,
                region="FRA",
                is_final=True,
                written=False,
            )

            # Write initial status
            job.write_status()

            # Now call finalize_status with cleanup_events=True
            import asyncio

            async def run_finalize():
                return await job.finalize_status(cleanup_events=True)

            asyncio.run(run_finalize())

            # Check carbon storage has the record
            storage = CarbonStorage(workspace)
            records = storage.get_job_records(job_id)
            assert len(records) == 1
            assert records[0].co2_kg == 0.5
            assert records[0].energy_kwh == 1.2
            assert records[0].task_id == task_id

            # Check that written flag is now True
            assert job.carbon_metrics.written is True

    def test_finalize_status_skips_already_written_carbon_record(self):
        """Test that finalize_status doesn't duplicate already-written records."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            task_id = "test.CarbonTask"
            job_id = "carbon_job_456"
            job_path = workspace / "jobs" / task_id / job_id
            job_path.mkdir(parents=True)
            xpm_dir = job_path / ".experimaestro"
            xpm_dir.mkdir()

            # Write a record to storage first
            storage = CarbonStorage(workspace)
            storage.write_record(
                CarbonRecord(
                    job_id=job_id,
                    task_id=task_id,
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
            )

            # Create a MockJob with written=True
            job = MockJob(
                identifier=job_id,
                task_id=task_id,
                path=job_path,
                state="running",
                submittime=datetime.now(),
                starttime=datetime.now(),
                endtime=None,
                progress=[],
                updated_at="2025-01-15T10:00:00",
            )
            job.carbon_metrics = CarbonMetricsData(
                co2_kg=0.5,
                energy_kwh=1.2,
                written=True,  # Already written
            )
            job.write_status()

            # Call finalize_status
            import asyncio

            async def run_finalize():
                return await job.finalize_status(cleanup_events=True)

            asyncio.run(run_finalize())

            # Should still have only 1 record (not duplicated)
            records = storage.get_job_records(job_id)
            assert len(records) == 1

    def test_carbon_accumulation_across_multiple_runs(self):
        """Test that carbon records accumulate when using aggregate_for_jobs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            storage = CarbonStorage(workspace)

            job_id = "resumable_job_789"
            task_id = "test.ResumableTask"

            # Simulate first run (hard timeout - no graceful exit)
            storage.write_record(
                CarbonRecord(
                    job_id=job_id,
                    task_id=task_id,
                    started_at="2025-01-15T10:00:00",
                    ended_at="2025-01-15T10:30:00",
                    co2_kg=0.1,
                    energy_kwh=0.3,
                    cpu_power_w=40.0,
                    gpu_power_w=100.0,
                    ram_power_w=8.0,
                    duration_s=1800.0,
                    region="FRA",
                )
            )

            # Simulate second run (also hard timeout)
            storage.write_record(
                CarbonRecord(
                    job_id=job_id,
                    task_id=task_id,
                    started_at="2025-01-15T10:35:00",
                    ended_at="2025-01-15T11:00:00",
                    co2_kg=0.15,
                    energy_kwh=0.4,
                    cpu_power_w=45.0,
                    gpu_power_w=110.0,
                    ram_power_w=9.0,
                    duration_s=1500.0,
                    region="FRA",
                )
            )

            # Simulate third run (completes successfully)
            storage.write_record(
                CarbonRecord(
                    job_id=job_id,
                    task_id=task_id,
                    started_at="2025-01-15T11:05:00",
                    ended_at="2025-01-15T11:30:00",
                    co2_kg=0.12,
                    energy_kwh=0.35,
                    cpu_power_w=42.0,
                    gpu_power_w=105.0,
                    ram_power_w=8.5,
                    duration_s=1500.0,
                    region="FRA",
                )
            )

            # Get all records for this job
            records = storage.get_job_records(job_id)
            assert len(records) == 3

            # Aggregate all records (sum across retries)
            result = storage.aggregate_for_jobs({job_id}, use_latest_only=False)
            assert result["job_count"] == 1  # 1 unique job
            assert abs(result["co2_kg"] - 0.37) < 0.001  # 0.1 + 0.15 + 0.12
            assert abs(result["energy_kwh"] - 1.05) < 0.001  # 0.3 + 0.4 + 0.35
            assert abs(result["duration_s"] - 4800.0) < 0.001  # 1800 + 1500 + 1500

            # Aggregate latest only (for final metrics)
            result_latest = storage.aggregate_for_jobs({job_id}, use_latest_only=True)
            assert result_latest["job_count"] == 1
            assert abs(result_latest["co2_kg"] - 0.12) < 0.001  # Only the last run
            assert abs(result_latest["energy_kwh"] - 0.35) < 0.001

    def test_hard_timeout_scenario_with_status(self):
        """Test simulating a hard SLURM timeout where carbon metrics are in status.

        In this scenario:
        - Job runs and carbon metrics are saved to status.json (via periodic updates)
        - Job is killed by SLURM (hard timeout) - carbon record not written to storage
        - On restart, finalize_status writes the carbon record to storage
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            task_id = "test.HardTimeoutTask"
            job_id = "hard_timeout_job"
            job_path = workspace / "jobs" / task_id / job_id
            job_path.mkdir(parents=True)
            xpm_dir = job_path / ".experimaestro"
            xpm_dir.mkdir()

            # Create MockJob with carbon metrics that weren't written (hard timeout)
            job = MockJob(
                identifier=job_id,
                task_id=task_id,
                path=job_path,
                state="error",
                failure_reason=JobFailureStatus.TIMEOUT,
                submittime=datetime.now(),
                starttime=datetime.now(),
                endtime=datetime.now(),
                progress=[],
                updated_at="2025-01-15T10:00:00",
            )
            # Simulate carbon metrics from periodic updates (not written to storage)
            job.carbon_metrics = CarbonMetricsData(
                co2_kg=0.10,
                energy_kwh=0.30,
                cpu_power_w=40.0,
                gpu_power_w=100.0,
                ram_power_w=8.0,
                duration_s=600.0,
                region="FRA",
                is_final=False,  # Was periodic, not final
                written=False,  # Not written to storage (hard timeout)
            )
            job.write_status()

            # On restart, finalize_status is called to recover state
            import asyncio

            async def run_finalize():
                return await job.finalize_status(cleanup_events=True)

            asyncio.run(run_finalize())

            # Should have written carbon record to storage
            assert job.carbon_metrics is not None
            assert job.carbon_metrics.written is True

            # Check storage has the record
            storage = CarbonStorage(workspace)
            records = storage.get_job_records(job_id)
            assert len(records) == 1
            assert records[0].co2_kg == 0.10

    def test_graceful_timeout_scenario_no_duplicate(self):
        """Test graceful timeout where carbon record was already written.

        In this scenario:
        - Job runs and catches the timeout signal
        - Job writes carbon record to storage with written=True
        - On restart, no duplicate record should be written
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            task_id = "test.GracefulTimeoutTask"
            job_id = "graceful_timeout_job"
            job_path = workspace / "jobs" / task_id / job_id
            job_path.mkdir(parents=True)
            xpm_dir = job_path / ".experimaestro"
            xpm_dir.mkdir()

            # Pre-write the carbon record (simulating successful write during graceful shutdown)
            storage = CarbonStorage(workspace)
            storage.write_record(
                CarbonRecord(
                    job_id=job_id,
                    task_id=task_id,
                    started_at="2025-01-15T10:00:00",
                    ended_at="2025-01-15T10:45:00",
                    co2_kg=0.20,
                    energy_kwh=0.60,
                    cpu_power_w=40.0,
                    gpu_power_w=100.0,
                    ram_power_w=8.0,
                    duration_s=2700.0,
                    region="FRA",
                )
            )

            # Create MockJob with written=True (graceful shutdown wrote to storage)
            job = MockJob(
                identifier=job_id,
                task_id=task_id,
                path=job_path,
                state="error",
                failure_reason=JobFailureStatus.TIMEOUT,
                submittime=datetime.now(),
                starttime=datetime.now(),
                endtime=datetime.now(),
                progress=[],
                updated_at="2025-01-15T10:00:00",
            )
            job.carbon_metrics = CarbonMetricsData(
                co2_kg=0.20,
                energy_kwh=0.60,
                duration_s=2700.0,
                is_final=True,
                written=True,  # Storage write succeeded
            )
            job.write_status()

            # On restart
            import asyncio

            async def run_finalize():
                return await job.finalize_status(cleanup_events=True)

            asyncio.run(run_finalize())

            # Check metrics still marked as written
            assert job.carbon_metrics is not None
            assert job.carbon_metrics.written is True

            # Should still have only 1 record (no duplicate)
            records = storage.get_job_records(job_id)
            assert len(records) == 1
