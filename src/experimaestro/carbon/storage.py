"""Carbon storage for carbon metrics.

This module handles persistent storage of carbon metrics in the workspace.
Metrics are stored in JSONL files with automatic rotation when file limits
are reached.

Storage location: WORKSPACE/carbon/
- measures-0.jsonl (max 32K records)
- measures-1.jsonl (next file when limit reached)
- index.json (metadata: current file, total records)
"""

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from experimaestro.locking import create_file_lock
from typing import Iterator

logger = logging.getLogger(__name__)

# Maximum records per file (32K)
MAX_RECORDS_PER_FILE = 32 * 1024

# Carbon directory name
CARBON_DIR = "carbon"


@dataclass
class CarbonRecord:
    """A single carbon measurement record for a job."""

    job_id: str
    """Unique job identifier."""

    task_id: str
    """Task type identifier (e.g., 'mymodule.MyTask')."""

    started_at: str
    """ISO format timestamp when job started."""

    ended_at: str
    """ISO format timestamp when job ended."""

    co2_kg: float
    """Total CO2 equivalent emissions in kilograms."""

    energy_kwh: float
    """Total energy consumed in kilowatt-hours."""

    cpu_power_w: float
    """Average CPU power in watts."""

    gpu_power_w: float
    """Average GPU power in watts."""

    ram_power_w: float
    """Average RAM power in watts."""

    duration_s: float
    """Total duration in seconds."""

    region: str
    """Region code used for carbon intensity."""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "job_id": self.job_id,
            "task_id": self.task_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "co2_kg": self.co2_kg,
            "energy_kwh": self.energy_kwh,
            "cpu_power_w": self.cpu_power_w,
            "gpu_power_w": self.gpu_power_w,
            "ram_power_w": self.ram_power_w,
            "duration_s": self.duration_s,
            "region": self.region,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CarbonRecord":
        """Create from dictionary."""
        return cls(
            job_id=data["job_id"],
            task_id=data["task_id"],
            started_at=data["started_at"],
            ended_at=data["ended_at"],
            co2_kg=data.get("co2_kg", 0.0),
            energy_kwh=data.get("energy_kwh", 0.0),
            cpu_power_w=data.get("cpu_power_w", 0.0),
            gpu_power_w=data.get("gpu_power_w", 0.0),
            ram_power_w=data.get("ram_power_w", 0.0),
            duration_s=data.get("duration_s", 0.0),
            region=data.get("region", ""),
        )

    def to_json_line(self) -> str:
        """Convert to JSON line for JSONL file."""
        return json.dumps(self.to_dict(), separators=(",", ":"))


@dataclass
class CarbonIndex:
    """Index metadata for carbon storage."""

    current_file: int = 0
    """Current file number being written to."""

    total_records: int = 0
    """Total number of records across all files."""

    records_in_current: int = 0
    """Number of records in current file."""

    last_updated: float = 0.0
    """Timestamp of last update."""

    def to_dict(self) -> dict:
        return {
            "current_file": self.current_file,
            "total_records": self.total_records,
            "records_in_current": self.records_in_current,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CarbonIndex":
        return cls(
            current_file=data.get("current_file", 0),
            total_records=data.get("total_records", 0),
            records_in_current=data.get("records_in_current", 0),
            last_updated=data.get("last_updated", 0.0),
        )


class CarbonStorage:
    """Storage manager for carbon metrics.

    Handles reading and writing carbon records to JSONL files with
    automatic rotation when file limits are reached.
    """

    def __init__(self, workspace_path: Path):
        """Initialize carbon storage.

        Args:
            workspace_path: Path to the workspace directory.
        """
        self.workspace_path = workspace_path
        self._dir = workspace_path / CARBON_DIR
        self._index_path = self._dir / "index.json"
        self._lock_path = self._dir / ".lock"

    @property
    def storage_dir(self) -> Path:
        """Get the carbon storage directory."""
        return self._dir

    def _ensure_dir(self) -> None:
        """Ensure storage directory exists."""
        self._dir.mkdir(parents=True, exist_ok=True)

    def _get_file_path(self, file_num: int) -> Path:
        """Get path for a specific measures file."""
        return self._dir / f"measures-{file_num}.jsonl"

    def _load_index(self) -> CarbonIndex:
        """Load index from disk, creating if needed."""
        if not self._index_path.exists():
            return CarbonIndex(last_updated=time.time())

        try:
            with self._index_path.open() as f:
                data = json.load(f)
            return CarbonIndex.from_dict(data)
        except Exception as e:
            logger.warning("Failed to load carbon index: %s", e)
            return CarbonIndex(last_updated=time.time())

    def _save_index(self, index: CarbonIndex) -> None:
        """Save index to disk."""
        self._ensure_dir()
        index.last_updated = time.time()
        temp_path = self._index_path.with_suffix(".tmp")
        with temp_path.open("w") as f:
            json.dump(index.to_dict(), f, indent=2)
        temp_path.replace(self._index_path)

    def write_record(self, record: CarbonRecord) -> None:
        """Write a carbon record to storage.

        Uses file locking for concurrent access safety.
        Automatically rotates to new file when limit is reached.

        Args:
            record: Carbon record to write.
        """
        self._ensure_dir()

        with create_file_lock(self._lock_path):
            index = self._load_index()

            # Check if we need to rotate
            if index.records_in_current >= MAX_RECORDS_PER_FILE:
                index.current_file += 1
                index.records_in_current = 0

            # Write record
            file_path = self._get_file_path(index.current_file)
            with file_path.open("a") as f:
                f.write(record.to_json_line() + "\n")

            # Update index
            index.records_in_current += 1
            index.total_records += 1
            self._save_index(index)

            logger.debug(
                "Wrote carbon record for job %s (total: %d)",
                record.job_id,
                index.total_records,
            )

    def read_records(
        self,
        *,
        job_ids: set[str] | None = None,
        since: datetime | None = None,
    ) -> Iterator[CarbonRecord]:
        """Read carbon records from storage.

        Args:
            job_ids: Optional set of job IDs to filter by.
            since: Optional datetime to filter records after.

        Yields:
            CarbonRecord objects matching the filters.
        """
        if not self._dir.exists():
            return

        index = self._load_index()

        # Read all files up to current
        for file_num in range(index.current_file + 1):
            file_path = self._get_file_path(file_num)
            if not file_path.exists():
                continue

            try:
                with file_path.open() as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue

                        try:
                            data = json.loads(line)
                            record = CarbonRecord.from_dict(data)

                            # Apply filters
                            if job_ids is not None and record.job_id not in job_ids:
                                continue

                            if since is not None:
                                try:
                                    ended = datetime.fromisoformat(record.ended_at)
                                    if ended < since:
                                        continue
                                except Exception:
                                    pass

                            yield record

                        except Exception as e:
                            logger.warning("Failed to parse carbon record: %s", e)
                            continue

            except Exception as e:
                logger.warning("Failed to read %s: %s", file_path, e)
                continue

    def get_job_records(self, job_id: str) -> list[CarbonRecord]:
        """Get all records for a specific job.

        Args:
            job_id: Job identifier.

        Returns:
            List of CarbonRecord for this job (may have multiple runs).
        """
        return list(self.read_records(job_ids={job_id}))

    def get_latest_job_record(self, job_id: str) -> CarbonRecord | None:
        """Get the most recent record for a job.

        Args:
            job_id: Job identifier.

        Returns:
            Most recent CarbonRecord or None if not found.
        """
        records = self.get_job_records(job_id)
        if not records:
            return None

        # Sort by ended_at timestamp (most recent first)
        records.sort(key=lambda r: r.ended_at, reverse=True)
        return records[0]

    def aggregate_for_jobs(
        self,
        job_ids: set[str],
        *,
        use_latest_only: bool = False,
    ) -> dict:
        """Aggregate carbon metrics for a set of jobs.

        Args:
            job_ids: Set of job IDs to aggregate.
            use_latest_only: If True, only use the latest record per job.
                            If False, sum all records (including retries).

        Returns:
            Dictionary with aggregated metrics:
            {
                "co2_kg": float,
                "energy_kwh": float,
                "duration_s": float,
                "job_count": int
            }
        """
        total_co2 = 0.0
        total_energy = 0.0
        total_duration = 0.0
        job_count = 0

        if use_latest_only:
            # Get latest record per job
            for job_id in job_ids:
                record = self.get_latest_job_record(job_id)
                if record:
                    total_co2 += record.co2_kg
                    total_energy += record.energy_kwh
                    total_duration += record.duration_s
                    job_count += 1
        else:
            # Sum all records
            seen_jobs = set()
            for record in self.read_records(job_ids=job_ids):
                total_co2 += record.co2_kg
                total_energy += record.energy_kwh
                total_duration += record.duration_s
                seen_jobs.add(record.job_id)
            job_count = len(seen_jobs)

        return {
            "co2_kg": total_co2,
            "energy_kwh": total_energy,
            "duration_s": total_duration,
            "job_count": job_count,
        }

    def get_stats(self) -> dict:
        """Get storage statistics.

        Returns:
            Dictionary with storage statistics.
        """
        if not self._dir.exists():
            return {
                "total_records": 0,
                "file_count": 0,
                "size_bytes": 0,
            }

        index = self._load_index()

        # Calculate total size
        total_size = 0
        file_count = 0
        for file_num in range(index.current_file + 1):
            file_path = self._get_file_path(file_num)
            if file_path.exists():
                total_size += file_path.stat().st_size
                file_count += 1

        return {
            "total_records": index.total_records,
            "file_count": file_count,
            "size_bytes": total_size,
        }
