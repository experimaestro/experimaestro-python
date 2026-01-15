"""Base classes for carbon tracking abstraction."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import time
from typing import Protocol, runtime_checkable


@dataclass
class CarbonMetrics:
    """Carbon metrics from a tracking session.

    All values are cumulative from when tracking started.
    """

    co2_kg: float = 0.0
    """CO2 equivalent emissions in kilograms."""

    energy_kwh: float = 0.0
    """Energy consumed in kilowatt-hours."""

    cpu_power_w: float = 0.0
    """Average CPU power consumption in watts."""

    gpu_power_w: float = 0.0
    """Average GPU power consumption in watts."""

    ram_power_w: float = 0.0
    """Average RAM power consumption in watts."""

    duration_s: float = 0.0
    """Duration of tracking in seconds."""

    region: str = ""
    """Region/country code used for carbon intensity."""

    timestamp: float = field(default_factory=time.time)
    """Timestamp when metrics were captured."""

    @property
    def co2_g(self) -> float:
        """CO2 equivalent emissions in grams."""
        return self.co2_kg * 1000

    @property
    def energy_wh(self) -> float:
        """Energy consumed in watt-hours."""
        return self.energy_kwh * 1000

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "co2_kg": self.co2_kg,
            "energy_kwh": self.energy_kwh,
            "cpu_power_w": self.cpu_power_w,
            "gpu_power_w": self.gpu_power_w,
            "ram_power_w": self.ram_power_w,
            "duration_s": self.duration_s,
            "region": self.region,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CarbonMetrics":
        """Create from dictionary."""
        return cls(
            co2_kg=data.get("co2_kg", 0.0),
            energy_kwh=data.get("energy_kwh", 0.0),
            cpu_power_w=data.get("cpu_power_w", 0.0),
            gpu_power_w=data.get("gpu_power_w", 0.0),
            ram_power_w=data.get("ram_power_w", 0.0),
            duration_s=data.get("duration_s", 0.0),
            region=data.get("region", ""),
            timestamp=data.get("timestamp", time.time()),
        )


@dataclass
class CarbonAggregateData:
    """Aggregated carbon metrics for a set of jobs."""

    co2_kg: float = 0.0
    """Total CO2 equivalent emissions in kilograms."""

    energy_kwh: float = 0.0
    """Total energy consumed in kilowatt-hours."""

    duration_s: float = 0.0
    """Total duration in seconds."""

    job_count: int = 0
    """Number of jobs included in the aggregate."""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "co2_kg": self.co2_kg,
            "energy_kwh": self.energy_kwh,
            "duration_s": self.duration_s,
            "job_count": self.job_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CarbonAggregateData":
        """Create from dictionary."""
        return cls(
            co2_kg=d.get("co2_kg", 0.0),
            energy_kwh=d.get("energy_kwh", 0.0),
            duration_s=d.get("duration_s", 0.0),
            job_count=d.get("job_count", 0),
        )


@dataclass
class CarbonImpactData:
    """Carbon impact data for an experiment (sum and latest aggregations)."""

    sum: CarbonAggregateData | None = None
    """Sum of all job runs (including retries)."""

    latest: CarbonAggregateData | None = None
    """Sum of only the latest run of each unique job."""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {}
        if self.sum is not None:
            result["sum"] = self.sum.to_dict()
        if self.latest is not None:
            result["latest"] = self.latest.to_dict()
        return result

    @classmethod
    def from_dict(cls, d: dict | None) -> "CarbonImpactData | None":
        """Create from dictionary."""
        if d is None:
            return None
        return cls(
            sum=CarbonAggregateData.from_dict(d["sum"]) if "sum" in d else None,
            latest=CarbonAggregateData.from_dict(d["latest"])
            if "latest" in d
            else None,
        )


@runtime_checkable
class CarbonTracker(Protocol):
    """Protocol for carbon tracking implementations.

    Implementations should track energy consumption and CO2 emissions
    during code execution.
    """

    def start(self) -> None:
        """Start tracking carbon emissions.

        Should be called before the code to be tracked begins execution.
        """
        ...

    def stop(self) -> CarbonMetrics:
        """Stop tracking and return final metrics.

        Returns:
            CarbonMetrics with cumulative values from the tracking session.
        """
        ...

    def get_current_metrics(self) -> CarbonMetrics:
        """Get current metrics without stopping tracking.

        Returns:
            CarbonMetrics with cumulative values up to this point.
        """
        ...

    @property
    def is_running(self) -> bool:
        """Whether tracking is currently active."""
        ...


class BaseCarbonTracker(ABC):
    """Abstract base class for carbon tracker implementations."""

    _running: bool = False

    @abstractmethod
    def start(self) -> None:
        """Start tracking carbon emissions."""
        ...

    @abstractmethod
    def stop(self) -> CarbonMetrics:
        """Stop tracking and return final metrics."""
        ...

    @abstractmethod
    def get_current_metrics(self) -> CarbonMetrics:
        """Get current metrics without stopping tracking."""
        ...

    @property
    def is_running(self) -> bool:
        """Whether tracking is currently active."""
        return self._running


class NullCarbonTracker(BaseCarbonTracker):
    """No-op carbon tracker for when tracking is disabled."""

    def start(self) -> None:
        """No-op start."""
        self._running = True

    def stop(self) -> CarbonMetrics:
        """Return empty metrics."""
        self._running = False
        return CarbonMetrics()

    def get_current_metrics(self) -> CarbonMetrics:
        """Return empty metrics."""
        return CarbonMetrics()
