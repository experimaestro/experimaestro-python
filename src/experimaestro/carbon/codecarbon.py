"""CodeCarbon-based carbon tracker implementation."""

import logging
import tempfile
import time
from pathlib import Path

from experimaestro.carbon.base import BaseCarbonTracker, CarbonMetrics
from experimaestro.carbon.utils import to_float

logger = logging.getLogger(__name__)


class CodeCarbonTracker(BaseCarbonTracker):
    """Carbon tracker using CodeCarbon library.

    This wraps CodeCarbon's EmissionsTracker to provide carbon metrics
    during job execution.
    """

    def __init__(
        self,
        *,
        country_iso_code: str | None = None,
        region: str | None = None,
        output_dir: Path | None = None,
    ):
        """Initialize the CodeCarbon tracker.

        Args:
            country_iso_code: Override detected country (ISO 3166-1 alpha-3)
            region: Override detected region/cloud region
            output_dir: Directory for CodeCarbon output files (temp dir if None)
        """
        self._country_iso_code = country_iso_code
        self._region = region
        self._output_dir = output_dir
        self._tracker = None
        self._start_time: float | None = None
        self._temp_dir: tempfile.TemporaryDirectory | None = None

    def start(self) -> None:
        """Start tracking carbon emissions."""
        if self._running:
            logger.warning("Carbon tracker already running")
            return

        try:
            from codecarbon import EmissionsTracker

            # Use temp directory if no output dir specified
            if self._output_dir is None:
                self._temp_dir = tempfile.TemporaryDirectory(
                    prefix="experimaestro_carbon_"
                )
                output_dir = Path(self._temp_dir.name)
            else:
                output_dir = self._output_dir

            # Build tracker kwargs
            tracker_kwargs = {
                "output_dir": str(output_dir),
                "save_to_file": False,  # We handle our own storage
                "log_level": "error",  # Suppress verbose output
                "tracking_mode": "process",  # Track this process
            }

            if self._country_iso_code:
                tracker_kwargs["country_iso_code"] = self._country_iso_code
            if self._region:
                tracker_kwargs["region"] = self._region

            self._tracker = EmissionsTracker(**tracker_kwargs)
            self._tracker.start()
            self._start_time = time.time()
            self._running = True

            logger.debug("Carbon tracking started")

        except Exception as e:
            logger.error("Failed to start carbon tracking: %s", e)
            self._running = False
            raise

    def stop(self) -> CarbonMetrics:
        """Stop tracking and return final metrics."""
        if not self._running or self._tracker is None:
            logger.warning("Carbon tracker not running")
            return CarbonMetrics()

        try:
            # Stop the tracker - returns total emissions in kg CO2
            emissions_kg = self._tracker.stop()
            self._running = False

            metrics = self._extract_metrics(emissions_kg)

            logger.debug(
                "Carbon tracking stopped: %.4f kg CO2, %.4f kWh",
                metrics.co2_kg,
                metrics.energy_kwh,
            )

            return metrics

        except Exception as e:
            logger.error("Error stopping carbon tracker: %s", e)
            self._running = False
            return CarbonMetrics()
        finally:
            # Cleanup temp directory
            if self._temp_dir:
                try:
                    self._temp_dir.cleanup()
                except Exception:
                    pass
                self._temp_dir = None

    def get_current_metrics(self) -> CarbonMetrics:
        """Get current metrics without stopping tracking."""
        if not self._running or self._tracker is None:
            return CarbonMetrics()

        try:
            # CodeCarbon's flush() updates internal state and returns emissions
            # Note: This is a relatively expensive operation
            emissions_kg = self._tracker.flush()
            return self._extract_metrics(emissions_kg)
        except Exception as e:
            logger.debug("Error getting current carbon metrics: %s", e)
            # Return partial metrics based on duration
            return CarbonMetrics(
                duration_s=time.time() - self._start_time if self._start_time else 0.0,
            )

    def _extract_metrics(self, emissions_kg: float | None) -> CarbonMetrics:
        """Extract metrics from CodeCarbon tracker state.

        Args:
            emissions_kg: Total emissions returned by stop()/flush()

        Returns:
            CarbonMetrics with all available values
        """
        if self._tracker is None:
            return CarbonMetrics()

        duration_s = time.time() - self._start_time if self._start_time else 0.0

        # Get energy from tracker's internal state
        # CodeCarbon tracks energy in kWh - may return Energy object in newer versions
        try:
            energy_kwh = getattr(self._tracker, "_total_energy", None)
            if energy_kwh is None:
                # Try alternative attribute names
                energy_kwh = getattr(self._tracker, "final_emissions_data", {}).get(
                    "energy_consumed", 0.0
                )
            energy_kwh = to_float(energy_kwh)
        except Exception:
            energy_kwh = 0.0

        # Get power metrics - also may be Energy objects in newer versions
        try:
            cpu_power = to_float(getattr(self._tracker, "_total_cpu_power", None))
            gpu_power = to_float(getattr(self._tracker, "_total_gpu_power", None))
            ram_power = to_float(getattr(self._tracker, "_total_ram_power", None))
        except Exception:
            cpu_power = 0.0
            gpu_power = 0.0
            ram_power = 0.0

        # Get region
        region = ""
        try:
            region = getattr(self._tracker, "_conf", {}).get("country_iso_code", "")
            if not region:
                region = self._country_iso_code or ""
        except Exception:
            region = self._country_iso_code or ""

        return CarbonMetrics(
            co2_kg=to_float(emissions_kg),
            energy_kwh=energy_kwh,
            cpu_power_w=cpu_power,
            gpu_power_w=gpu_power,
            ram_power_w=ram_power,
            duration_s=duration_s,
            region=region,
            timestamp=time.time(),
        )
