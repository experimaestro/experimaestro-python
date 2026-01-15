"""Zeus-based carbon tracker for Apple Silicon Macs."""

import logging
import time
import uuid

from experimaestro.carbon.base import BaseCarbonTracker, CarbonMetrics

logger = logging.getLogger(__name__)


def is_zeus_available() -> bool:
    """Check if zeus-apple-silicon is available."""
    try:
        from zeus_apple_silicon import AppleEnergyMonitor  # noqa: F401

        return True
    except ImportError:
        return False


class ZeusAppleSiliconTracker(BaseCarbonTracker):
    """Carbon tracker using zeus-apple-silicon for Apple Silicon Macs.

    This uses Apple's IOKit/IOReport API to measure energy consumption
    across CPU, GPU, DRAM, and ANE subsystems.

    Note: Energy is measured in millijoules. CO2 estimation requires
    a carbon intensity factor which varies by region.
    """

    # Default carbon intensity (gCO2/kWh) - France average
    # Can be overridden via settings
    DEFAULT_CARBON_INTENSITY = 56.0  # gCO2/kWh for France

    def __init__(
        self,
        *,
        country_iso_code: str | None = None,
        region: str | None = None,
        carbon_intensity_gco2_kwh: float | None = None,
    ):
        """Initialize the Zeus Apple Silicon tracker.

        Args:
            country_iso_code: Country code (for reference, not used for intensity)
            region: Region (for reference)
            carbon_intensity_gco2_kwh: Carbon intensity in gCO2/kWh. If not provided,
                uses a default value or attempts to look up by country.
        """
        self._country_iso_code = country_iso_code
        self._region = region
        self._carbon_intensity = (
            carbon_intensity_gco2_kwh or self._get_carbon_intensity(country_iso_code)
        )
        self._monitor = None
        self._window_label: str | None = None
        self._start_time: float | None = None

    def _get_carbon_intensity(self, country_iso_code: str | None) -> float:
        """Get carbon intensity for a country.

        Args:
            country_iso_code: ISO 3166-1 alpha-3 country code

        Returns:
            Carbon intensity in gCO2/kWh
        """
        # Carbon intensity by country (approximate averages, gCO2/kWh)
        # Source: https://app.electricitymaps.com/
        INTENSITY_BY_COUNTRY = {
            "FRA": 56,  # France (nuclear)
            "SWE": 41,  # Sweden (hydro/nuclear)
            "NOR": 26,  # Norway (hydro)
            "CHE": 48,  # Switzerland
            "CAN": 130,  # Canada (varies by province)
            "USA": 390,  # USA (average)
            "GBR": 230,  # UK
            "DEU": 385,  # Germany
            "AUS": 510,  # Australia
            "CHN": 540,  # China
            "IND": 630,  # India
            "POL": 650,  # Poland (coal)
        }

        if country_iso_code and country_iso_code in INTENSITY_BY_COUNTRY:
            return float(INTENSITY_BY_COUNTRY[country_iso_code])

        # Default to world average
        return 450.0  # World average ~450 gCO2/kWh

    def start(self) -> None:
        """Start tracking energy consumption."""
        if self._running:
            logger.warning("Zeus tracker already running")
            return

        try:
            from zeus_apple_silicon import AppleEnergyMonitor

            self._monitor = AppleEnergyMonitor()
            self._window_label = f"xpm_{uuid.uuid4().hex[:8]}"
            self._monitor.begin_window(self._window_label)
            self._start_time = time.time()
            self._running = True

            logger.debug("Zeus Apple Silicon tracking started")

        except Exception as e:
            logger.error("Failed to start Zeus tracking: %s", e)
            self._running = False
            raise

    def stop(self) -> CarbonMetrics:
        """Stop tracking and return final metrics."""
        if not self._running or self._monitor is None or self._window_label is None:
            logger.warning("Zeus tracker not running")
            return CarbonMetrics()

        try:
            metrics = self._monitor.end_window(self._window_label)
            self._running = False

            carbon_metrics = self._convert_metrics(metrics)

            logger.debug(
                "Zeus tracking stopped: %.4f kg CO2, %.4f kWh",
                carbon_metrics.co2_kg,
                carbon_metrics.energy_kwh,
            )

            return carbon_metrics

        except Exception as e:
            logger.error("Error stopping Zeus tracker: %s", e)
            self._running = False
            return CarbonMetrics()

    def get_current_metrics(self) -> CarbonMetrics:
        """Get current metrics without stopping tracking."""
        if not self._running or self._monitor is None:
            return CarbonMetrics()

        try:
            # Get cumulative energy since start
            metrics = self._monitor.get_cumulative_energy()
            return self._convert_metrics(metrics)
        except Exception as e:
            logger.debug("Error getting current Zeus metrics: %s", e)
            return CarbonMetrics(
                duration_s=time.time() - self._start_time if self._start_time else 0.0,
            )

    def _convert_metrics(self, metrics) -> CarbonMetrics:
        """Convert Zeus metrics to CarbonMetrics.

        Args:
            metrics: AppleEnergyMetrics from zeus-apple-silicon

        Returns:
            CarbonMetrics with converted values
        """
        duration_s = time.time() - self._start_time if self._start_time else 0.0

        # Sum all energy components (in millijoules)
        # Zeus API uses: cpu_total_mj, gpu_mj, dram_mj, ane_mj
        total_energy_mj = 0.0

        # CPU energy (total across all cores)
        cpu_energy_mj = getattr(metrics, "cpu_total_mj", None)
        if cpu_energy_mj is not None:
            total_energy_mj += cpu_energy_mj

        # GPU energy
        gpu_energy_mj = getattr(metrics, "gpu_mj", None)
        if gpu_energy_mj is not None:
            total_energy_mj += gpu_energy_mj

        # DRAM energy
        dram_energy_mj = getattr(metrics, "dram_mj", None)
        if dram_energy_mj is not None:
            total_energy_mj += dram_energy_mj

        # ANE (Apple Neural Engine) energy
        ane_energy_mj = getattr(metrics, "ane_mj", None)
        if ane_energy_mj is not None:
            total_energy_mj += ane_energy_mj

        # GPU SRAM energy (if available)
        gpu_sram_mj = getattr(metrics, "gpu_sram_mj", None)
        if gpu_sram_mj is not None:
            total_energy_mj += gpu_sram_mj

        # Convert mJ to kWh: 1 kWh = 3.6e9 mJ
        energy_kwh = total_energy_mj / 3.6e9

        # Calculate CO2 from energy and carbon intensity
        # co2_kg = energy_kwh * carbon_intensity_gco2_kwh / 1000
        co2_kg = energy_kwh * self._carbon_intensity / 1000.0

        # Calculate average power (watts)
        # Power = Energy / Time
        # energy_mj / duration_s = mW, so divide by 1000 for W
        cpu_power_w = 0.0
        gpu_power_w = 0.0
        ram_power_w = 0.0

        if duration_s > 0:
            if cpu_energy_mj is not None:
                cpu_power_w = cpu_energy_mj / duration_s / 1000.0
            if gpu_energy_mj is not None:
                gpu_power_w = gpu_energy_mj / duration_s / 1000.0
            if dram_energy_mj is not None:
                ram_power_w = dram_energy_mj / duration_s / 1000.0

        return CarbonMetrics(
            co2_kg=co2_kg,
            energy_kwh=energy_kwh,
            cpu_power_w=cpu_power_w,
            gpu_power_w=gpu_power_w,
            ram_power_w=ram_power_w,
            duration_s=duration_s,
            region=self._country_iso_code or "",
            timestamp=time.time(),
        )
