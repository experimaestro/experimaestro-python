"""Zeus-based carbon tracker for energy measurement.

This module provides carbon tracking using the Zeus library, which supports:
- Apple Silicon Macs (via IOKit/IOReport through zeus-apple-silicon)
- NVIDIA GPUs (via NVML)
- AMD GPUs (via AMDSMI)
- Intel/AMD CPUs (via RAPL)

Note: On Apple Silicon, we use the zeus_apple_silicon API directly due to
a bug in zeus 0.13.x with AppleSiliconMeasurement instantiation.
"""

import logging
import platform
import time
import uuid

from experimaestro.carbon.base import BaseCarbonTracker, CarbonMetrics

logger = logging.getLogger(__name__)


def _is_apple_silicon() -> bool:
    """Check if running on Apple Silicon."""
    return platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64")


def is_zeus_available() -> bool:
    """Check if zeus is available and can measure energy."""
    # On Apple Silicon, check for zeus_apple_silicon directly
    if _is_apple_silicon():
        try:
            from zeus_apple_silicon import AppleEnergyMonitor  # noqa: F401

            return True
        except ImportError:
            return False

    # On other platforms, check for ZeusMonitor
    try:
        from zeus.monitor.energy import ZeusMonitor

        # Try to create a monitor to verify it works
        monitor = ZeusMonitor(gpu_indices=[], cpu_indices=[])
        del monitor
        return True
    except ImportError:
        return False
    except Exception as e:
        logger.debug("Zeus available but initialization failed: %s", e)
        return False


class ZeusTracker(BaseCarbonTracker):
    """Carbon tracker using Zeus library for energy measurement.

    Zeus provides unified energy measurement across multiple platforms:
    - Apple Silicon: CPU, GPU, DRAM, ANE via IOKit
    - NVIDIA GPUs: via NVML
    - AMD GPUs: via AMDSMI
    - Intel/AMD CPUs: via RAPL

    Note: Energy is measured in Joules (or millijoules for Apple Silicon).
    CO2 estimation requires a carbon intensity factor which varies by region.
    """

    # Default carbon intensity (gCO2/kWh) - world average
    DEFAULT_CARBON_INTENSITY = 450.0

    def __init__(
        self,
        *,
        country_iso_code: str | None = None,
        region: str | None = None,
        carbon_intensity_gco2_kwh: float | None = None,
        gpu_indices: list[int] | None = None,
        cpu_indices: list[int] | None = None,
    ):
        """Initialize the Zeus tracker.

        Args:
            country_iso_code: Country code for carbon intensity lookup
            region: Region (for reference)
            carbon_intensity_gco2_kwh: Carbon intensity in gCO2/kWh
            gpu_indices: GPU indices to monitor (None = all available)
            cpu_indices: CPU indices to monitor (None = all available)
        """
        self._country_iso_code = country_iso_code
        self._region = region
        self._carbon_intensity = (
            carbon_intensity_gco2_kwh or self._get_carbon_intensity(country_iso_code)
        )
        self._gpu_indices = gpu_indices
        self._cpu_indices = cpu_indices
        self._monitor = None
        self._window_label: str | None = None
        self._start_time: float | None = None
        self._use_apple_silicon_api = _is_apple_silicon()
        # Store initial cumulative energy for get_current_metrics
        self._initial_cumulative_energy = None

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

        return self.DEFAULT_CARBON_INTENSITY

    def start(self) -> None:
        """Start tracking energy consumption."""
        if self._running:
            logger.warning("Zeus tracker already running")
            return

        try:
            self._window_label = f"xpm_{uuid.uuid4().hex[:8]}"

            if self._use_apple_silicon_api:
                # Use zeus_apple_silicon directly (workaround for zeus bug)
                from zeus_apple_silicon import AppleEnergyMonitor

                self._monitor = AppleEnergyMonitor()
                # Store initial cumulative energy for delta calculation in get_current_metrics
                try:
                    self._initial_cumulative_energy = (
                        self._monitor.get_cumulative_energy()
                    )
                except Exception:
                    self._initial_cumulative_energy = None
                self._monitor.begin_window(self._window_label)
            else:
                # Use ZeusMonitor for other platforms
                from zeus.monitor.energy import ZeusMonitor

                self._monitor = ZeusMonitor(
                    gpu_indices=self._gpu_indices,
                    cpu_indices=self._cpu_indices,
                )
                self._monitor.begin_window(self._window_label)

            self._start_time = time.time()
            self._running = True

            logger.debug(
                "Zeus tracking started (apple_silicon=%s)", self._use_apple_silicon_api
            )

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
            measurement = self._monitor.end_window(self._window_label)
            self._running = False
            self._initial_cumulative_energy = None

            if self._use_apple_silicon_api:
                carbon_metrics = self._convert_apple_silicon_metrics(measurement)
            else:
                carbon_metrics = self._convert_measurement(measurement)

            logger.debug(
                "Zeus tracking stopped: %.6f kg CO2, %.6f kWh",
                carbon_metrics.co2_kg,
                carbon_metrics.energy_kwh,
            )

            return carbon_metrics

        except Exception as e:
            logger.error("Error stopping Zeus tracker: %s", e)
            self._running = False
            self._initial_cumulative_energy = None
            return CarbonMetrics()

    def get_current_metrics(self) -> CarbonMetrics:
        """Get current metrics without stopping tracking.

        Note: Zeus doesn't support getting intermediate measurements
        without ending the window.

        For Apple Silicon, we can use get_cumulative_energy().
        """
        if not self._running or self._monitor is None:
            return CarbonMetrics()

        duration_s = time.time() - self._start_time if self._start_time else 0.0

        if self._use_apple_silicon_api:
            try:
                # Apple Silicon API supports getting cumulative energy
                # We need to compute the delta from the initial reading
                current_metrics = self._monitor.get_cumulative_energy()
                return self._convert_apple_silicon_metrics_delta(
                    current_metrics, self._initial_cumulative_energy, duration_s
                )
            except Exception as e:
                logger.debug("Error getting current metrics: %s", e)

        # Fallback: return just duration
        return CarbonMetrics(
            duration_s=duration_s,
            region=self._country_iso_code or "",
        )

    def _convert_apple_silicon_metrics_delta(
        self, current_metrics, initial_metrics, duration_s: float | None = None
    ) -> CarbonMetrics:
        """Convert Apple Silicon metrics delta to CarbonMetrics.

        Computes the difference between current and initial cumulative readings.

        Args:
            current_metrics: Current AppleEnergyMetrics from get_cumulative_energy()
            initial_metrics: Initial AppleEnergyMetrics from start()
            duration_s: Optional duration override

        Returns:
            CarbonMetrics with converted values
        """
        if duration_s is None:
            duration_s = time.time() - self._start_time if self._start_time else 0.0

        # Get initial values (or 0 if not available)
        initial_cpu_mj = 0.0
        initial_gpu_mj = 0.0
        initial_dram_mj = 0.0
        initial_ane_mj = 0.0
        initial_gpu_sram_mj = 0.0

        if initial_metrics is not None:
            initial_cpu_mj = getattr(initial_metrics, "cpu_total_mj", None) or 0.0
            initial_gpu_mj = getattr(initial_metrics, "gpu_mj", None) or 0.0
            initial_dram_mj = getattr(initial_metrics, "dram_mj", None) or 0.0
            initial_ane_mj = getattr(initial_metrics, "ane_mj", None) or 0.0
            initial_gpu_sram_mj = getattr(initial_metrics, "gpu_sram_mj", None) or 0.0

        # Get current values
        current_cpu_mj = getattr(current_metrics, "cpu_total_mj", None) or 0.0
        current_gpu_mj = getattr(current_metrics, "gpu_mj", None) or 0.0
        current_dram_mj = getattr(current_metrics, "dram_mj", None) or 0.0
        current_ane_mj = getattr(current_metrics, "ane_mj", None) or 0.0
        current_gpu_sram_mj = getattr(current_metrics, "gpu_sram_mj", None) or 0.0

        # Compute delta (energy used since start)
        cpu_energy_mj = current_cpu_mj - initial_cpu_mj
        gpu_energy_mj = current_gpu_mj - initial_gpu_mj
        dram_energy_mj = current_dram_mj - initial_dram_mj
        ane_energy_mj = current_ane_mj - initial_ane_mj
        gpu_sram_mj = current_gpu_sram_mj - initial_gpu_sram_mj

        total_energy_mj = (
            cpu_energy_mj + gpu_energy_mj + dram_energy_mj + ane_energy_mj + gpu_sram_mj
        )

        # Convert mJ to kWh: 1 kWh = 3.6e9 mJ
        energy_kwh = total_energy_mj / 3.6e9

        # Calculate CO2: co2_kg = energy_kwh * carbon_intensity_gco2_kwh / 1000
        co2_kg = energy_kwh * self._carbon_intensity / 1000.0

        # Calculate average power (watts)
        # energy_mj / duration_s = mW, so divide by 1000 for W
        cpu_power_w = cpu_energy_mj / duration_s / 1000.0 if duration_s > 0 else 0.0
        gpu_power_w = gpu_energy_mj / duration_s / 1000.0 if duration_s > 0 else 0.0
        ram_power_w = dram_energy_mj / duration_s / 1000.0 if duration_s > 0 else 0.0

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

    def _convert_apple_silicon_metrics(
        self, metrics, duration_s: float | None = None
    ) -> CarbonMetrics:
        """Convert Apple Silicon metrics to CarbonMetrics.

        Args:
            metrics: AppleEnergyMetrics from zeus_apple_silicon
            duration_s: Optional duration override

        Returns:
            CarbonMetrics with converted values
        """
        if duration_s is None:
            duration_s = time.time() - self._start_time if self._start_time else 0.0

        # Sum all energy components (in millijoules)
        total_energy_mj = 0.0

        # CPU energy (total across all cores)
        cpu_energy_mj = getattr(metrics, "cpu_total_mj", None) or 0.0
        total_energy_mj += cpu_energy_mj

        # GPU energy
        gpu_energy_mj = getattr(metrics, "gpu_mj", None) or 0.0
        total_energy_mj += gpu_energy_mj

        # DRAM energy
        dram_energy_mj = getattr(metrics, "dram_mj", None) or 0.0
        total_energy_mj += dram_energy_mj

        # ANE (Apple Neural Engine) energy
        ane_energy_mj = getattr(metrics, "ane_mj", None) or 0.0
        total_energy_mj += ane_energy_mj

        # GPU SRAM energy
        gpu_sram_mj = getattr(metrics, "gpu_sram_mj", None) or 0.0
        total_energy_mj += gpu_sram_mj

        # Convert mJ to kWh: 1 kWh = 3.6e9 mJ
        energy_kwh = total_energy_mj / 3.6e9

        # Calculate CO2: co2_kg = energy_kwh * carbon_intensity_gco2_kwh / 1000
        co2_kg = energy_kwh * self._carbon_intensity / 1000.0

        # Calculate average power (watts)
        # energy_mj / duration_s = mW, so divide by 1000 for W
        cpu_power_w = cpu_energy_mj / duration_s / 1000.0 if duration_s > 0 else 0.0
        gpu_power_w = gpu_energy_mj / duration_s / 1000.0 if duration_s > 0 else 0.0
        ram_power_w = dram_energy_mj / duration_s / 1000.0 if duration_s > 0 else 0.0

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

    def _convert_measurement(self, measurement) -> CarbonMetrics:
        """Convert Zeus Measurement to CarbonMetrics.

        Args:
            measurement: Measurement object from ZeusMonitor.end_window()

        Returns:
            CarbonMetrics with converted values
        """
        duration_s = (
            measurement.time
            if hasattr(measurement, "time")
            else (time.time() - self._start_time if self._start_time else 0.0)
        )

        # Collect energy from all sources (in Joules)
        total_energy_j = 0.0
        cpu_energy_j = 0.0
        gpu_energy_j = 0.0
        ram_energy_j = 0.0

        # GPU energy (dict mapping GPU index to Joules)
        if hasattr(measurement, "gpu_energy") and measurement.gpu_energy:
            gpu_energy_j = sum(measurement.gpu_energy.values())
            total_energy_j += gpu_energy_j

        # CPU energy (dict mapping CPU index to Joules)
        if hasattr(measurement, "cpu_energy") and measurement.cpu_energy:
            cpu_energy_j = sum(measurement.cpu_energy.values())
            total_energy_j += cpu_energy_j

        # DRAM energy (dict mapping CPU index to Joules)
        if hasattr(measurement, "dram_energy") and measurement.dram_energy:
            ram_energy_j = sum(measurement.dram_energy.values())
            total_energy_j += ram_energy_j

        # SoC energy (Apple Silicon) - contains CPU, GPU, DRAM, ANE
        if hasattr(measurement, "soc_energy") and measurement.soc_energy:
            soc = measurement.soc_energy
            # SoC metrics are in Joules
            if hasattr(soc, "cpu_energy") and soc.cpu_energy is not None:
                cpu_energy_j += soc.cpu_energy
                total_energy_j += soc.cpu_energy
            if hasattr(soc, "gpu_energy") and soc.gpu_energy is not None:
                gpu_energy_j += soc.gpu_energy
                total_energy_j += soc.gpu_energy
            if hasattr(soc, "dram_energy") and soc.dram_energy is not None:
                ram_energy_j += soc.dram_energy
                total_energy_j += soc.dram_energy
            if hasattr(soc, "ane_energy") and soc.ane_energy is not None:
                total_energy_j += soc.ane_energy

        # Convert Joules to kWh: 1 kWh = 3.6e6 J
        energy_kwh = total_energy_j / 3.6e6

        # Calculate CO2: co2_kg = energy_kwh * carbon_intensity_gco2_kwh / 1000
        co2_kg = energy_kwh * self._carbon_intensity / 1000.0

        # Calculate average power (watts) = Energy (J) / Time (s)
        cpu_power_w = cpu_energy_j / duration_s if duration_s > 0 else 0.0
        gpu_power_w = gpu_energy_j / duration_s if duration_s > 0 else 0.0
        ram_power_w = ram_energy_j / duration_s if duration_s > 0 else 0.0

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
