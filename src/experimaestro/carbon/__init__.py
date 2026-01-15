"""Environmental impact tracking for experimaestro.

This module provides carbon tracking capabilities using platform-specific
backends:
- Linux/Windows: CodeCarbon (Intel RAPL, NVIDIA GPU)
- macOS Apple Silicon: zeus-apple-silicon (IOKit API)

Usage:
    Carbon tracking is enabled by default when the appropriate library is installed.
    Install with: pip install experimaestro[carbon]

    To disable tracking:
    - Use experiment(..., no_environmental_impact=True)
    - Or set carbon.enabled: false in settings.yaml

Platform support:
    - Linux with Intel CPUs: CodeCarbon (RAPL interface)
    - Linux/Windows with NVIDIA GPUs: CodeCarbon
    - macOS Apple Silicon: zeus-apple-silicon (IOKit/IOReport API)
    - macOS Intel: Not supported (no RAPL access)
"""

import logging
import platform

from experimaestro.carbon.base import (
    BaseCarbonTracker,
    CarbonAggregateData,
    CarbonImpactData,
    CarbonMetrics,
    CarbonTracker,
    NullCarbonTracker,
)

logger = logging.getLogger(__name__)

__all__ = [
    "BaseCarbonTracker",
    "CarbonAggregateData",
    "CarbonImpactData",
    "CarbonMetrics",
    "CarbonTracker",
    "NullCarbonTracker",
    "create_tracker",
    "is_available",
    "get_backend_name",
    "get_region_info",
]

# Track if we've already shown the platform warning
_platform_warning_shown = False


def _is_macos_apple_silicon() -> bool:
    """Check if running on macOS with Apple Silicon."""
    return platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64")


def _is_zeus_available() -> bool:
    """Check if zeus is available for energy measurement."""
    try:
        from experimaestro.carbon.zeus_tracker import is_zeus_available

        return is_zeus_available()
    except ImportError:
        return False


def _is_codecarbon_available() -> bool:
    """Check if CodeCarbon is available and platform supports it."""
    try:
        import codecarbon  # noqa: F401
    except ImportError:
        return False

    system = platform.system()
    machine = platform.machine()

    # macOS doesn't support CodeCarbon (no RAPL, no NVIDIA)
    if system == "Darwin":
        return False

    # ARM Linux without NVIDIA GPU typically can't use CodeCarbon
    if system == "Linux" and machine in ("aarch64", "arm64"):
        try:
            import pynvml

            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            pynvml.nvmlShutdown()
            return count > 0
        except Exception:
            return False

    # Linux on x86_64 or Windows - assume supported
    return True


def is_available() -> bool:
    """Check if carbon tracking is available and functional.

    Returns:
        True if a suitable carbon tracking backend is available.
    """
    # On macOS Apple Silicon, check for Zeus
    if _is_macos_apple_silicon():
        return _is_zeus_available()

    # On other platforms, check for CodeCarbon
    return _is_codecarbon_available()


def get_backend_name() -> str | None:
    """Get the name of the available carbon tracking backend.

    Returns:
        Backend name ("zeus", "codecarbon") or None if unavailable.
    """
    if _is_macos_apple_silicon() and _is_zeus_available():
        return "zeus"
    if _is_codecarbon_available():
        return "codecarbon"
    return None


def get_unavailable_reason() -> str | None:
    """Get the reason why carbon tracking is not available.

    Returns:
        Human-readable reason string, or None if tracking is available.
    """
    if is_available():
        return None

    system = platform.system()
    machine = platform.machine()

    if system == "Darwin":
        if machine in ("arm64", "aarch64"):
            return (
                "zeus[apple] not installed. "
                "Install with: pip install experimaestro[carbon]"
            )
        else:
            return (
                "macOS on Intel does not support power measurement. "
                "Only Apple Silicon Macs are supported via Zeus."
            )

    if system == "Linux" and machine in ("aarch64", "arm64"):
        return (
            "Linux on ARM without NVIDIA GPU does not support power measurement. "
            "CodeCarbon requires Intel RAPL or NVIDIA GPU interfaces."
        )

    # Linux x86_64 or Windows without CodeCarbon
    return "CodeCarbon not installed. Install with: pip install experimaestro[carbon]"


def create_tracker(
    *,
    country_iso_code: str | None = None,
    region: str | None = None,
    report_interval_s: float = 60.0,
    show_warning: bool = True,
) -> BaseCarbonTracker:
    """Create a carbon tracker instance.

    Args:
        country_iso_code: Override detected country (ISO 3166-1 alpha-3)
        region: Override detected region
        report_interval_s: How often to update internal metrics
        show_warning: Whether to show warning if not available

    Returns:
        A CarbonTracker instance appropriate for the platform,
        or NullCarbonTracker if unavailable.
    """
    global _platform_warning_shown

    # On macOS Apple Silicon, use Zeus
    if _is_macos_apple_silicon():
        if _is_zeus_available():
            from experimaestro.carbon.zeus_tracker import ZeusTracker

            return ZeusTracker(
                country_iso_code=country_iso_code,
                region=region,
            )
        else:
            if show_warning and not _platform_warning_shown:
                logger.warning(
                    "Carbon tracking disabled: zeus[apple] not installed. "
                    "Install with: pip install experimaestro[carbon]"
                )
                _platform_warning_shown = True
            return NullCarbonTracker()

    # On other platforms, use CodeCarbon
    if _is_codecarbon_available():
        from experimaestro.carbon.codecarbon import CodeCarbonTracker

        return CodeCarbonTracker(
            country_iso_code=country_iso_code,
            region=region,
        )

    # Not available
    if show_warning and not _platform_warning_shown:
        reason = get_unavailable_reason()
        logger.warning("Carbon tracking disabled: %s", reason or "Not available")
        _platform_warning_shown = True

    return NullCarbonTracker()


def get_region_info() -> dict | None:
    """Get detected region information.

    Returns:
        Dictionary with 'country_iso_code', 'country_name', 'region' keys,
        or None if detection failed.
    """
    # Region detection uses CodeCarbon's geo detection on all platforms
    # Fall back to cached region info
    try:
        from experimaestro.carbon.region import get_cached_region_info

        return get_cached_region_info()
    except Exception:
        return None
