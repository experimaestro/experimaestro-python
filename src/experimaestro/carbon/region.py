"""Region detection and caching for carbon tracking."""

import json
import logging
import time
from pathlib import Path
from typing import TypedDict

logger = logging.getLogger(__name__)

# Cache file location
CACHE_DIR = Path.home() / ".cache" / "experimaestro"
CACHE_FILE = CACHE_DIR / "carbon-region.json"

# Cache validity period (7 days)
CACHE_TTL_SECONDS = 7 * 24 * 60 * 60


class RegionInfo(TypedDict):
    """Region information for carbon tracking."""

    country_iso_code: str
    country_name: str
    region: str
    detected_at: float


# Country ISO codes to names mapping (common ones)
COUNTRY_NAMES = {
    "USA": "United States",
    "GBR": "United Kingdom",
    "FRA": "France",
    "DEU": "Germany",
    "CAN": "Canada",
    "AUS": "Australia",
    "JPN": "Japan",
    "CHN": "China",
    "IND": "India",
    "BRA": "Brazil",
    "NLD": "Netherlands",
    "CHE": "Switzerland",
    "SWE": "Sweden",
    "NOR": "Norway",
    "DNK": "Denmark",
    "FIN": "Finland",
    "ITA": "Italy",
    "ESP": "Spain",
    "PRT": "Portugal",
    "BEL": "Belgium",
    "AUT": "Austria",
    "IRL": "Ireland",
    "POL": "Poland",
    "CZE": "Czech Republic",
    "GRC": "Greece",
    "RUS": "Russia",
    "KOR": "South Korea",
    "SGP": "Singapore",
    "NZL": "New Zealand",
    "ZAF": "South Africa",
    "MEX": "Mexico",
    "ARG": "Argentina",
}


def _detect_region() -> RegionInfo | None:
    """Detect region using CodeCarbon's detection.

    Returns:
        RegionInfo dict or None if detection failed.
    """
    try:
        # Import codecarbon's geo detection
        from codecarbon.external.geography import GeoMetadata
        from codecarbon.input import DataSource

        # Get the geo.js URL from codecarbon
        ds = DataSource()
        geo_url = ds.geo_js_url

        # Use CodeCarbon's geo detection from IP
        geo = GeoMetadata.from_geo_js(geo_url)
        country_code = geo.country_iso_code

        if not country_code:
            logger.warning("Could not detect country for carbon tracking")
            return None

        # Use the country name from codecarbon, or fall back to our mapping
        country_name = geo.country_name or COUNTRY_NAMES.get(country_code, country_code)

        # Get region if available
        region = geo.region or ""

        return RegionInfo(
            country_iso_code=country_code,
            country_name=country_name,
            region=region,
            detected_at=time.time(),
        )

    except ImportError:
        logger.debug("CodeCarbon not installed, cannot detect region")
        return None
    except Exception as e:
        logger.warning("Failed to detect region: %s", e)
        return None


def _load_cache() -> RegionInfo | None:
    """Load cached region info if valid.

    Returns:
        Cached RegionInfo or None if cache invalid/missing.
    """
    if not CACHE_FILE.exists():
        return None

    try:
        with CACHE_FILE.open() as f:
            data = json.load(f)

        # Check cache validity
        detected_at = data.get("detected_at", 0)
        if time.time() - detected_at > CACHE_TTL_SECONDS:
            logger.debug("Region cache expired")
            return None

        return RegionInfo(
            country_iso_code=data["country_iso_code"],
            country_name=data["country_name"],
            region=data.get("region", ""),
            detected_at=detected_at,
        )

    except Exception as e:
        logger.debug("Failed to load region cache: %s", e)
        return None


def _save_cache(info: RegionInfo) -> None:
    """Save region info to cache.

    Args:
        info: RegionInfo to cache.
    """
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with CACHE_FILE.open("w") as f:
            json.dump(info, f, indent=2)
        logger.debug("Saved region cache: %s", info["country_iso_code"])
    except Exception as e:
        logger.warning("Failed to save region cache: %s", e)


def get_cached_region_info() -> RegionInfo | None:
    """Get region info, using cache if available.

    This first checks for cached region info. If not found or expired,
    it performs fresh detection and caches the result.

    Returns:
        RegionInfo dict or None if detection failed.
    """
    # Try cache first
    cached = _load_cache()
    if cached is not None:
        logger.debug("Using cached region: %s", cached["country_iso_code"])
        return cached

    # Detect fresh
    info = _detect_region()
    if info is not None:
        _save_cache(info)

    return info


def clear_region_cache() -> None:
    """Clear the region cache file."""
    try:
        if CACHE_FILE.exists():
            CACHE_FILE.unlink()
            logger.debug("Cleared region cache")
    except Exception as e:
        logger.warning("Failed to clear region cache: %s", e)


def format_region_display(info: RegionInfo | None) -> str:
    """Format region info for display.

    Args:
        info: RegionInfo to format.

    Returns:
        Human-readable region string like "FRA (France)"
    """
    if info is None:
        return "Unknown"

    code = info["country_iso_code"]
    name = info["country_name"]

    if info["region"]:
        return f"{code} ({name}, {info['region']})"
    return f"{code} ({name})"
