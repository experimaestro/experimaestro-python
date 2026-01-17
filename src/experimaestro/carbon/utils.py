"""Formatting utilities for carbon metrics display."""


def format_co2(grams: float) -> str:
    """Format CO2 with auto-scaling: g for <1000, kg otherwise.

    Args:
        grams: CO2 equivalent in grams.

    Returns:
        Formatted string like "45.2g" or "1.23kg"
    """
    if grams < 0:
        return "-"
    if grams < 1:
        return f"{grams:.3f}g"
    if grams < 10:
        return f"{grams:.2f}g"
    if grams < 1000:
        return f"{grams:.1f}g"
    return f"{grams / 1000:.2f}kg"


def format_co2_kg(kg: float) -> str:
    """Format CO2 from kg value with auto-scaling.

    Args:
        kg: CO2 equivalent in kilograms.

    Returns:
        Formatted string like "45.2g" or "1.23kg"
    """
    return format_co2(kg * 1000)


def format_energy(wh: float) -> str:
    """Format energy with auto-scaling: Wh for <1000, kWh otherwise.

    Args:
        wh: Energy in watt-hours.

    Returns:
        Formatted string like "45.2Wh" or "1.23kWh"
    """
    if wh < 0:
        return "-"
    if wh < 1:
        return f"{wh:.3f}Wh"
    if wh < 10:
        return f"{wh:.2f}Wh"
    if wh < 1000:
        return f"{wh:.1f}Wh"
    return f"{wh / 1000:.2f}kWh"


def format_energy_kwh(kwh: float | dict) -> str:
    """Format energy from kWh value with auto-scaling.

    Args:
        kwh: Energy in kilowatt-hours. May be a dict with 'kWh' key
            from older codecarbon serialization.

    Returns:
        Formatted string like "45.2Wh" or "1.23kWh"
    """
    # Handle Energy object or dict from codecarbon
    if isinstance(kwh, dict):
        kwh = kwh.get("kWh", 0.0)
    elif hasattr(kwh, "kWh"):
        kwh = kwh.kWh
    return format_energy(kwh * 1000)


def format_power(watts: float) -> str:
    """Format power in watts.

    Args:
        watts: Power in watts.

    Returns:
        Formatted string like "45.2W"
    """
    if watts < 0:
        return "-"
    if watts < 1:
        return f"{watts:.3f}W"
    if watts < 10:
        return f"{watts:.2f}W"
    return f"{watts:.1f}W"


def format_carbon_summary(co2_kg: float, energy_kwh: float, duration_s: float) -> str:
    """Format a summary of carbon metrics.

    Args:
        co2_kg: CO2 equivalent in kg.
        energy_kwh: Energy in kWh.
        duration_s: Duration in seconds.

    Returns:
        Formatted summary string.
    """
    co2_str = format_co2_kg(co2_kg)
    energy_str = format_energy_kwh(energy_kwh)

    # Format duration
    if duration_s < 60:
        duration_str = f"{duration_s:.0f}s"
    elif duration_s < 3600:
        minutes = duration_s / 60
        duration_str = f"{minutes:.1f}m"
    else:
        hours = duration_s / 3600
        duration_str = f"{hours:.1f}h"

    return f"{co2_str} CO2 | {energy_str} | {duration_str}"
