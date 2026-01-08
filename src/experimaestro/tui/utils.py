"""Utility functions for the TUI"""


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string"""
    if seconds < 0:
        return "-"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        return f"{seconds // 60}m {seconds % 60}s"
    elif seconds < 86400:
        return f"{seconds // 3600}h {(seconds % 3600) // 60}m"
    else:
        return f"{seconds // 86400}d {(seconds % 86400) // 3600}h"


def get_status_icon(status: str, failure_reason=None, transient=None):
    """Get status icon for a job state.

    Args:
        status: Job state name (e.g., "done", "error", "running")
        failure_reason: Optional JobFailureStatus enum for error states
        transient: Optional TransientMode enum

    Returns:
        Status icon string
    """
    if status == "done":
        return "✅"
    elif status == "error":
        # Show different icons for different failure types
        if failure_reason is not None:
            from experimaestro.scheduler.interfaces import JobFailureStatus

            if failure_reason == JobFailureStatus.DEPENDENCY:
                return "🔗"  # Dependency failed
            elif failure_reason == JobFailureStatus.TIMEOUT:
                return "⏱"  # Timeout
            elif failure_reason == JobFailureStatus.MEMORY:
                return "💾"  # Memory issue
            # FAILED or unknown - use default error icon
        return "❌"
    elif status == "running":
        return "▶"
    elif status == "waiting":
        return "⌛"  # Waiting for dependencies
    elif status == "unscheduled" and transient is not None and transient.is_transient:
        # Transient job that was skipped (not needed)
        return "💤"  # Sleeping - dormant, not activated
    else:
        # phantom, unscheduled or unknown
        return "👻"
