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


def get_status_icon(status: str, failure_reason=None):
    """Get status icon for a job state.

    Args:
        status: Job state name (e.g., "done", "error", "running", "transient")
        failure_reason: Optional JobFailureStatus enum for error states

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
            elif failure_reason == JobFailureStatus.REJECTED_TIMELIMIT:
                return "🚫"  # Rejected (time limit)
            elif failure_reason == JobFailureStatus.REJECTED_OTHER:
                return "🚫"  # Rejected (other reason)
            # FAILED or unknown - use default error icon
        return "❌"
    elif status == "running":
        return "▶"
    elif status == "scheduled":
        return "🕐"  # Scheduled (e.g., in SLURM queue)
    elif status == "waiting":
        return "⌛"  # Waiting for dependencies
    elif status == "transient":
        return "💤"  # Sleeping - dormant, not activated
    else:
        # phantom, unscheduled or unknown
        return "👻"
