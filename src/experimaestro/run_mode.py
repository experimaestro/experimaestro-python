"""Run modes of an experiment

Kept out of :mod:`experimaestro.scheduler` (which re-exports it) so that the
command line can use it without loading the scheduler.
"""

from enum import Enum


class RunMode(str, Enum):
    NORMAL = "normal"
    """Normal run"""

    GENERATE_ONLY = "generate"
    """Do not run, but generate the params.json file"""

    DRY_RUN = "dry-run"
    """Do not run"""

    PREPARE = "prepare"
    """Only run discovered Prepare configs; skip Task execution"""
