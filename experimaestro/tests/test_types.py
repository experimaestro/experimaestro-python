# --- Task and types definitions

import pytest
import os
from pathlib import Path
import logging

from experimaestro import Workspace

from .utils import TemporaryExperiment
from experimaestro.scheduler import JobState

from .definitions_types import *


def test_simple():
    with TemporaryExperiment("simple") as xp:
        assert IntegerTask(value=5).submit().__xpm__.job.wait() == JobState.DONE
        assert FloatTask(value=5.1).submit().__xpm__.job.wait() == JobState.DONE
