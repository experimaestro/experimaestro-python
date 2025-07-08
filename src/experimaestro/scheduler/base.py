from collections import ChainMap
from functools import cached_property
import itertools
import logging
import os
from pathlib import Path
from shutil import rmtree
import threading
import time
from typing import (
    Any,
    ClassVar,
    Iterator,
    List,
    Optional,
    Set,
    TypeVar,
    Union,
    TYPE_CHECKING,
)
import enum
import signal
import asyncio
from experimaestro.exceptions import HandledException
from experimaestro.notifications import LevelInformation, Reporter
from typing import Dict
from experimaestro.scheduler.services import Service
from experimaestro.settings import WorkspaceSettings, get_settings


from experimaestro.core.objects import Config, ConfigWalkContext, WatchedOutput
from experimaestro.utils import logger
from experimaestro.locking import Locks, LockError, Lock
from experimaestro.utils.asyncio import asyncThreadcheck
from .workspace import RunMode, Workspace
from .dependencies import Dependency, DependencyStatus, Resource
import concurrent.futures


if TYPE_CHECKING:
    from experimaestro.connectors import Process
    from experimaestro.launchers import Launcher


class Listener:
    def job_submitted(self, job):
        pass

    def job_state(self, job):
        pass

    def service_add(self, service: Service):
        """Notify when a new service is added"""
        pass
