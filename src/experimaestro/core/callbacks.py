from collections import defaultdict
import threading
from typing import Callable, ClassVar, Optional
from experimaestro.core.objects import ConfigInformation
from experimaestro.scheduler import Listener, Job, JobState, experiment


class TaskEventListener(Listener):
    INSTANCE: ClassVar[Optional["TaskEventListener"]] = None
    """The general instance"""

    def __init__(self):
        self.lock = threading.Lock()
        self.experiments: set[int] = set()

        self._on_completed: defaultdict[int, Callable] = defaultdict(list)

    @staticmethod
    def connect(xp: experiment):
        _self = TaskEventListener.instance()
        with _self.lock:
            if id(xp) not in _self.experiments:
                _self.experiments.add(id(xp))
                xp.scheduler.addlistener(_self)

    @staticmethod
    def instance():
        if TaskEventListener.INSTANCE is None:
            TaskEventListener.INSTANCE = TaskEventListener()

        return TaskEventListener.INSTANCE

    def job_state(self, job: Job):
        if job.state == JobState.DONE:
            with self.lock:
                for callback in self._on_completed.get(id(job.config.__xpm__), []):
                    callback()

    @staticmethod
    def on_completed(
        config_information: ConfigInformation, callback: Callable[[], None]
    ):
        instance = TaskEventListener.instance()

        with instance.lock:
            instance._on_completed[id(config_information)].append(callback)

            if (
                config_information.job is not None
                and config_information.job == JobState.DONE
            ):
                callback()
