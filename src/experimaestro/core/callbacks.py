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

        self._on_completed: defaultdict[int, Callable] = defaultdict(list)

    @staticmethod
    def connect(xp: experiment):
        _self = TaskEventListener.instance()
        # Always call addlistener — it uses a set internally so duplicates
        # are harmless, and listeners may have been cleared between tests
        # or experiment runs (e.g., scheduler.clear_listeners()).
        xp.scheduler.addlistener(_self)

    @staticmethod
    def instance():
        if TaskEventListener.INSTANCE is None:
            TaskEventListener.INSTANCE = TaskEventListener()

        return TaskEventListener.INSTANCE

    def on_job_state_changed(self, job: Job):
        if job.scheduler_state == JobState.DONE:
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
                and config_information.job.scheduler_state == JobState.DONE
            ):
                callback()
