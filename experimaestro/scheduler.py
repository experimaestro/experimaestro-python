from pathlib import Path
import threading
import re
import time
from typing import Optional, Set
import enum

from .workspace import Workspace
from experimaestro.launchers import Launcher
from .api import PyObject
from .utils import logger
from .dependencies import LockError, Dependency

class JobState(enum.Enum):
    WAITING = 0
    READY = 1
    RUNNING = 2
    DONE = 3
    ERROR = 4

class Job():

    """Context of a job"""
    def __init__(self, parameters: PyObject, *, workspace:Workspace = None, launcher:Launcher = None):
        self.workspace = workspace or Workspace.CURRENT
        assert self.workspace is not None, "No experiment has been defined"

        self.launcher = launcher or self.workspace.launcher
        assert self.launcher is not None, "No default launcher in workspace %s" % workspace

        self.type = parameters.__class__.__xpm__
        self.name = str(self.type.typename).rsplit(".", 1)[-1]

        self.scheduler:Optional["Scheduler"] = None
        self.parameters = parameters   
        self.starttime:Optional[float] = None
        self.state:JobState = JobState.WAITING

        # Dependencies
        self.dependencies:Set[Dependency] = set() # as target
        self.dependents:Set[Dependency] = set() # as source

        # Process
        self.process = None

    @property
    def ready(self):
        return self.state == JobState.READY

    @property
    def jobpath(self):
        return self.workspace.jobspath  / str(self.type.typename) / self.identifier

    @property
    def identifier(self):
        return self.parameters.__xpm__.identifier.hex()

    def run(self, jobLock, locks):
        """Actually run the code"""
        raise NotImplementedError()

    def wait(self):
        raise NotImplementedError()

    @property
    def donepath(self):
        return self.jobpath / ("%s.done" % self.name)

    @property
    def lockpath(self):
        return self.jobpath / ("%s.lock" % self.name)

    @property
    def startlockpath(self):
        return self.jobpath / ("%s.startlock" % self.name)

    @property
    def codepath(self):
        return self.jobpath / ("%s.code" % self.name)

    @property
    def pidpath(self):
        return self.jobpath / ("%s.pid" % self.name)

    @property
    def stdout(self) -> Path:
        return self.jobpath / ("%s.out" % self.name)

    @property
    def stderr(self) -> Path:
        return self.jobpath / ("%s.err" % self.name)


class Listener:
    def job_submitted(self, job):
        pass

    def job_state(self, job):
        pass

class JobError(Exception): 
    def __init__(self, code):
        super.__init__("Job exited with code %d", self.code)

class ForwardWith:
    """Useful to forward the with statement in a function"""
    def __init__(self, context):
        self.context = context
        self.count = 0

    def __enter__(self):
        if self.count == 0:
            return self.context.__enter__()
        self.count += 1
        return self
    
    def previous(self):
        self.count -= 1
        return self

    def __exit__(self, *args):
        self.count -= 1
        if self.count == 0:
            logger.debug("Release lock in forward")
            return self.context.__exit__(*args)

class JobThread(threading.Thread):
    def __init__(self, job: Job):
        super().__init__()
        self.job = job

    def run(self):
        locks = []
        try:
            with ForwardWith(self.job.scheduler.cv) as joblock:
                for dependency in self.job.dependencies:
                    try:
                        locks.append(dependency.lock())
                    except LockError:
                        logger.warning("Could not lock %s, aborting start for job %s", dependency, self.job)
                        dependency.check()
                        self.job.state = JobState.READY
                        return
                
                    self.job.scheduler.jobstarted(self.job)
                    self.job.run(joblock, locks)
            self.job.state = JobState.DONE

        except JobError:
            logger.warning("Error while running job")
            self.job.state = JobState.ERROR

        except:
            logger.warning("Error while running job (in experimaestro)", exc_info=True)
            self.job.state = JobState.ERROR

        finally:
            if self.job.state in [JobState.DONE, JobState.ERROR]:
                self.job.scheduler.jobfinished(self.job)

            with self.job.scheduler.cv:
                for lock in locks:
                    lock.release()

class Scheduler():
    """Represents an experiment"""
    CURRENT = None

    def __init__(self, name):
        # Name of the experiment
        self.name = name

        # Whether jobs are submitted
        self.submitjobs = True

        # Condition variable for scheduler access
        self.cv = threading.Condition()

        # Exit mode activated
        self.exitmode = False

        # List of jobs
        self.waitingjobs = set()

        # Listeners
        self.listeners:Set[Listener] = set()

    def __enter__(self):
        self.old_experiment = Scheduler.CURRENT
        Scheduler.CURRENT = self


    def __exit__(self, *args):
        # Wait until all tasks are completed
        with self.cv:
            self.cv.wait_for(lambda : not self.waitingjobs)

        # Set back the old scheduler, if any
        logger.info("Exiting experiment %s", self.name)
        Scheduler.CURRENT = self.old_experiment

    def submit(self, job: Job):
        if self.exitmode:
            logger.warning("Exit mode: not submitting")
            return

        with self.cv:
            job.starttime = time.time()
            job.scheduler = self

            # Add dependencies, and add to blocking resources
            for dependency in job.dependencies:
                dependency.origin.dependents.add(dependency)
                dependency.check()

            self.waitingjobs.add(job)

            for listener in self.listeners:
                listener.job_submitted(job)

            job.state = JobState.WAITING if job.dependencies else JobState.READY

            if job.ready:
                self.start(job)

    def start(self, job: Job):
        with self.cv:
            if self.exitmode:
                logger.warning("Exit mode: not starting")
                return
            JobThread(job).start()
            
    def jobstarted(self, job: Job):
        """Called just before a job is run"""
        for listener in self.listeners:
            listener.job_state(job)

    def jobfinished(self, job: Job):
        """Called when the job is finished (state = error or done)"""
        with self.cv:
            self.waitingjobs.remove(job)
            self.cv.notify_all()
            for dependency in job.dependents:
                dependency.check()

            

        for listener in self.listeners:
            listener.job_state(job)

class experiment:
    """Experiment environment"""
    def __init__(self, path: Path, name: str, *, port:int=None):
        self.workspace = Workspace(path)
        self.experiment = Scheduler(name)
        self.port = port

    def __enter__(self):
        self.workspace.__enter__()
        self.experiment.__enter__()
        return self.workspace

    def __exit__(self, *args):
        self.experiment.__exit__()
        self.workspace.__exit__()
