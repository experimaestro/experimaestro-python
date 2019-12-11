from pathlib import Path
import threading
import re
import time
from typing import Optional, Set
import enum

from .workspace import Workspace
from .api import PyObject
from .utils import logger
from .dependencies import LockError, Dependency, DependencyStatus
from .locking import Lock

class JobState(enum.Enum):
    WAITING = 0
    READY = 1
    RUNNING = 2
    DONE = 3
    ERROR = 4

    def notstarted(self):
        return self.value <= JobState.READY.value

class JobLock(Lock):
    def __init__(self, job):
        self.job = job
    
    def acquire(self): 
        return self.job.status == JobState.DONE

    def release(self): 
        return False

class JobDependency(Dependency):
    def __init__(self, job):
        super().__init__(job)

    def status(self) -> DependencyStatus:
        if self.origin.state == JobState.DONE:
            return DependencyStatus.OK
        elif self.origin.state == JobState.ERROR:
            return DependencyStatus.FAIL
        return DependencyStatus.WAIT

    def lock(self):
        return JobLock(self.origin)

class Job():
    """Context of a job"""
    def __init__(self, parameters: PyObject, *, workspace:Workspace = None, launcher:"experimaestro.launchers" = None):
        self.workspace = workspace or Workspace.CURRENT
        assert self.workspace is not None, "No experiment has been defined"

        self.launcher = launcher or self.workspace.launcher
        assert self.launcher is not None, "No default launcher in workspace %s" % workspace

        self.type = parameters.__class__.__xpm__
        self.name = str(self.type.typename).rsplit(".", 1)[-1]

        self.scheduler:Optional["Scheduler"] = None
        self.parameters = parameters   
        self.state:JobState = JobState.WAITING

        # Dependencies
        self.dependencies:Set[Dependency] = set() # as target
        self.dependents:Set[Dependency] = set() # as source

        # Process
        self._process = None
        self.unsatisfied = 0

        # Meta-information
        self.starttime:Optional[float] = None
        self.submittime:Optional[float] = None
        self.endtime:Optional[float] = None
        self.progress = 0
        self.tags = parameters.tags()
        
    def __str__(self):
        return "Job[{}]".format(self.identifier)

    @property
    def ready(self):
        return self.state == JobState.READY

    @property
    def jobpath(self):
        return self.workspace.jobspath  / str(self.type.typename) / self.identifier

    @property
    def identifier(self):
        return self.parameters.__xpm__.identifier.hex()

    def run(self, locks):
        """Actually run the code"""
        raise NotImplementedError()

    def wait(self):
        """Waiting for job to finish"""
        with self.scheduler.cv:
            logger.debug("Waiting for job %s to finish", self.jobpath)
            self.scheduler.cv.wait_for(lambda: self.state in [JobState.ERROR, JobState.DONE])

        logger.debug("Job %s finished with state %s", self, self.state)
        return self.state

    @property
    def process(self):
        """Returns the process"""
        if self._process: 
            return self._process
        raise NotImplementedError("Should construct a new checker object")

    @property
    def lockpath(self):
        """This file is used as a lock and also stores the PID of the job when running"""
        return self.jobpath / ("%s.lock" % self.name)

    @property
    def donepath(self):
        """When a job has been successful, this file is written"""
        return self.jobpath / ("%s.done" % self.name)

    @property
    def failedpath(self):
        """When a job has been unsuccessful, this file is written with an error code inside"""
        return self.jobpath / ("%s.failed" % self.name)

    @property
    def stdout(self) -> Path:
        return self.jobpath / ("%s.out" % self.name)

    @property
    def stderr(self) -> Path:
        return self.jobpath / ("%s.err" % self.name)

    def dependencychanged(self, dependency, oldstatus, status):
        value = lambda s: 1 if s == DependencyStatus.OK else 0
        self.unsatisfied -= value(status) - value(oldstatus)


        logger.info("Job %s: unsatisfied %d", self, self.unsatisfied)

        if status == DependencyStatus.FAIL:    
            # Job completed
            self.scheduler.jobfinished(self, JobState.ERROR)
            return

        if self.unsatisfied == 0:    
            logger.info("Job %s is ready to run", self)
            self.scheduler.start(self)



class Listener:
    def job_submitted(self, job):
        pass

    def job_state(self, job):
        pass

class JobError(Exception): 
    def __init__(self, code):
        super.__init__("Job exited with code %d", self.code)

class JobThread(threading.Thread):
    """Job starting and monitoring thread"""
    def __init__(self, job: Job):
        super().__init__()
        self.job = job

    def run(self):
        locks = []
        try:
            with self.job.scheduler.cv:
                logger.info("Starting job %s with %d dependencies", self.job, len(self.job.dependencies))

                for dependency in self.job.dependencies:
                    try:
                        locks.append(dependency.lock())
                    except LockError:
                        logger.warning("Could not lock %s, aborting start for job %s", dependency, self.job)
                        dependency.check()
                        self.job.state = JobState.READY
                        return
                
                logger.info("Running job %s", self.job)
                self.job.scheduler.jobstarted(self.job)
                self.job.starttime = time.time()
                process = self.job.run(locks)
            
            if isinstance(process, JobState):
                state = process
            else:
                code = process.wait()
                state = JobState.DONE if code == 0 else JobState.ERROR

            
        except JobError:
            logger.warning("Error while running job")
            state = JobState.ERROR

        except:
            logger.warning("Error while running job (in experimaestro)", exc_info=True)
            state = JobState.ERROR

        finally:
            if state in [JobState.DONE, JobState.ERROR]:
                self.job.scheduler.jobfinished(self.job, state)

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

        # List of all jobs
        self.jobs = set()

        # List of jobs
        self.waitingjobs = set()

        # Listeners
        self.listeners:Set[Listener] = set()

    def __enter__(self):
        self.old_experiment = Scheduler.CURRENT
        Scheduler.CURRENT = self


    def __exit__(self, *args):
        # Wait until all tasks are completed
        logger.info("Waiting that experiment %s finishes", self.name)
        with self.cv:
            logger.debug("Waiting for %d jobs to complete", len(self.waitingjobs))
            self.cv.wait_for(lambda : not self.waitingjobs)

        # Set back the old scheduler, if any
        logger.info("Exiting experiment %s", self.name)
        Scheduler.CURRENT = self.old_experiment

    def submit(self, job: Job):
        with self.cv:
            if self.exitmode:
                logger.warning("Exit mode: not submitting")
                return

            job.submittime = time.time()
            job.scheduler = self

            # Add dependencies, and add to blocking resources
            job.unsatisfied = len(job.dependencies)
            for dependency in job.dependencies:
                dependency.target = job
                dependency.origin.dependents.add(dependency)
                dependency.check()

            self.jobs.add(job)
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

    def jobfinished(self, job: Job, state: JobState):
        """Called when the job is finished (state = error or done)"""
        with self.cv:
            job.endtime = time.time()
            job.state = state
            logger.debug("Job %s has finished (%s)", job, job.state)
            self.waitingjobs.remove(job)
            for dependency in job.dependents:
                dependency.check()
            self.cv.notify_all()

        for listener in self.listeners:
            listener.job_state(job)

    def addlistener(self, listener: Listener):
        self.listeners.add(listener)
    def removelistener(self, listener: Listener):
        self.listeners.remove(listener)

class experiment:
    """Experiment environment"""
    def __init__(self, path: Path, name: str, *, port:int=None):
        from experimaestro.server import Server
    
        self.workspace = Workspace(path)
        self.scheduler = Scheduler(name)
        self.server = Server(self.scheduler, port) if port else None

    def __enter__(self):
        if self.server:
            self.server.start()

        self.workspace.__enter__()
        self.scheduler.__enter__()
        return self.workspace

    def __exit__(self, *args):
        self.scheduler.__exit__()
        self.workspace.__exit__()
        if self.server:
            self.server.stop()


def progress(value: float):
    logger.debug("Got progress value of %s", value)
