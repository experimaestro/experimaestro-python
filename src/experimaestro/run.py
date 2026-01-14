"""Command line parsing"""

import os
from pathlib import Path
import signal
import sys
import json
from typing import List
import filelock
from experimaestro.notifications import progress, report_eoj, start_of_job
from experimaestro.utils.multiprocessing import delayed_shutdown
from experimaestro.exceptions import GracefulTimeout
from experimaestro.locking import JobDependencyLocks
from .core.types import ObjectType
from experimaestro.utils import logger
from experimaestro.core.objects import ConfigInformation
import experimaestro.taskglobals as taskglobals
import atexit


def parse_commandline(argv=None):
    """Called when executing a task"""
    if argv is None:
        argv = sys.argv[1:]
    taskid, params = argv
    tasktype = ObjectType.REGISTERED[taskid]
    with open(params, "r") as fp:
        params = json.load(fp)
        ConfigInformation.LOADING = True
        task = tasktype(**params)
        ConfigInformation.LOADING = False
        task.execute()


def run(parameters: Path):
    with open(parameters, "r") as fp:
        params = json.load(fp)
        env = taskglobals.Env.instance()

        env.wspath = Path(params["workspace"])
        env.taskpath = parameters.parent

        task = ConfigInformation.fromParameters(params["objects"])
        task.__taskdir__ = Path.cwd()

        # Notify that the task has started
        progress(0)

        # Execute the task
        task.execute()


def rmfile(path: Path):
    if path.is_file():
        logger.debug("Removing file %s", path)
        path.unlink()


class TaskRunner:
    """Runs a task, after locking"""

    def __init__(self, scriptpath: str, lockfiles: List[str]):
        # Sets the working directory
        self.scriptpath = Path(scriptpath)
        self.pidfile = self.scriptpath.with_suffix(".pid")
        self.lockfiles = lockfiles
        self.donepath = self.scriptpath.with_suffix(".done")
        self.failedpath = self.scriptpath.with_suffix(".failed")
        self.started = False
        self.locks = []
        self.dynamic_locks = JobDependencyLocks()
        env = taskglobals.Env.instance()
        env.taskpath = self.scriptpath.parent

        self.cleaned = False

    def cleanup(self):
        if not self.cleaned:
            self.cleaned = True
            logger.info("Cleaning up")
            rmfile(self.pidfile)

            # Release IPC locks
            for lock in self.locks:
                try:
                    if lock.acquired:
                        logger.info("Releasing lock")
                        lock.release()
                        logger.info("Released lock")
                except Exception:
                    logger.error("Error while releasing lock %s", lock)

            # Note: dynamic dependency locks are released via context manager
            # in the run() method, not here

            if self.started:
                report_eoj()
            logger.info("Finished cleanup")

    def handle_error(self, code, frame_type, reason: str = "failed", message: str = ""):
        """Handle task error and write failure information.

        Args:
            code: Exit code
            frame_type: Signal frame type (unused)
            reason: Failure reason (e.g., "failed", "timeout")
            message: Optional message with details
        """
        logger.info("Error handler: finished with code %d, reason=%s", code, reason)
        failure_info = {"code": code, "reason": reason}
        if message:
            failure_info["message"] = message
        self.failedpath.write_text(json.dumps(failure_info))
        self.cleanup()
        logger.info("Exiting")
        delayed_shutdown(60, exit_code=code)
        sys.exit(1)

    def run(self):
        atexit.register(self.cleanup)
        sigterm_handler = signal.signal(signal.SIGTERM, self.handle_error)
        sigint_handler = signal.signal(signal.SIGINT, self.handle_error)

        def remove_signal_handlers(remove_cleanup=True):
            """Removes cleanup in forked processes"""
            signal.signal(signal.SIGTERM, sigterm_handler)
            signal.signal(signal.SIGINT, sigint_handler)
            atexit.unregister(self.cleanup)

        if sys.platform != "win32":
            os.register_at_fork(after_in_child=remove_signal_handlers)

        try:
            workdir = self.scriptpath.parent
            os.chdir(workdir)
            os.getpid()
            logger.info("Working in directory %s", workdir)

            for lockfile in self.lockfiles:
                fullpath = str(Path(lockfile).resolve())
                logger.info("Locking %s", fullpath)
                lock = filelock.FileLock(fullpath)
                # MAYBE: should have a clever way to lock
                # Problem = slurm would have a job doing nothing...
                # Fix = maybe with two files
                lock.acquire()
                self.locks.append(lock)

            # Load and setup dynamic dependency locks from locks.json
            locks_path = workdir / "locks.json"
            if locks_path.exists():
                logger.info("Loading dynamic dependency locks from %s", locks_path)
                with locks_path.open() as f:
                    locks_data = json.load(f)
                self.dynamic_locks = JobDependencyLocks.from_json(
                    locks_data.get("dynamic_locks", [])
                )
                logger.info(
                    "Loaded %d dynamic dependency locks", len(self.dynamic_locks.locks)
                )

            # Check if failed/done have been generated by another job
            if self.donepath.is_file():
                logger.info("Job already completed")
            else:
                logger.info("Running task")
                rmfile(self.failedpath)
                self.started = True

                # Notify that the job has started
                start_of_job()

                # Acquire dynamic dependency locks while running the task
                with self.dynamic_locks.dependency_locks():
                    run(workdir / "params.json")

                # ... remove the handlers
                remove_signal_handlers(remove_cleanup=False)

                # Everything went OK
                logger.info("Task ended successfully")
                self.cleanup()
                sys.exit(0)
        except GracefulTimeout as e:
            logger.info("Task requested graceful timeout: %s", e.message)
            self.handle_error(1, None, reason="timeout", message=e.message)

        except Exception:
            logger.exception("Got exception while running")
            self.handle_error(1, None)

        except SystemExit as e:
            if e.code == 0:
                # Normal exit, just create the ".done" file
                self.donepath.touch()

                # ... and finish the exit process
                raise
            else:
                self.handle_error(e.code, None)
