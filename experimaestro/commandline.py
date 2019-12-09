"""Command line jobs"""

import os
import io
from pathlib import Path
from typing import Union, Callable, Dict

from experimaestro.utils import logger
from .scheduler import Job, JobError, JobState
from .connectors import Redirect, Connector
from .scheduler import Workspace
from .api import PyObject


# 5 seconds wait for locking file
LOCKFILE_WAIT_DURATION = 5

class NamedPipeRedirections:
    """List of named pipes"""
    def __init__(self):
        self.outputRedirections: List[Path] = []
        self.errorRedirections: List[Path] = []

EMPTY_REDIRECTIONS = NamedPipeRedirections()

class CommandPart:
  def forEach(self, f: Callable[["CommandPart"], None]):
      raise NotImplementedError()

  def output(self, context: "CommandContext", out: io.TextIOBase):
      raise NotImplementedError()

class AbstractCommandComponent(CommandPart): pass

class CommandContext: 
    def __init__(self, workspace: Workspace, connector: Connector, path: Path, name: str, parameters: PyObject):
        self.workspace = workspace
        self.connector = connector
        self.path = path
        self.name = name
        self.parameters = parameters
        self.namedPipeRedirectionsMap:Dict["CommandPart", NamedPipeRedirections] = {}

    def getNamedRedirections(self, key: "CommandPart", create: bool) -> NamedPipeRedirections:
        x = self.namedPipeRedirectionsMap.get(key, None)        
        if x:
            return x

        if not create:
            return EMPTY_REDIRECTIONS

        x = NamedPipeRedirections()
        self.namedPipeRedirectionsMap[key] = x
        return x

class CommandPath(AbstractCommandComponent):
    def __init__(self, path: Union[Path, str]):
        self.path = Path(path)

class CommandString(AbstractCommandComponent):
    def __init__(self, string: str):
        self.string = string

class CommandParameters(AbstractCommandComponent):
    def output(self, context: CommandContext, out: io.TextIOBase): 
        raise NotImplementedError()

class AbstractCommand(CommandPart):
    def output(self, context: CommandContext, out: io.TextIOBase):
        raise NotImplementedError()

class Command(AbstractCommand):
    def __init__(self):
        self.components = []

    def add(self, component: AbstractCommandComponent):
        self.components.append(component)

class CommandLine():
    """A command line is composed of one or more commands"""
    def __init__(self):
        self.commands = []

    def add(self, command: Command):
        self.commands.append(command)




class CommandLineJob(Job): 

    def run(self, jobLock, locks):
        with jobLock.previous():
            # Use the lock during preparation
            logger.info("Running job %s...", self)

            scriptBuilder = self.launcher.scriptbuilder()
            processBuilder = self.launcher.processbuilder()
            connector = self.launcher.connector()
            donepath = self.job.donepath

            # Check if already done
            def check():
                if donepath.is_file():      
                    logger.info("Job %s is already done", self)
                    job.state = JobState.DONE
                    self.job.scheduler.jobfinished(self.job)
                    return True

                # check if done
                self.process = self.launcher.check(self)
                if self.process:      
                    waitUntilFinished()

                return False

            if check():
                return

            # Lock the job and check done again (just in case)
            logger.debug("Making directories job %s...", self.locator)
            directory = self.locator.parent
            directory.mkdir(directory, parents_ok=True, exist_ok=False)

            # Lock
            lockPath = self.job.lockpath
            lock = self.launcher.connector.lock(lockPath, LOCKFILE_WAIT_DURATION)
            if not lock:    
                # FIXME: put on hold for a while
                logger.warn("Could not lock %s", self.locator)
                return


            # Check again if done (now that we have locked everything)
            if check():
                return

            # Now we can write the script
            scriptBuilder.command = self.command
            scriptBuilder.lockFiles.push_back(lockPath)
            scriptPath = scriptBuilder.write(self.workspace, connector, self.locator, self)
            startlock = self.launcher.connector().lock(self.job.startlockpath, LOCKFILE_WAIT_DURATION)
            if not startlock:    
                logger.warn("Could not lock start file %s", self.locator)
                return


            logger.info("Starting job %s", self.locator)
            processBuilder.environment = self.launcher.environment()
            processBuilder.command.push_back(self.launcher.connector().resolve(scriptPath))
            processBuilder.stderr = Redirect.file(connector.resolve(directory.resolve({self.locator.name() + ".err"})))
            processBuilder.stdout = Redirect.file(connector.resolve(directory.resolve({self.locator.name() + ".out"})))

            self.process = processBuilder.start()
            self.state = JobState.RUNNING

            # Avoid to remove the locks ourselves
            startlock.detachState(True)
            lock.detachState(True)

            # Unlock since started
            jobLock.unlock()

        # Wait for end of execution
        def waitUntilFinished():
            logger.info("Waiting for job %s to finish", self.locator)
            exitCode = -1
            # try      exitCode = self.process.exitCode()
            # } catch(exited_error &)      # Could not read the exit value, fallback
            #     logger.info("Process exited before wait process was in place, from file")
            #     codepath = pathTo(EXIT_CODE_PATH)
            #     istream = connector.istream(codepath)
            #     *istream >> exitCode
            # } catch(...)      logger.warn("Unhandled exception while waiting for job to finish: setting state to fail")
            # state(exitCode == 0 ? JobState.DONE : JobState.ERROR)
            # logger.info("Job %s finished with exit code %s (state %s)", self.locator, exitCode, state())

        waitUntilFinished()


class CommandLineTask():
    def __init__(self, commandline: CommandLine):
        self.commandline = commandline
        

    def __call__(self, pyobject, *, launcher=None, workspace=None) -> Job:
        return CommandLineJob(pyobject, launcher=launcher, workspace=workspace)

