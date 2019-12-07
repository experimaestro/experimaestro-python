"""Command line jobs"""

import os
from pathlib import Path
from .scheduler import Job, JobError
class AbstractCommandComponent(): pass

# 5 seconds wait for locking file
LOCKFILE_WAIT_DURATION = 5

class CommandContext: pass

class CommandPath(AbstractCommandComponent):
    def __init__(self, path: [Path, str]):
        self.path = Path(path)

class CommandString(AbstractCommandComponent):
    def __init__(self, string: str):
        self.string = string

class CommandParameters(AbstractCommandComponent):
    def output(self, context: CommandContext): 
        pass

class AbstractCommand(): pass

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

    def run(self, locks):
        LOGGER.info("Running job %s...", self)

        scriptBuilder = self.launcher.scriptbuilder()
        processBuilder = self.launcher.processbuilder()
        connector = self.launcher.connector()
        donepath = self.job.donepath

        def waitUntilFinished():
            LOGGER.info("Waiting for job {} to finish", _locator)
            exitCode = -1
            # try      exitCode = self.process.exitCode()
            # } catch(exited_error &)      # Could not read the exit value, fallback
            #     LOGGER.info("Process exited before wait process was in place, from file")
            #     codePath = pathTo(EXIT_CODE_PATH)
            #     istream = connector.istream(codePath)
            #     *istream >> exitCode
            # } catch(...)      LOGGER.warn("Unhandled exception while waiting for job to finish: setting state to fail")
            # state(exitCode == 0 ? JobState.DONE : JobState.ERROR)
            # LOGGER.info("Job {} finished with exit code {} (state {})", _locator, exitCode, state())


        # Check if already done
        def check():
            if donepath.is_file():      
                LOGGER.info("Job {} is already done", *self)
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
        LOGGER.debug("Making directories job %s...", self.locator)
        directory = self.locator.parent
        directory.mkdir(directory, parents_ok=True, exist_ok=False)

        # Lock
        lockPath = self.job.lockpath
        lock = self.launcher.connector.lock(lockPath, LOCKFILE_WAIT_DURATION)
        if not lock:    
            # FIXME: put on hold for a while
            LOGGER.warn("Could not lock %s", self.locator)
            return


        # Check again if done (now that we have locked everything)
        if check():
            return

        # Now we can write the script
        scriptBuilder.command = _command
        scriptBuilder.lockFiles.push_back(lockPath)
        scriptPath = scriptBuilder.write(*_workspace, connector, _locator, *self)
        startlock = self.launcher.connector().lock(pathTo(LOCK_START_PATH), LOCKFILE_WAIT_DURATION)
        if not startlock:    LOGGER.warn("Could not lock start file {}", _locator)
        return


        LOGGER.info("Starting job {}", _locator)
        processBuilder.environment = self.launcher.environment()
        processBuilder.command.push_back(self.launcher.connector().resolve(scriptPath))
        processBuilder.stderr = Redirect.file(connector.resolve(directory.resolve({_locator.name() + ".err"})))
        processBuilder.stdout = Redirect.file(connector.resolve(directory.resolve({_locator.name() + ".out"})))
        # os.spawnl(os.P_DETACH, 'some_long_running_command')

        self.process = processBuilder.start()
        state(JobState.RUNNING)

        # Avoid to remove the locks ourselves
        startlock.detachState(True)
        lock.detachState(True)

        # Unlock since started
        jobLock.unlock()

        # Wait for end of execution
        waitUntilFinished()
                

class CommandLineTask():
    def __init__(self, commandline: CommandLine):
        self.commandline = commandline
        

    def __call__(self, pyobject, *, launcher=None, workspace=None) -> Job:
        return CommandLineJob(pyobject, launcher=launcher, workspace=workspace)

