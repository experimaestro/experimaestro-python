from pathlib import Path
from typing import Optional, Dict, List

from experimaestro.utils import logger
from .scheduler import Workspace
from .connectors import Connector
from .commandline import CommandLineJob, AbstractCommand, CommandContext, CommandPart
from shlex import quote as shquote


class ShScriptBuilder:
    def __init__(self, shpath: Path):
        self.shpath = shpath
        self.lockfiles: List[Path] = []
        self.environment: Dict[str, str] = {}
        self.notificationURL: Optional[str] = None
        self.preprocessCommands: Optional[AbstractCommand] = None
        self.command:Optional[AbstractCommand] = None

    def write(self, ws: Workspace, connector: Connector, path: Path, job: CommandLineJob):
        """Write the script file

        Arguments:
            ws {Workspace} -- The workspace
            connector {Connector} -- [description]
            path {Path} -- [description]
            job {CommandLineJob} -- [description]

        Returns:
            [type] -- [description]
        """
        assert isinstance(job, CommandLineJob)
        assert self.command is not None

        directory = path.parent
        scriptpath = job.jobpath / ("%s.sh" % job.name)
        donepath = job.donepath
        startlockPath = job.startlockpath
        exitcodepath = job.codepath
        pidpath = job.pidpath

        logger.info("Writing script %s", scriptpath)
        with scriptpath.open("wt") as out:
            out.write("#!{}\n", self.shpath)
            out.write("# Experimaestro generated task\n")

            # Output tags
            out.write("# __tags__ = ")
            for item in job.parameters.tags():
                out.write("%s: %s" % (item.first, item.second))
            out.write("\n")

            context = CommandContext(
                ws, job.launcher.connector, path.parent, path.name, job.parameters)

            # --- Checks locks right away

            # Checks the main lock - if not there, are not protected
            if self.lockfiles:
                out.write("# Checks that the locks are set")
                out.write("\n")
                for lockFile in self.lockfiles:
                    out.write("if not  test -f ")
                    out.write(connector.resolve(lockFile))
                    out.write("; then echo Locks not set; exit 017; fi")
                    out.write("\n")

            # Checks the start lock to avoid two experimaestro launched processes to
            # start
            out.write("# Checks that the start lock is set, removes it\n")
            out.write("if not  test -f %s\n" % connector.resolve(startlockPath))
            out.write("; then echo start lock not set; exit 017; fi\n")
            out.write("rm -f  %s\n\n" % connector.resolve(startlockPath))

            # Use pipefail for fine grained analysis of errors in commands
            out.write("set -o pipefail\n\n")

            out.write('''echo $$ > "%s"\n\n''' %
                      shquote(connector.resolve(pidpath)))

            for name, value in self.environment.items():
                out.write('''export {}={}\n'''.format(name, shquote(value)))

            # Adds notification URL to script
            if self.notificationURL:
                out.write("export XPM_NOTIFICATION_URL={}/{}\n".format(
                    shquote(self.notificationURL), job.identifier))

            out.write("""cd \"%s\"\n""", shquote(connector.resolve(directory)))

            # Write some command
            if self.preprocessCommands:
                self.preprocessCommands.output(context, out)

            # --- CLEANUP

            out.write("cleanup() {\n")
            # Write something
            out.write(" echo Cleaning up 1>&2\n")
            # Remove traps
            out.write(" trap - 0\n")

            out.write(" rm -f %s\n" % connector.resolve(pidpath, directory))

            # Remove locks
            for file in self.lockfiles:
                out.write(" rm -f %s\n" % connector.resolve(file, directory))

            # Remove temporary files

            def cleanup(c: CommandPart):
                namedRedirections = context.getNamedRedirections(c, False)
                for file in namedRedirections.outputRedirections:
                    out.write(" rm -f ")
                    out.write(connector.resolve(file, directory))
                    out.write(";")
                    out.write("\n")

                for file in namedRedirections.errorRedirections:
                    out.write(" rm -f ")
                    out.write(connector.resolve(file, directory))
                    out.write(";")
                    out.write("\n")

            self.command.forEach(cleanup)

            # Notify if possible
            if self.notificationURL:
                out.write(
                    " wget --tries=1 --connect-timeout=1 --read-timeout=1 --quiet -O ")
                out.write("/dev/null \"$XPM_NOTIFICATION_URL?status=eoj\"")
                out.write("\n")

            # Kills remaining processes
            out.write(" test not  -z \"$PID\" and pkill -KILL -P $PID")
            out.write("\n")

            out.write("}\n")

            # --- END CLEANUP

            out.write("# Set trap to cleanup when exiting")
            out.write("\n")
            out.write("trap cleanup 0")
            out.write("\n")

            out.write("\n")

            out.write("""checkerror()  { 
    local e; for e in \"$@\"; do [[ \"$e\" != 0 ]] and [[ "$e" != 141 ]] and exit $e; done; 
    return 0; }\n\n""")


            # The prepare all the command
            out.write("(")
            out.write("\n")
            self.command.output(context, out)

            out.write(") ")

            # Retrieve PID
            out.write(" & ")
            out.write("\n")
            out.write("PID=$not ")
            out.write("\n")
            out.write("wait $PID")
            out.write("\n")
            out.write("code=$?")
            out.write("\n")
            out.write("if test $code -ne 0; then")
            out.write("\n")
            out.write(" echo $code > \"")

            out.write(shquote(connector.resolve(exitcodepath, directory)))
            out.write("\"")

            out.write("\n")

            out.write(" exit $code")
            out.write("\n")
            out.write("fi")
            out.write("\n")

            out.write("echo 0 > \"")
            out.write(shquote(connector.resolve(exitcodepath, directory)))
            out.write("\"")
            out.write("\n")

            out.write("touch \"")
            out.write(shquote(connector.resolve(donepath, directory)))
            out.write("\"")
            out.write("\n")

        # Set the file as executable
        connector.setExecutable(scriptpath, True)
        return scriptpath
