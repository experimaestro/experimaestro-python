from pathlib import Path
from typing import Optional, Dict, List

from experimaestro.utils import logger
from .scheduler import Workspace, NOTIFICATIONURL_VARNAME
from .connectors import Connector, RedirectType, Redirect
from .commandline import CommandLineJob, AbstractCommand, CommandContext, CommandPart
from shlex import quote as shquote


class ShCommandContext(CommandContext):
    def writeRedirection(self, out, redirect, stream):
        if redirect.type == RedirectType.INHERIT:
            pass
        elif redirect.type == RedirectType.FILE:
            relpath = self.resolve(redirect.path)
            out.write(" > {}".format(relpath))
        else:
            raise ValueError("Unsupported output redirection type %s" % redirect.type)

    def printRedirections(
        self, stream: int, out, outputRedirect: Redirect, outputRedirects
    ):
        if outputRedirects:
            # Special case : just one redirection
            if (
                len(outputRedirects) == 1
                and outputRedirect.type == RedirectType.INHERIT
            ):
                self.writeRedirection(
                    out, Redirect.file(outputRedirects[0].toString()), stream
                )
            else:
                out.write(' : {} > >(tee ")'.format(stream))
                for file in outputRedirects:
                    out.write(self.relpath(file))
                self.writeRedirection(out, outputRedirect, stream)
                out << ")"

        else:
            # Finally, the main redirection
            self.writeRedirection(out, outputRedirect, stream)


SH_FLOCK = """exec {0}<> {1}
if ! flock -n {0}; then 
    echo Could not lock {1} - stopping 1>&2
    exit 017
fi
"""


class ShScriptBuilder:
    def __init__(self, shpath: Path = "/bin/bash"):
        self.shpath = shpath
        self.lockfiles: List[Path] = []
        self.notificationURL: Optional[str] = None
        self.preprocessCommands: Optional[AbstractCommand] = None
        self.command: Optional[AbstractCommand] = None
        self.processtype = "local"

    def write(self, job: CommandLineJob):
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
        assert job.workspace

        directory = job.jobpath
        connector = job.launcher.connector
        directorypath = connector.resolve(directory)
        ws = job.workspace
        context = ShCommandContext(
            ws, job.launcher.connector, directory, job.name, job.config
        )

        relpath = lambda path: shquote(context.relpath(path))

        scriptpath = job.jobpath / ("%s.sh" % job.name)
        donepath = relpath(job.donepath)
        lockpath = relpath(job.lockpath)
        pidpath = relpath(job.pidpath)

        logger.info("Writing script %s", scriptpath)
        with scriptpath.open("wt") as out:
            out.write("#!{}\n".format(self.shpath))
            out.write("# Experimaestro generated task\n")

            # --- Checks locks right away

            # change directory
            out.write(f"""cd {shquote(directorypath)}\n""")

            # Lock all the needed files
            FIRST_FD = 9
            for i, path in enumerate(self.lockfiles):
                out.write(SH_FLOCK.format(i + FIRST_FD, relpath(path)))

            # Use pipefail for fine grained analysis of errors in commands
            out.write("set -o pipefail\n\n")

            out.write(
                """echo "{ \\"type\\": \\"%s\\", \\"pid\\": $$ }" > %s\n\n"""
                % (self.processtype, pidpath)
            )

            for name, value in job.launcher.environ.items():
                out.write("""export {}={}\n""".format(name, shquote(value)))

            # Adds notification URL to script
            if self.notificationURL:
                out.write(
                    "export {}={}/{}\n".format(
                        NOTIFICATIONURL_VARNAME,
                        shquote(self.notificationURL),
                        job.identifier,
                    )
                )

            # Write some command
            if self.preprocessCommands:
                self.preprocessCommands.output(context, out)

            # --- CLEANUP

            out.write("cleanup() {\n")

            # Write something
            out.write(" echo Cleaning up 1>&2\n")

            # Remove traps
            out.write(" trap - 0\n")

            # Remove PID file
            out.write("""rm %s\n""" % pidpath)

            # Remove temporary files

            def cleanup(c: CommandPart):
                namedRedirections = context.getNamedRedirections(c, False)
                for file in namedRedirections.redirections():
                    out.write(" rm -f {}\n".format(relpath(file)))

            self.command.forEach(cleanup)

            # Notify if possible
            if self.notificationURL:
                out.write(
                    " wget --tries=1 --connect-timeout=1 --read-timeout=1 --quiet -O "
                )
                out.write('/dev/null "$XPM_NOTIFICATION_URL?status=eoj"\n')

            # Kills remaining processes
            out.write(' test ! -z "$PID" && pkill -KILL -P $PID')
            out.write("\n")

            out.write("}\n")

            # --- END CLEANUP

            out.write("# Set trap to cleanup when exiting")
            out.write("\n")
            out.write("trap cleanup 0")
            out.write("\n")

            out.write("\n")

            out.write(
                """checkerror()  { 
    local e; for e in \"$@\"; do [[ \"$e\" != 0 ]] && [[ "$e" != 141 ]] && exit $e; done; 
    return 0; }\n\n"""
            )

            # Output the full command
            out.write("(\n")
            self.command.output(context, out)
            out.write(") & \n")

            # Retrieve PID
            out.write("PID=$!\n")
            out.write("wait $PID\n")
            out.write("code=$?\n")
            out.write("if test $code -ne 0; then\n")
            out.write(" echo $code > {}\n".format(relpath(job.failedpath)))
            out.write(" exit $code\n")
            out.write("fi\n")
            out.write("\n")
            out.write("touch {}\n".format(relpath(job.donepath)))

        # Set the file as executable
        connector.setExecutable(scriptpath, True)
        return scriptpath
