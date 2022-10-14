from pathlib import Path
import sys
from typing import Optional, List
import os
from experimaestro.utils import logger
from .connectors import RedirectType, Redirect
from .commandline import CommandLineJob, AbstractCommand, CommandContext, CommandPart
from shlex import quote as shquote


# TODO: should be reworked with the new way to build commands
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


class PythonScriptBuilder:
    """Builds a Python script"""

    def __init__(self, pythonpath: Path = None):
        self.pythonpath = pythonpath or Path(sys.executable)
        self.lockfiles: List[Path] = []
        self.notificationURL: Optional[str] = None
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
        assert isinstance(
            job, CommandLineJob
        ), "Cannot handle a job which is not a command line job"
        assert self.command is not None
        assert job.workspace, "No workspace defined for the job"
        assert job.launcher is not None, "No launcher defined for the job"

        directory = job.jobpath
        connector = job.launcher.connector
        ws = job.workspace
        context = ShCommandContext(
            ws, job.launcher.connector, directory, job.name, job.config
        )

        def relpath(path: Path):
            return shquote(context.relpath(path))

        # FIXME: big hack to generate the params.json file
        # which is all what we need for now, but this might not be true in the future
        with open(os.devnull, "w") as fpnull:
            self.command.output(context, fpnull)
        scriptpath = job.jobpath / ("%s.py" % job.name)

        logger.debug("Writing script %s", scriptpath)
        with scriptpath.open("wt") as out:
            out.write("#!{}\n".format(self.pythonpath))
            out.write("# Experimaestro generated task\n")

            # --- Checks locks right away

            out.write("""from experimaestro.run import TaskRunner\nimport os\n\n""")

            out.write("lockfiles = [\n")
            for path in self.lockfiles:
                out.write(f"   '''{relpath(path)}''',\n")
            out.write("]\n")

            for name, value in job.environ.items():
                out.write(f"""os.environ["{name}"] = "{shquote(value)}"\n""")
            out.write("\n")

            out.write(
                f"""TaskRunner("{shquote(connector.resolve(scriptpath))}", lockfiles).run()\n"""
            )

        # Set the file as executable
        connector.setExecutable(scriptpath, True)
        return scriptpath
