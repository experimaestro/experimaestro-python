try:
    from pathlib import Path, _posix_flavour
except ImportError:
    # Avoids problem with python 3.12 where this module does not work
    # anyways
    _posix_flavour = None

from dataclasses import dataclass
import io
import os
import re
from experimaestro.launcherfinder import LauncherRegistry
from urllib.parse import urlparse
from itertools import chain
from . import Connector
from . import (
    Process,
    ProcessBuilder,
    RedirectType,
    Redirect,
)
from experimaestro.locking import Lock
from experimaestro.tokens import Token

try:
    from fabric import Connection
    from invoke import Promise
    from invoke.exceptions import Failure
except Exception:
    # Just define placeholders
    class Connection:
        pass

    class Promise:
        pass

    class Failure(Exception):
        pass


# Might be wise to switch to https://github.com/marian-code/ssh-utilities


class SshPath(Path):
    """SSH path

    Absolute:
    ssh://[user@]host[:port]//this/is/a/path

    Relative:
    ssh://[user@]host[:port]/relative/path
    """

    _flavour = _posix_flavour

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, url: str):
        parsed = urlparse(url)
        assert parsed.scheme == "ssh"
        self._host = parsed.hostname

        self._parts = re.split(r"/+", parsed.path)
        if parsed.path.startswith("//"):
            self._parts[0] = "/"
        else:
            self._parts = self._parts[1:]

    @property
    def hostpath(self):
        if self.is_absolute():
            return "/" + "/".join(self._parts[1:])
        return "/".join(self._parts)

    @property
    def host(self):
        return self._host

    def is_absolute(self):
        return self._parts and self._parts[0] == "/"

    @classmethod
    def _parse_args(cls, args):
        drv = ""
        if args[0].startswith("ssh:"):
            url = urlparse(args[0])
            assert not url.fragment and not url.query

            path = url.path

            if path.startswith("//"):
                # Absolute path
                args = tuple(chain(["/", path[2:]], args[1:]))
            else:
                args = tuple(chain([path[1:]], args[1:]))

            drv = url.hostname
        _, root, parts = super()._parse_args(args)
        return (drv, root, parts)

    def _make_child(self, args):
        drv, root, parts = self._parse_args(args)
        assert self._host == drv or drv == "", f"{self._host} and {drv}"
        drv, root, parts = self._flavour.join_parsed_parts(
            "", "", self._parts, "", root, parts
        )

        child = object.__new__(SshPath)
        child._parts = parts
        child._host = self._host
        return child

    def open(self, mode="r", buffering=-1, encoding=None, errors=None, newline=None):
        # FIXME: should probably be wiser
        fileobj = (
            SshConnector.get(self.host)
            .connection.sftp()
            .open(self.hostpath, mode, buffering)
        )
        if "b" in mode:
            return fileobj

        return io.TextIOWrapper(
            fileobj, encoding=encoding, newline=newline, errors=errors
        )

    def absolute(self):
        """Return an absolute version of this path.  This function works
        even if the path doesn't point to anything.

        No normalization is done, i.e. all '.' and '..' will be kept along.
        Use resolve() to get the canonical path to a file.
        """
        # XXX untested yet!
        if self._closed:
            self._raise_closed()
        if self.is_absolute():
            return self
        # FIXME this must defer to the specific flavour (and, under Windows,
        # use nt._getfullpathname())
        obj = self._from_parts([os.getcwd()] + self._parts, init=False)
        obj._init(template=self)
        return obj

    def __repr__(self):
        return "SshPath(%s,%s)" % (self._host, self._flavour.join(self._parts[1:]))

    def __str__(self):
        return "ssh://%s/%s" % (self._host, self._flavour.join(self._parts[1:]))


@dataclass
class SshConfiguration:
    hostname: str

    def create(self, registry: LauncherRegistry):
        return SshConnector.get(self.hostname)


def get_stream(redirect: Redirect, write: bool):
    if redirect.type == RedirectType.FILE:
        raise NotImplementedError()

    if redirect.type == RedirectType.PIPE:
        raise NotImplementedError()

    if redirect.type == RedirectType.INHERIT:
        return None

    raise NotImplementedError("For %s", redirect)


class SshProcess(Process):
    def __init__(self, promise: Promise):
        self.promise = promise

    def __repr__(self):
        return f"SshProcess({self._process.pid})"

    def tospec(self):
        return {"type": "ssh", "pid": self._process.pid}

    def kill(self):
        self._process.kill()

    @staticmethod
    def fromspec(connector, spec):
        return None

    def wait(self) -> int:
        try:
            self.promise.join()
        except Failure:
            raise


class SshProcessBuilder(ProcessBuilder):
    def __init__(self, connector: "SshConnector"):
        super().__init__()
        self.connector = connector

    def start(self, task_mode: bool = False):
        """Start the process"""

        trans = str.maketrans({'"': r"\"", "$": r"\$"})
        command = f'''"{'", "'.join([c.translate(trans) for c in self.command])}"'''

        # stdin = get_stream(self.stdin, False)
        # stdout = get_stream(self.stdout, True)
        # stderr = get_stream(self.stderr, True)

        self.connector.connection.run(
            command, asynchronous=not self.detach, disown=self.detach
        )
        raise NotImplementedError()
        # if self.detach:
        #     self.connector.connection
        #     p = subprocess.Popen(
        #         self.command,
        #         stdin=stdin,
        #         stderr=stderr,
        #         stdout=stdout,
        #         env=self.environ,
        #         close_fds=True,
        #         cwd="/",
        #     )
        # else:
        #     p = subprocess.Popen(
        #         self.command,
        #         stdin=stdin,
        #         stderr=stderr,
        #         stdout=stdout,
        #         env=self.environ,
        #     )

        # process = LocalProcess(p)

        # if self.stdout and self.stdout.type == RedirectType.PIPE:
        #     self.stdout.function(p.stdout)
        # if self.stderr and self.stderr.type == RedirectType.PIPE:
        #     self.stderr.function(p.stderr)

        # return process


class SshConnector(Connector):
    @staticmethod
    def init_registry(registry: LauncherRegistry):
        registry.register_connector("ssh", SshConfiguration)

    def __init__(self, hostname: str):
        """Creates a new SSH connector

        :param hostname: a hostname specification
        """
        self.connection = Connection(hostname)
        # self.hostname = hostname
        # self.port = None

        # # FIXME: should connect on demand
        # config = paramiko.SSHConfig()
        # with Path("~/.ssh/config").expanduser().open("r") as fp:
        #     config.parse(fp)

        # lookup = config.lookup(hostname)

        # self.client = paramiko.SSHClient()

        # if "proxycommand" in lookup:
        #     paramiko.proxy.ProxyCommand(lookup["proxycommand"])

        # self.client.load_system_host_keys()
        # self.client.connect(lookup["hostname"])
        # self.sftp = self.client.open_sftp()

    @staticmethod
    def fromPath(path: SshPath):
        """Creates an SSH connector from an SshPath"""
        return SshConnector.get(path.host)

    @staticmethod
    def get(hostname):
        """Get an SSH connector from a hostname

        This method can caches SSH connectors, and is thus preferred
        to direct initialization
        """
        # TODO: cache values?
        return SshConnector(hostname)

    def __str__(self):
        return "%s" % self.hostname

    def processbuilder(self) -> ProcessBuilder:
        return SshProcessBuilder(self)

    def lock(self, path: Path, max_delay: int = -1) -> Lock:
        """Returns a lock on a file"""
        raise NotImplementedError()

    def resolve(self, path: Path, basepath: Path = None):
        raise NotImplementedError()

    def setExecutable(self, path: Path, flag: bool):
        raise NotImplementedError()

    def createtoken(self, name: str, total: int) -> Token:
        """Returns a token in the default path for the connector"""
        raise NotImplementedError()
