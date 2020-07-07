from pathlib import Path, PurePosixPath, PosixPath, _posix_flavour
from typing import Union
import io
import os
from fabric import Connection
from urllib.parse import urlparse
from itertools import chain
from . import Connector
from . import (
    Connector,
    Process,
    ProcessBuilder,
    RedirectType,
    Redirect,
    ProcessThreadError,
)
from experimaestro.locking import Lock
from experimaestro.tokens import Token, CounterToken

class SshPath(Path, PurePosixPath):
    """SSH path

    Absolute:
    ssh://[user@]host[:port]//this/is/a/path

    Relative:
    ssh://[user@]host[:port]/relative/path
    """
    @property
    def hostpath(self):
        # path = "/" if self._parts[:0]  + "/".join(self._parts[1:])
        if self.is_absolute():
            return "/" + self._flavour.join(self._parts[1:])
        return self._flavour.join(self._parts)

    @property
    def host(self):
        return self._drv

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
        assert self._drv == drv or drv == "", f"{self._drv} and {drv}"
        drv, root, parts = self._flavour.join_parsed_parts(
            "", self._root, self._parts, "", root, parts)
        return self._from_parsed_parts(self._drv, root, parts)

    def open(self, mode="r", buffering=-1, encoding=None, errors=None, newline=None):
        # FIXME: should probably be wiser
        fileobj = SshConnector.get(self.host).connection.sftp().open(self.hostpath, mode, buffering)
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
        return "SshPath(%s,%s)" % (self._drv, self._flavour.join(self._parts[1:]))

    def __str__(self):
        return "ssh://%s/%s" % (self._drv, self._flavour.join(self._parts[1:]))


class SshConnector(Connector):
    def __init__(self, hostname: str):
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
        return SshConnector.get(path.host)

    @staticmethod
    def get(hostname):
        # TODO: cache values?
        return SshConnector(hostname)

    def __str__(self):
        return "%s" % self.hostname

    def processbuilder(self) -> ProcessBuilder:
        raise NotImplementedError()

    def lock(self, path: Path, max_delay: int = -1) -> Lock:
        """Returns a lock on a file"""
        print(path.hostpath())
        raise NotImplementedError()

    def resolve(self, path: Path, basepath: Path = None):
        raise NotImplementedError()

    def setExecutable(self, path: Path, flag: bool):
        raise NotImplementedError()

    def createtoken(self, name: str, total: int) -> Token:
        """Returns a token in the default path for the connector"""
        raise NotImplementedError()