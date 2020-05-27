from pathlib import Path, PurePosixPath, PosixPath, _posix_flavour
from typing import Union
import io
import os
from fabric import Connection
from urllib.parse import urlparse

class SshPath(Path, PurePosixPath):

    # def __repr__(self):
    #     return "ssh://%s/%s" % (self._connector, self._path)
    def _parse_args(cls, args):
        drv = ""
        if args[0].startswith("ssh:"):
            url = urlparse(args[0])
            assert not url.fragment and not url.query
            _args = [url.path]
            _args.extend(args[1:])
            args = _args
            drv = url.hostname
        _, root, parts = super()._parse_args(args)

        return (drv, root, parts)
        

    def open(self, mode="r", buffering=-1, encoding=None, errors=None, newline=None):
        # FIXME: should probably be wiser
        path = "/" + "/".join(self._parts[1:])
        fileobj = SshConnector.get(self._drv).connection.sftp().open(path, mode, buffering)
        if "b" in mode:
            return fileobj

        return io.TextIOWrapper(
            fileobj, encoding=encoding, newline=newline, errors=errors
        )


class SshConnector:
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
    def get(hostname):
        # TODO: cache values?
        return SshConnector(hostname)

    def __str__(self):
        return "%s" % self.hostname

    def listdir(self, path: SshPath):
        return self.sftp.listdir(PurePosixPath.__str__(path))
