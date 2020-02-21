from pathlib import Path, PurePosixPath, PosixPath
import io
import os
import paramiko


class SshPath(Path, PurePosixPath):
    def __new__(cls, _accessor, *parts):
        self = SshPath._from_parts(parts)
        self._accessor = _accessor
        return self

    def __str__(self):
        return "ssh://%s/%s" % (self._accessor, super().__str__())

    def _make_child_relpath(self, part):
        child = super()._make_child_relpath(part)
        child._accessor = self._accessor
        return child

    def _make_child(self, args):
        child = super()._make_child(args)
        child._accessor = self._accessor
        return child

    def open(self, mode="r", buffering=-1, encoding=None, errors=None, newline=None):
        """
        Open the file pointed by this path and return a file object, as
        the built-in open() function does.
        """
        if self._closed:
            self._raise_closed()

        fileobj = self._accessor.sftp.open(super().__str__(), mode, buffering)
        if "b" in mode:
            return fileobj

        return io.TextIOWrapper(
            fileobj, encoding=encoding, newline=newline, errors=errors
        )


class SshConnector:
    def __init__(self, hostname: str):
        self.hostname = hostname
        self.port = None

        self.client = paramiko.SSHClient()
        self.client.load_system_host_keys()
        self.client.connect(hostname)
        self.sftp = self.client.open_sftp()

    def __str__(self):
        return "%s" % self.hostname

    def listdir(self, path: SshPath):
        return self.sftp.listdir(PurePosixPath.__str__(path))
