"""Atomic file writes, safe for files shared between processes and hosts

Readers of a file written with :func:`atomic_write` see either the previous
content or the new one, never an empty or partially written file. This holds on
local file systems as well as on network/cluster ones (NFS, Spectrum Scale/GPFS,
Lustre):

- the content goes to a uniquely named temporary file *in the same directory*
  (renames are only atomic within a file system, and a unique name avoids
  collisions between writers on different hosts), which is then renamed over
  the target with :func:`os.replace` — a single atomic operation on the server
  for NFS;
- temporary files are named ``.<name>.<random>.tmp`` so that folder watchers
  that look for e.g. ``*.json`` files ignore them;
- the permissions of the replaced file are kept (otherwise the umask applies,
  as for a plain ``open``), so that group-shared workspaces stay shared;
- ``fsync=True`` flushes the data to stable storage before the rename. It is
  off by default since it is costly (one disk flush, or one round-trip to the
  server) and only protects against machine crashes; use it for files whose
  loss cannot be recovered (job markers, parameters).

:func:`atomic_create` only creates the file if it does not exist yet, using a
hard link (the NFS-safe equivalent of ``O_EXCL``).

On NFS, readers may still see the previous version for a few seconds
(attribute caching), and get ``ESTALE`` when the file they opened has been
replaced: :func:`read_json` retries in that case.
"""

import errno
import json
import logging
import os
import secrets
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any

logger = logging.getLogger("xpm.atomic")

#: Errors raised by :func:`os.link` on file systems without hard links
_NO_HARDLINK_ERRNOS = {errno.EPERM, errno.ENOTSUP, errno.EOPNOTSUPP, errno.ENOSYS}

#: Sentinel for :func:`read_json` (no default value)
_MISSING = object()


def _create_temporary(path: Path) -> tuple[int, Path]:
    """Exclusively creates a uniquely named temporary file next to ``path``"""
    while True:
        tmp = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
        try:
            # 0o666 lets the umask apply, as with a plain open()
            return os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666), tmp
        except FileExistsError:
            continue


@contextmanager
def _temporary(
    path: Path, mode: str, fsync: bool, encoding: str | None
) -> Iterator[tuple[IO, Path]]:
    """Opens a temporary file next to ``path``, removed if anything fails"""
    if mode not in ("w", "wt", "wb"):
        raise ValueError(f"Unsupported mode for an atomic write: {mode!r}")

    fd, tmp = _create_temporary(path)
    try:
        # Keep the permissions of the file being replaced
        try:
            os.chmod(tmp, os.stat(path).st_mode & 0o7777)
        except FileNotFoundError:
            pass

        if "b" not in mode and encoding is None:
            encoding = "utf-8"
        with os.fdopen(fd, mode, encoding=encoding) as fp:
            yield fp, tmp
            if fsync:
                fp.flush()
                os.fsync(fp.fileno())
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


@contextmanager
def atomic_open(
    path: Path | str,
    mode: str = "w",
    *,
    fsync: bool = False,
    encoding: str | None = None,
) -> Iterator[IO]:
    """Opens a file for writing, atomically replacing ``path`` on success

    If the ``with`` block raises, ``path`` is left untouched.

    :param path: The file to write
    :param mode: ``"w"`` (text, the default) or ``"wb"`` (binary)
    :param fsync: Flush the data to stable storage before replacing ``path``
    :param encoding: Text encoding (defaults to UTF-8)
    """
    path = Path(path)
    with _temporary(path, mode, fsync, encoding) as (fp, tmp):
        yield fp
    try:
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def atomic_write(path: Path | str, data: str | bytes, *, fsync: bool = False) -> None:
    """Atomically replaces the content of ``path`` (see :func:`atomic_open`)"""
    with atomic_open(path, "wb" if isinstance(data, bytes) else "w", fsync=fsync) as fp:
        fp.write(data)


def atomic_write_json(
    path: Path | str, data: Any, *, fsync: bool = False, **json_kwargs
) -> None:
    """Atomically writes ``data`` as JSON (see :func:`atomic_open`)

    ``json_kwargs`` are passed to :func:`json.dump`
    """
    with atomic_open(path, fsync=fsync) as fp:
        json.dump(data, fp, **json_kwargs)


def atomic_create(path: Path | str, data: str | bytes, *, fsync: bool = False) -> bool:
    """Creates ``path`` with ``data`` only if it does not exist yet

    The file appears with its full content. Uses a hard link from a temporary
    file, which (unlike ``O_EXCL``) is reliable on NFS; on file systems without
    hard links, falls back to an exclusive create followed by a write (the
    file may then briefly be seen empty).

    :return: True if the file was created, False if it already existed
    """
    path = Path(path)
    mode = "wb" if isinstance(data, bytes) else "w"
    with _temporary(path, mode, fsync, None) as (fp, tmp):
        fp.write(data)

    try:
        os.link(tmp, path)
        return True
    except FileExistsError:
        # On NFS, a retransmitted LINK request may fail although the first
        # one succeeded: the temporary file then has two links
        return os.stat(tmp).st_nlink == 2
    except OSError as e:
        if e.errno not in _NO_HARDLINK_ERRNOS:
            raise
        logger.debug("No hard links for %s (%s), using O_EXCL", path, e)
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
        except FileExistsError:
            return False
        with os.fdopen(fd, mode) as out:
            out.write(data)
        return True
    finally:
        tmp.unlink(missing_ok=True)


def read_json(
    path: Path | str,
    *,
    default: Any = _MISSING,
    retries: int = 3,
    delay: float = 0.05,
) -> Any:
    """Reads a JSON file written by another process or host

    Retries when the file is stale (``ESTALE`` on NFS, when it has been
    replaced since it was opened) or not valid JSON (an older writer that is
    not atomic may be writing it).

    :param default: Returned if the file does not exist or stays invalid
        (otherwise the error is raised)
    :param retries: Number of retries
    :param delay: Delay between two retries, in seconds
    """
    path = Path(path)
    for attempt in range(retries + 1):
        last = attempt == retries
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            if default is _MISSING:
                raise
            return default
        except json.JSONDecodeError:
            if last:
                if default is _MISSING:
                    raise
                logger.warning("Invalid JSON in %s, using the default", path)
                return default
        except OSError as e:
            if e.errno != errno.ESTALE or last:
                raise
        time.sleep(delay)
