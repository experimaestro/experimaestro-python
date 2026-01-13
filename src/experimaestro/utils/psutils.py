"""Process utilities for async process waiting

Provides true async process waiting using OS-native mechanisms:
- Linux 5.3+: pidfd_open
- macOS/BSD: kqueue with EVFILT_PROC
- Fallback: polling with psutil
"""

import asyncio
import logging
import os
import select
import sys

import psutil

logger = logging.getLogger("xpm.psutils")


async def _aio_wait_pidfd(pid: int) -> int:
    """Wait for process using pidfd (Linux 5.3+, true async)"""
    loop = asyncio.get_event_loop()
    pidfd = os.pidfd_open(pid)

    try:
        future = loop.create_future()
        loop.add_reader(pidfd, future.set_result, None)

        try:
            await future
        finally:
            loop.remove_reader(pidfd)

        _, status = os.waitpid(pid, os.WNOHANG)
        return os.waitstatus_to_exitcode(status)
    finally:
        os.close(pidfd)


async def _aio_wait_kqueue(pid: int) -> int:
    """Wait for process using kqueue (macOS/BSD, true async)"""
    loop = asyncio.get_event_loop()
    kq = select.kqueue()

    try:
        event = select.kevent(
            pid,
            filter=select.KQ_FILTER_PROC,
            flags=select.KQ_EV_ADD | select.KQ_EV_ONESHOT,
            fflags=select.KQ_NOTE_EXIT,
        )
        kq.control([event], 0)

        future = loop.create_future()
        loop.add_reader(kq.fileno(), future.set_result, None)

        try:
            await future
        finally:
            loop.remove_reader(kq.fileno())

        events = kq.control([], 1, 0)
        if events:
            return events[0].data
        return -1
    finally:
        kq.close()


async def _aio_wait_polling(pid: int) -> int:
    """Fallback: wait for process using polling"""
    poll_interval = 0.01

    try:
        proc = psutil.Process(pid)
        while proc.is_running():
            await asyncio.sleep(poll_interval)
            poll_interval = min(poll_interval * 1.5, 10.0)
        return proc.wait()
    except psutil.NoSuchProcess:
        return -1


def _has_pidfd() -> bool:
    """Check if pidfd_open is available (Linux 5.3+)"""
    return sys.platform == "linux" and hasattr(os, "pidfd_open")


def _has_kqueue() -> bool:
    """Check if kqueue is available (macOS/BSD)"""
    return sys.platform == "darwin" and hasattr(select, "kqueue")


async def aio_wait_pid(pid: int) -> int:
    """Cross-platform async wait for process exit (true async when available)

    Uses pidfd on Linux 5.3+, kqueue on macOS/BSD, falls back to polling.

    Args:
        pid: Process ID to wait for

    Returns:
        Exit code of the process
    """
    if _has_pidfd():
        try:
            return await _aio_wait_pidfd(pid)
        except OSError:
            logger.debug("pidfd_open failed for PID %s, falling back to polling", pid)
    elif _has_kqueue():
        try:
            return await _aio_wait_kqueue(pid)
        except OSError:
            logger.debug("kqueue failed for PID %s, falling back to polling", pid)

    return await _aio_wait_polling(pid)
