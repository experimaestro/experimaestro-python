"""IPC utilities"""

from typing import Optional
from pathlib import Path
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import sys
from .utils import logger


class IPCom:
    """IPC async thread"""

    INSTANCE: Optional["IPCom"] = None

    def __init__(self):
        self.observer = Observer()
        self.observer.start()
        self.pid = os.getpid()

    def fswatch(self, watcher: FileSystemEventHandler, path: Path, recursive=False):
        if not self.observer.is_alive():
            self.observer.start()
        return self.observer.schedule(watcher, path, recursive=recursive)

    def fsuwatch(self, watcher):
        self.observer.unschedule(watch)

    # def run(self):
    #     logger.info("Starting IPC thread")
    #     self.loop = asyncio.new_event_loop()
    #     asyncio.set_event_loop(self.loop)
    #     self.loop.run_forever()


def fork_childhandler():
    if IPCom.INSTANCE:
        logger.warning(
            "Removing IPCom instance in child process (watchers won't be copied)"
        )
        IPCom.INSTANCE = None


if sys.version_info[0] == 3 and sys.version_info[1] >= 7:
    os.register_at_fork(after_in_child=fork_childhandler)


def ipcom():
    # If multiprocessing, remove the IPCom instance if this does not
    # belong to our process
    if IPCom.INSTANCE is not None and IPCom.INSTANCE.pid != os.getpid():
        fork_childhandler()

    if IPCom.INSTANCE is None:
        IPCom.INSTANCE = IPCom()
        logger.info("Started IPCom instance (%s)", id(IPCom.INSTANCE))
        # IPCom.INSTANCE.start()
    return IPCom.INSTANCE
