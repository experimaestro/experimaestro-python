"""IPC utilities"""

from typing import Optional
from pathlib import Path

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from .utils import logger

class IPCom:
    """IPC async thread"""
    
    INSTANCE: Optional["IPCom"] = None

    def __init__(self):
        # Initialize so that we stop when the main thread exits
        # super().__init__(daemon=True)
        self.observer = Observer()
        self.observer.start()

    def fswatch(self, watcher: FileSystemEventHandler, path: Path, recursive=False):
        return self.observer.schedule(watcher, path, recursive=recursive)
    
    def fsuwatch(self, watcher):
        self.observer.unschedule(watch)

    # def run(self):
    #     logger.info("Starting IPC thread")
    #     self.loop = asyncio.new_event_loop()
    #     asyncio.set_event_loop(self.loop)
    #     self.loop.run_forever()

def ipcom():
    if IPCom.INSTANCE is None:
        IPCom.INSTANCE = IPCom()
        # IPCom.INSTANCE.start()
    return IPCom.INSTANCE

