from pathlib import Path
import time
from experimaestro import Task, Param
import logging

logging.basicConfig(level=logging.INFO)


class TokenTask(Task):
    """Wait until the file is given"""

    path: Param[Path]
    x: Param[int]

    def execute(self):
        print(time.time())
        while not self.path.is_file():
            time.sleep(0.1)
        print(time.time())
