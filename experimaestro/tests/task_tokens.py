from pathlib import Path
import time

from experimaestro import task, Param


@task()
class TokenTask:
    path: Param[Path]
    x: Param[int]

    def execute(self):
        print(time.time())
        while not self.path.is_file():
            time.sleep(1)
        print(time.time())
