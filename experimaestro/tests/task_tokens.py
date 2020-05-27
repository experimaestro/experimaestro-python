from pathlib import Path
import time
import logging

from experimaestro import task, argument, pathoption


@argument("x", int)
@argument("path", Path)
@task()
def TokenTask(path: Path, x: int):
    print(time.time())
    while not path.is_file():
        time.sleep(1)
    print(time.time())
