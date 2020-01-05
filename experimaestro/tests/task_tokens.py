from pathlib import Path
import time

from experimaestro import task, argument

@argument("path", Path)
@task()
def TokenTask(path: Path):
    print(time.time())
    while not path.is_file():
        time.sleep(0.1)
    print(time.time())
