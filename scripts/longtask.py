from tempfile import TemporaryDirectory
import time
from experimaestro import Task, experiment
from experimaestro.notifications import tqdm
import logging

logging.basicConfig(level=logging.INFO)


class TaskA(Task):
    def execute(self):
        # Runs 10 hours
        for _ in tqdm(range(60 * 10)):
            time.sleep(60)


if __name__ == "__main__":
    with TemporaryDirectory() as xppath:
        with experiment(xppath, "longxp", port=12349) as xp:
            taska = TaskA().tag("task", "a").submit()
            print("Standard error", taska.__xpm__.stderr())
            xp.wait()
