import time
import sys
from experimaestro import task, argument, pathargument, experiment

if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)


@pathargument("touch", "touch")
@pathargument("wait", "wait")
@task("restart")
class Restart:
    def execute(self):
        # Write the file "touch" to notify that we started
        with open(self.touch, "w") as out:
            out.write("hello")

        # Wait for the file "wait" before exiting
        while not self.wait.is_file():
            time.sleep(0.1)


if __name__ == "__main__":
    with experiment(sys.argv[1], "restart"):
        Restart().submit()
