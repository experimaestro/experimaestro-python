"""Script for running partial task in subprocess for concurrent testing"""

if __name__ == "__main__":
    import sys
    import logging
    from pathlib import Path
    import time

    from experimaestro.scheduler import JobState
    from experimaestro.tests.utils import TemporaryExperiment
    from experimaestro.tests.task_partial import PartialTask

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    logging.getLogger("xpm").setLevel(logging.DEBUG)

    workdir, x, lockingpath, readypath, timepath = sys.argv[1:]

    handler = logging.StreamHandler()
    bf = logging.Formatter(
        f"[XP{x}] [%(levelname)s] %(asctime)s.%(msecs)03d %(name)s "
        f"[%(process)d/%(threadName)s]: %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(bf)
    root.handlers.clear()
    root.addHandler(handler)

    with TemporaryExperiment("partial_reschedule%s" % x, workdir=workdir) as xp:
        logging.info("Partial reschedule [%s]: starting task in %s", x, workdir)
        task = PartialTask.C(path=lockingpath, x=int(x)).submit()

        logging.info("Waiting for task (partial with %s) to be scheduled", lockingpath)
        while task.job.state == JobState.UNSCHEDULED:
            time.sleep(0.01)

        # Write so that the test knows we are ready
        Path(readypath).write_text("hello")
        logging.info("Partial reschedule [%s]: ready", x)

        # Wait until the experiment finishes
        task.__xpm__.task.job.wait()
        logging.info("Partial reschedule [%s]: finished", x)

        # Write the timestamp from the task so the test can retrieve them easily
        Path(timepath).write_text(Path(task.stdout()).read_text())
