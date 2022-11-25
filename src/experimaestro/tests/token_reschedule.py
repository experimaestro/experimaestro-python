if __name__ == "__main__":
    import sys
    import logging
    from pathlib import Path
    import time

    from experimaestro.scheduler import JobState
    from experimaestro.tests.utils import (
        TemporaryExperiment,
        TemporaryDirectory,
        timeout,
    )
    from experimaestro.tests.task_tokens import TokenTask

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    logging.getLogger("xpm").setLevel(logging.DEBUG)

    workdir, x, lockingpath, readypath, timepath = sys.argv[1:]

    handler = logging.StreamHandler()
    bf = logging.Formatter(
        f"[XP{x}] "
        + "[%(levelname)s] %(asctime)s %(name)s [%(process)d/%(threadName)s]: %(message)s",
        datefmt="%H:%M:%S.%f",
    )
    handler.setFormatter(bf)
    root.handlers.clear()
    root.addHandler(handler)

    with TemporaryExperiment("reschedule%s" % x, workdir=workdir) as xp:
        logging.info("Reschedule with token [%s]: starting task in %s", x, workdir)
        token = xp.workspace.connector.createtoken("test-token-reschedule", 1)
        task = (
            TokenTask(path=lockingpath, x=int(x))
            .add_dependencies(token.dependency(1))
            .submit()
        )
        logging.info("Waiting for task (token with %s) to be scheduled", lockingpath)
        while task.job.state == JobState.UNSCHEDULED:
            time.sleep(0.01)

        # Write so that the test now we are ready
        Path(readypath).write_text("hello")
        logging.info("Reschedule with token [%s]: ready", x)

        # Wait until the experiment
        task.__xpm__.task.job.wait()
        logging.info("Reschedule with token [%s]: finished", x)
        Path(timepath).write_text(Path(task.__xpm__.stdout()).read_text())
