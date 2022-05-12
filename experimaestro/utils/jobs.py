import time
from experimaestro.scheduler import JobState
from experimaestro.core.objects import TaskOutput
from experimaestro.scheduler import Listener
from threading import Condition


def jobmonitor(output: TaskOutput):
    """Follow the progress of a job"""
    from tqdm.autonotebook import tqdm

    cv = Condition()
    job = output.__xpm__.job

    class LocalListener(Listener):
        def job_state(self, job):
            with cv:
                # Just notify when something happens
                cv.notify()

    listener = LocalListener()

    try:
        lastprogress = 0
        while job.scheduler is None:
            time.sleep(0.1)

        job.scheduler.addlistener(listener)

        while job.state.notstarted():
            with cv:
                cv.wait(timeout=5000)

        if not job.state.finished():
            with tqdm(total=10000) as reporter:
                while not job.state.finished():
                    if job.progress and job.progress[0]:
                        delta = int(job.progress[0].progress * 100) - lastprogress
                    else:
                        delta = 0

                    if delta >= 0:
                        reporter.update(delta)

                    # Wait for an event
                    with cv:
                        cv.wait(timeout=5000)

        if job.state.value != JobState.DONE.value:
            raise RuntimeError(
                f"Job did not complete successfully ({job.state.name}). Check the error log {job.stderr}"
            )

    finally:
        job.scheduler.removelistener(listener)
