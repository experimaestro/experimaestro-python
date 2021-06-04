from build.lib.experimaestro.scheduler import JobState
from experimaestro.core.objects import TaskOutput
from experimaestro.scheduler import Listener
from threading import Condition


def jobmonitor(output: TaskOutput):
    """Follow the progress of a job"""
    from tqdm import tqdm

    cv = Condition()
    job = output.__xpm__.job

    class LocalListener(Listener):
        def job_state(self, job):
            with cv:
                cv.notify()

    listener = LocalListener()

    try:
        lastprogress = 0
        job.scheduler.addlistener(listener)

        with tqdm(total=10000) as reporter:
            while not job.state.finished():
                delta = int(job.progress * 100) - lastprogress
                if delta >= 0:
                    reporter.update(delta)
                with cv:
                    cv.wait(timeout=5000)

        if job.state.value != JobState.DONE.value:
            raise RuntimeError(
                f"Job did not complete successfully ({job.state.name}). Check the error log {job.stderr}"
            )

    finally:
        job.scheduler.removelistener(listener)
