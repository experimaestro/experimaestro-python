import logging
import multiprocessing as mp
import time
import os
import signal
import threading


def delayed_shutdown(delay=60, *, exit_code=1, grace_period=5):
    """After *delay*'s try a graceful stop, then SIGKILL anything left.

    :param delay: Delay in seconds before killing
    :param grace_period: Delay in seconds before force-killing a child process
    :param exit_code: The exit code to use
    """

    def _killer():
        time.sleep(delay)

        logging.info("Stall dectected – killing all subprocesses")

        # 1️⃣ Try graceful termination
        for p in mp.active_children():
            # sends SIGTERM / TerminateProcess
            p.terminate()

        alive = mp.active_children()
        deadline = time.time() + grace_period
        while alive and time.time() < deadline:
            alive = [p for p in alive if p.is_alive()]
            time.sleep(0.1)

        # 2️⃣ Anything still alive? Nuke it.
        for p in alive:
            try:
                os.kill(p.pid, signal.SIGKILL)
            except OSError:
                pass

        # 3️⃣ Finally kill the parent
        os.kill(os.getpid(), signal.SIGKILL)

    # Start the thread (non blocking)
    threading.Thread(target=_killer, daemon=True).start()
