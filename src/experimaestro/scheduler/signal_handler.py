import signal
from typing import Set
from experimaestro.scheduler import experiment
from experimaestro.utils import logger


class SignalHandler:
    def __init__(self):
        self.experiments: Set["experiment"] = set()
        self.original_sigint_handler = None

    def add(self, xp: "experiment"):
        if not self.experiments:
            self.original_sigint_handler = signal.getsignal(signal.SIGINT)

            signal.signal(signal.SIGINT, self)

        self.experiments.add(xp)

    def remove(self, xp):
        self.experiments.remove(xp)
        if not self.experiments:
            signal.signal(signal.SIGINT, self.original_sigint_handler)

    def __call__(self, signum, frame):
        """SIGINT signal handler"""
        logger.warning("Signal received")
        for xp in self.experiments:
            xp.stop()


SIGNAL_HANDLER = SignalHandler()
