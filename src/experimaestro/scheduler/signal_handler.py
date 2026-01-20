import signal
from typing import Set, TYPE_CHECKING
from experimaestro.utils import logger

if TYPE_CHECKING:
    from experimaestro.scheduler.experiment import experiment as Experiment


class SignalHandler:
    def __init__(self):
        self.experiments: Set["Experiment"] = set()
        self.original_sigint_handler = None

    def add(self, xp: "Experiment"):
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
