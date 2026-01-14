import logging
from threading import Thread
import asyncio

logger = logging.getLogger("xpm.asyncio")


def asyncThreadcheck(name, func, *args, **kwargs) -> asyncio.Future:
    """Launch a thread that will return a future"""
    loop = asyncio.get_running_loop()
    future = loop.create_future()

    def dowait():
        logger.debug("Running %s", func)
        try:
            result = func(*args, **kwargs)
            logger.debug("Got result from %s", func)
        except Exception:
            logger.exception("Got an error in the thread")
            raise
        loop.call_soon_threadsafe(future.set_result, result)

    # Start thread
    logger.debug("Starting thread to run %s", func)
    Thread(name=name, target=dowait).start()
    return future
