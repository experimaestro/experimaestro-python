import logging
from threading import Thread
import asyncio


def asyncThreadcheck(name, func, *args, **kwargs) -> asyncio.Future:
    """Launch a thread that will return a future"""
    loop = asyncio.get_running_loop()
    future = loop.create_future()

    def dowait():
        logging.debug("Running %s", func)
        result = func(*args, **kwargs)
        logging.debug("Got result from %s", func)
        loop.call_soon_threadsafe(future.set_result, result)

    # Start thread
    logging.debug("Starting thread to run %s", func)
    Thread(name=name, target=dowait).start()
    return future
