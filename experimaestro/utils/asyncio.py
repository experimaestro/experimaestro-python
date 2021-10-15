from threading import Thread
import asyncio


def asyncThreadcheck(name, func, *args, **kwargs) -> asyncio.Future:
    """Launch a thread that will return a future"""
    loop = asyncio.get_running_loop()
    future = loop.create_future()

    def dowait():
        result = func(*args, **kwargs)
        loop.call_soon_threadsafe(future.set_result, result)

    # Start thread
    Thread(name=name, target=dowait).start()
    return future
