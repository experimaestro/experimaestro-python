import asyncio
from pynng import Pair1
from experimaestro.connectors.local import LocalConnector
from experimaestro.utils import logger
from pynng import Pair1
import pynng.exceptions
import sys

class Server():
    """Use to communicate between different experimaestro processes"""
    def __init__(self):
        connector = LocalConnector()
        self.address = connector.ipcsocket

    async def run(self):
        logger.info("Serving on %s", self.address)
        try:
            with Pair1(listen=self.address, polyamorous=True) as s:
                while True:
                    logger.info("Waiting for next message...")
                    msg = await s.arecv_msg()
                    logger.info("Got one...")
                    print(msg)

        except pynng.exceptions.AddressInUse:
            logger.info("Server already running")

def serve():
    asyncio.run(Server().run())

def client():
    local = LocalConnector()
    builder = local.processbuilder()
    builder.detach = True
    builder.command = [sys.executable, "-m", "experimaestro", "serve"]
    builder.start()

    ipc = Pair1(dial=local.ipcsocket, polyamorous=True)
    ipc.__enter__()
    ipc.send_msg(b'HELLO')
    logger.info("Connected to %s" % local.ipcsocket)
    return ipc