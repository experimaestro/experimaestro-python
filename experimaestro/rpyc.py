import atexit
import shutil
import tempfile
from pathlib import Path
from subprocess import Popen, PIPE, run
import time
from rpyc.utils.server import OneShotServer
import rpyc
import threading
import logging

logger = logging.getLogger("rpyc")
logger.setLevel(logging.WARNING)

class client():
    def __init__(self, hostname: str, pythonpath: str, port: int = None):
        """[summary]

        Arguments:
            hostname -- The hostname to connect to
            pythonpath -- The remote python path
        """
        self.hostname = hostname
        self.pythonpath = pythonpath
        self.port = port

    # Constructs the SSH command line
    def ssh(self, *args):
        command = ["ssh"]
        command.extend(args)
        if self.port:
            command.extend(["-p", str(self.port)])
        command.append(self.hostname)
        return command

    def connect(self):
        # Get the local unix_path
        tmp = tempfile.TemporaryDirectory()
        tmpdirname = tmp.__enter__()
        local_unix_path = str(Path(tmpdirname) / "rpyc-client.sock")

        # Get the remote unix_path
        command = self.ssh()
        command.extend(["mktemp", "-d"])
        logger.debug("Runnning %s", command)
        p = run(command, capture_output=True)
        p.check_returncode()
        remote_unix_path = p.stdout.decode("utf-8").strip() + "/rpyc-server.sock"
        
        # Start server
        command = self.ssh(f"-L{local_unix_path}:{remote_unix_path}")
        command.extend([self.pythonpath, "-m", "experimaestro", "rpyc-server", "--clean", remote_unix_path])
        
        logger.debug("Runnning %s", command)
        process = Popen(command, stdout=PIPE)
        atexit.register(lambda process: process.kill(), process)

        # Wait for the server to be started        
        process.stdout.readline()

        # Connect to server
        logger.info("Connecting to %s", local_unix_path)
        self.connection = rpyc.classic.unix_connect(local_unix_path)

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.info("Closing connection")
        self.connection.close()


# --- Server part

server = None

class ClassicService(rpyc.core.service.ClassicService):
    """Full duplex master/slave service, i.e. both parties have full control
    over the other. Must be used by both parties."""
    __slots__ = ("connected")
    def __init__(self):
        super().__init__()
        self.connected = False

    def on_connect(self, conn):
        self.connected = True
        super().on_connect(conn)

    def on_disconnect(self, conn):
        print("Disconnected")
        super().on_disconnect(conn)


def cleanup(path):
    logger.info("Cleaning up %s", path)
    path.unlink()
    path.parent.rmdir()

def start_server(unix_path, clean=None):
    service = ClassicService()
    server = OneShotServer(socket_path=str(unix_path), listener_timeout=1, service=service, logger=logger)
    def sayhello():
        while not server.active:
            time.sleep(.01)
        print("HELLO", flush=True)
        logger.debug("Server started")
        
        time.sleep(5)
        if not service.connected:
            logger.warning("No inbound connection: stopping")
            server.close()

    threading.Thread(target=sayhello).start()
    if clean:
        atexit.register(cleanup, unix_path)
    server.start()

def stop_server():
    server.stop()
