class Launcher():
    def __init__(self, connector: Connector):
        self.connector = connector
        self.environ:Dict[str,str] = {}
        self.notificationURL = None

    def setenv(self, key: str, value: str):
        self.environ[key] = value

    def setNotificationURL(self, url: str):
        self.notificationURL = url

    @staticmethod
    def get(path: Path):
        """Get a default launcher for a given path"""
        if isinstance(path, PosixPath):
            return DirectLauncher(LocalConnector())
        raise ValueError("Cannot create a default launcher for %s", type(path))


class DirectLauncher(Launcher):
    pass
