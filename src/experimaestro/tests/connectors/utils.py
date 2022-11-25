import threading


class OutputCaptureHandler:
    """Catpure the full output of a file stream"""

    def __init__(self, threaded=True):
        self._output = b""
        self.thread = None
        self.threaded = threaded

    def __call__(self, fp):
        if self.threaded:
            self.thread = threading.Thread(target=self.read, args=(fp,))
            self.thread.start()
        else:
            self.read(fp)

    @property
    def output(self):
        if self.thread:
            self.thread.join()
        return self._output

    def read(self, fp):
        while True:
            data = fp.read()
            if data == b"":
                break
            self._output += data
