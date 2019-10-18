import tempfile
import shutil
class TemporaryDirectory:
    def __enter__(self):
        self.path = tempfile.mkdtemp()
        return self.path

    def __exit__(self, exc_type, exc_value, traceback):
        print("WARNNNINNNGG - NOT Removing %s" %  self.path)
        # shutil.rmtree(self.path)
