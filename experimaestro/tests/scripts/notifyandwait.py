import sys
from pathlib import Path
import time

_, notifypath, semaphorepath, *args = sys.argv
notifypath = Path(notifypath)
semaphorepath = Path(semaphorepath)

# Notify

notifypath.touch(exist_ok=False)
print("Notified", file=sys.stderr)

# ... and wait
print(f"Waiting for {semaphorepath}", file=sys.stderr)
while not semaphorepath.is_file():
    time.sleep(0.1)

print("Wait is over", file=sys.stderr)
