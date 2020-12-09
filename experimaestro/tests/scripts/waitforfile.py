import sys
from pathlib import Path
import time

path = Path(sys.argv[1])
while not path.is_file():
    time.sleep(0.1)
