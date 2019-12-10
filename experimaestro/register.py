"""Command line parsing"""

import sys
import json
from .api import ObjectType

def parse_commandline(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    taskid, params = argv
    task = ObjectType.REGISTERED[taskid]
    with open(params, "r") as fp:
        params = json.load(fp)
        task(**params).execute()