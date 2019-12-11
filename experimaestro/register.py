"""Command line parsing"""

import sys
import json
from .api import ObjectType, TypeInformation

def parse_commandline(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    taskid, params = argv
    tasktype = ObjectType.REGISTERED[taskid]
    with open(params, "r") as fp:
        params = json.load(fp)
        TypeInformation.LOADING = True
        task = tasktype(**params)
        TypeInformation.LOADING = False
        task.execute()