"""Command line parsing"""

import sys
import json
from .core.types import ObjectType
from .core.objects import ConfigInformation


def parse_commandline(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    taskid, params = argv
    tasktype = ObjectType.REGISTERED[taskid]
    with open(params, "r") as fp:
        params = json.load(fp)
        ConfigInformation.LOADING = True
        task = tasktype(**params)
        ConfigInformation.LOADING = False
        task.execute()
