import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--fail", action="store_true")
parser.add_argument("--nodes", type=int)
parser.add_argument("--time", type=int)
parser.add_argument("--gpus-per-node", type=int)
parser.add_argument("--gpus", type=int)
opts = parser.parse_args()

if opts.fail:
    sys.exit(1)

for key, value in vars(opts).items():
    if key != "fail" and value is not None:
        print(key, value, sep="=", file=sys.stdout)
