# Development documentation

Notes and tools about how experimaestro is organized and why.

## Planned work

See [planned.md](planned.md): standalone libraries, the execution library,
job completion hooks and the egress proxy. The documents themselves are in
[planned/](planned/).

## Reference

- [slurm.md](slurm.md): how SLURM priorities, fairshare and QOS work, with the
  commands to inspect them.

## Tools

- `fswatch_bench.py`: measures how fast a host notices files created or
  appended by another host on a shared filesystem (GPFS, Lustre, NFS), for
  each watching strategy. Its results led to the pluggable change sources in
  `filewatcher.py`.
- `probe_egress.py`: checks which route out to the internet a cluster's
  compute nodes have (direct, the site proxy, or a relay on the login node).
  Run it on the login node: `uv run dev/probe_egress.py -- <sbatch options>`.
