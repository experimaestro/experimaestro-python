# Configurations and tasks

## Experimental plan

An experimental plan is based on [configuration and tasks](../experiments/config), and define which tasks should be run with which parameters. Within an experiment, [tags](../experiments/plan#tags) can be used to track experimental parameters.

## Connectors

A [connector](../connectors) allow to specify how to access files on the computer where a task will be launched, and how to run processes on this computer. Two basic connectors exist, for localhost and SSH accesses (_alpha_).

## Launcher

A [launcher](../launchers) specifies how a given task can be run. The most basic method is direct execution, but experimaestro can launch and monitor oar (_planned_) and slurm (_planned_) jobs.
