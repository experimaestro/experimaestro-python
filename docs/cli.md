# Command Line Interface

## Job Control

Besides the web interface, it is possible to use the command line to check the job
status and control jobs.

## Cleaning up

Check for tasks that are not part of any experimental plan in the given
experimental folder.

```
Usage: experimaestro orphans [OPTIONS] PATH

Options:
  --size      Show size of each folder
  --clean     Prune the orphan folders
  --show-all  Show even not orphans
```

## Difference between two configurations

It is possible to look at the differences (that explain that two tasks have a different identifier) by using the `parameters-difference` command
