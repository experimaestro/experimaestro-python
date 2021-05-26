## Controlling tasks

### Wait a for a task to complete

```py3
# Submit the task
output = mytask.submit()

# Wait until it completes, and returns a boolean (success flag)
output.__xpm__.wait()
```

## Debugging

### How to run a task without scheduling it?

Method 1: creates an instance of a task with its `.instance(context)` method. The context contains information which might
be necessary to create the directory structure for executing the task. If no context is provided, the default context
is used.

```py3
context = DirectoryContext("/tmp/taskfolder")
task.instance(context).execute()
```

The main problem with this approach is that resources are shared between experimaestro and the task
