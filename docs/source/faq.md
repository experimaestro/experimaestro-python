# Frequently Asked Questions

## Controlling tasks

### Wait a for a task to complete

```python
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

```python
context = DirectoryContext("/tmp/taskfolder")
task.instance(context).execute()
```

The main problem with this approach is that resources are shared between experimaestro and the task

### How to Debug a failed task ?
If a task failed, you can rerun it with [debugpy](https://github.com/microsoft/debugpy).

#### Using vsCode
If the task is already generated, you can run it with the [python debugger](https://code.visualstudio.com/docs/python/debugging) directly within vsCode.
- open the task python file `.../HASHID/task_name.py`.
- Run the dubugger Using the following configuration:

In `.vscode/launch.json` :
```json5
 {
            "name": "Python: XPM Task",
            "type": "debugpy",
            "request": "launch",
            "module": "experimaestro",
            "console": "integratedTerminal",
            "justMyCode": false,
            "args": [
                "run",
                "params.json"
            ],
            // "python": "${workspaceFolder}/.venv/bin/python",
            "cwd": "${fileDirname}",
            "env": {
                "CUDA_VISIBLE_DEVICES": "1",
            }
}
```
- NOTE: if the task needs GPU support, you may need to open VS-Code on a node with access to a GPU.
