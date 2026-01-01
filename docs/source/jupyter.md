Experimaestro have several utility functions that helps
integrating with jupyter

## Starting the experimaestro server

A widget can be used to control the starting/stopping of the
experimaestro server

```py3
from experimaestro.utils.jupyter import serverwidget

def settoken(xp):
    # Set some useful variables
    xp.token = xp.current.token("main", 1)

xp = serverwidget("ri/intro", port=12500)
```

## Monitoring a job

Jobs can be monitored directly with `jobmonitor` that will display the progress (if the
information is provided by the task) and wait for the job to complete.

```py3
from experimaestro.utils.jupyter import jobmonitor
jobmonitor(myjob.submit())
```
