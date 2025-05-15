from experimaestro import Task, Param, get_experiment, tag
from experimaestro.tests.utils import TemporaryDirectory, TemporaryExperiment


class TaskA(Task):
    def execute(self):
        pass


class TaskB(Task):
    task_a: Param[TaskA]
    x: Param[int]

    def execute(self):
        pass


# xp = get_experiment(id="my-xp-1")

# # Returns a list of tasks which were submitted and successful
# tasks = xp.get_tasks(myxps.evaluation.Evaluation, status=Job.DONE)

# for task in tasks:
#     # Look at the tags
#     print(task.tags)

#     # Get some information
#     print("Task ran in {task.workdir}")

#     # Look at the parent jobs
#    print(task.depends_on)

#    # Look at the dependant
#    print(task.dependents)


def test_experiment_history():
    """Test retrieving experiment history"""
    with TemporaryDirectory() as workdir:
        with TemporaryExperiment("experiment", workdir=workdir):
            task_a = TaskA().submit()
            TaskB(task_a=task_a, x=tag(1)).submit()

        # Look at the experiment
        xp = get_experiment("experiment", workdir=workdir)

        (task_a_info,) = xp.get_jobs(TaskA)
        (task_b_info,) = xp.get_jobs(TaskB)
        assert task_b_info.tags == {"x": 1}
        assert task_b_info.depends_on == [task_a_info]


class FlagHandler:
    def __init__(self):
        self.flag = False

    def set(self):
        self.flag = True

    def is_set(self):
        return self.flag


def test_experiment_events():
    """Test handlers"""

    flag = FlagHandler()
    with TemporaryExperiment("experiment"):
        task_a = TaskA()
        task_a.submit()
        task_a.on_completed(flag.set)

    assert flag.is_set()
