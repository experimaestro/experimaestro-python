from typing import List
from .objects import Task
from .arguments import Param


class TaskSequence(Task):
    """Multiple task execution, the main one being the last"""

    tasks: Param[List[Task]]

    @property
    def __identifier__(self):
        return self.tasks[-1].__identifier__()

    def execute(self):
        self.tasks[-1].__tags__ = self.__tags__
        for task in self.tasks:
            task.execute()


def task_sequence(*tasks: Task):
    return TaskSequence(tasks=tasks)
