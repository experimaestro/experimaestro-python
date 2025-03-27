from experimaestro import Param, Task


class IntegerTask(Task):
    value: Param[int]

    def execute(self):
        if not isinstance(self.value, int):
            raise AssertionError("Not an integer")


class FloatTask(Task):
    value: Param[float]

    def execute(self):
        if not isinstance(self.value, float):
            raise AssertionError("Not a float but %s" % type(self.value))
