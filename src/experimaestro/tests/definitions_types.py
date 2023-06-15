from experimaestro import argument, Task


@argument("value", type=int)
class IntegerTask(Task):
    def execute(self):
        if not isinstance(self.value, int):
            raise AssertionError("Not an integer")


@argument("value", type=float)
class FloatTask(Task):
    def execute(self):
        if not isinstance(self.value, float):
            raise AssertionError("Not a float but %s" % type(self.value))
