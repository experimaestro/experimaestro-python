from experimaestro import Argument, Task, parse_commandline

@Argument("value", type=int)
@Task("testinteger")
class IntegerTask:
    def execute(self):
        if not isinstance(self.value, int):
            raise AssertionError("Not an integer")

@Argument("value", type=float)
@Task("testfloat")
class FloatTask:
    def execute(self):
        if not isinstance(self.value, float):
            raise AssertionError("Not a float but %s" % type(self.value))


if __name__ == "__main__":
    parse_commandline()
