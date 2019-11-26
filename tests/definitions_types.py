from experimaestro import Argument, Task, register

@Argument("value", type=int)
@Task("testinteger")
class TestInteger:
    def execute(self):
        if not isinstance(self.value, int):
            raise AssertionError("Not an integer")

@Argument("value", type=float)
@Task("testfloat")
class TestFloat:
    def execute(self):
        if not isinstance(self.value, float):
            raise AssertionError("Not a float but %s" % type(self.value))


if __name__ == "__main__":
    register.parse()
