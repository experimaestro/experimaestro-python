from experimaestro import argument, task, parse_commandline


@argument("value", type=int)
@task("testinteger")
class IntegerTask:
    def execute(self):
        if not isinstance(self.value, int):
            raise AssertionError("Not an integer")


@argument("value", type=float)
@task("testfloat")
class FloatTask:
    def execute(self):
        if not isinstance(self.value, float):
            raise AssertionError("Not a float but %s" % type(self.value))


if __name__ == "__main__":
    parse_commandline()
