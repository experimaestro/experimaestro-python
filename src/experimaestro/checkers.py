from experimaestro.core.arguments import TypeAnnotation, ArgumentOptions


class Checker(TypeAnnotation):
    def check(self, value):
        """Check the value"""
        raise NotImplementedError()

    def annotate(self, options: ArgumentOptions):
        assert options.kwargs.get("checker", None) is None
        options.kwargs["checker"] = self


class Choices(Checker):
    def __init__(self, choices: list):
        self.choices = choices

    def check(self, value):
        for choice in self.choices:
            if value == choice:
                return True
        return False

    def __str__(self):
        return "value in " + ", ".join(str(choice) for choice in self.choices)
