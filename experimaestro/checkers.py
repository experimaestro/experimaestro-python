class Checker:
    def check(self, value):
        """Check the value"""
        raise NotImplementedError()


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
