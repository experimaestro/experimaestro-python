from mypy.plugin import Plugin


class ExperimaestroPlugin(Plugin):
    """Just do nothing for now"""

    def get_class_decorator_hook(self, tada):
        pass

    def get_customize_class_mro_hook(self, tada):
        pass


def plugin(version: str):
    return ExperimaestroPlugin
