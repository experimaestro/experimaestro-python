from mypy.plugin import Plugin


class ExperimaestroPlugin(Plugin):
    def get_class_decorator_hook(self, tada):
        print("YEY", tada, type(tada))

    def get_customize_class_mro_hook(self, tada):
        print("YEY", tada, type(tada))


def plugin(version: str):
    return ExperimaestroPlugin
