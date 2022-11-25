import typing_extensions

class TypeConfig:
    def __validate__(self):
        pass

@typing_extensions.dataclass_transform(kw_only_default=True)
class Config(TypeConfig): ...
