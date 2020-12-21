from .all import ForeignClassB1, argument, config


@argument("y", type=int)
@config()
class ForeignClassB2(ForeignClassB1):
    pass
