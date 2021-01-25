from experimaestro import param, config
from .all import ForeignClassB1


@param("y", type=int)
@config()
class ForeignClassB2(ForeignClassB1):
    pass
