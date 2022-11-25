def documentation(method):
    """Indicates that the method should be included in the documentation"""
    method.__experimaestro_documentation__ = True
    return method


def shoulddocument(method):
    return hasattr(method, "__experimaestro_documentation__")
