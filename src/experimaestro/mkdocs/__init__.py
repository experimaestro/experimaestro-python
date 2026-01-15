try:
    from .base import Documentation  # noqa: F401

    MKDOCS_AVAILABLE = True
except ImportError:
    Documentation = None  # type: ignore
    MKDOCS_AVAILABLE = False
