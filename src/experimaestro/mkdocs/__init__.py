def __getattr__(name: str):
    # Loaded on demand: the plugin pulls mkdocs and requests, and this package
    # is imported by ``experimaestro`` itself (for ``documentation``)
    if name in ("Documentation", "MKDOCS_AVAILABLE"):
        try:
            from .base import Documentation
        except ImportError:
            Documentation = None
        globals().update(
            Documentation=Documentation, MKDOCS_AVAILABLE=Documentation is not None
        )
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
