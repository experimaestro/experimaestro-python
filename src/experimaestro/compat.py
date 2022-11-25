import sys

if sys.version_info.major == 3 and sys.version_info.minor < 9:
    from cached_property import cached_property
else:
    from functools import cached_property
