"""Pre-experiment script that mocks a module."""

import sys
from unittest.mock import MagicMock

# Mock a module that doesn't exist
sys.modules["xpm_fake_module"] = MagicMock()
sys.modules["xpm_fake_module"].value = 42
