__all__ = ['obs', 'models', 'opts', '__version__']

import pkg_resources
from . import driver
from . import obs
from . import models
from . import opts

try:
    __version__ = pkg_resources.get_distribution("xarray").version
except Exception:
    # Local copy or not installed with setuptools.
    # Disable minimum version checks on downstream libraries.
    __version__ = "999"
