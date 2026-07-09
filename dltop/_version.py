"""Single source of the installed package version."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("dltop")
except PackageNotFoundError:  # running from a checkout without an install
    __version__ = "0.0.0+uninstalled"
