"""Udgam's local command-parser SDK; import does not load models or use the network."""
from .api import CommandParser, preview_result

__all__ = ["CommandParser", "preview_result"]
__version__ = "0.1.0"
