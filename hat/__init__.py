"""HAT (Hard attention to the task) package for continual learning."""
from .payload import HATPayload
from .types_ import HATConfig

__all__ = [
    "HATPayload",
    "HATConfig",
    "modules",
    "models",
]
