"""HAT (Hard attention to the task) package for continual learning."""
from .parameter import TaskIndexedParameter
from .payload import HATPayload
from .types_ import HATConfig

__all__ = [
    "HATPayload",
    "HATConfig",
    "TaskIndexedParameter",
    "modules",
    "timm_models",
]
