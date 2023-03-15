import torch.nn as nn

from ._base import TaskDependentModuleABC

base_to_task_dependent_mapping: dict[
    type[nn.Module], type[TaskDependentModuleABC]
] = {}


def register_mapping(cls):
    """Register a HAT module class to the global registry."""
    # This import assures the correct mapping is used.
    # noinspection PyUnresolvedReferences
    from hat.modules.utils import base_to_task_dependent_mapping as mapping

    if cls.base_class in mapping:
        if mapping[cls.base_class] != cls:
            raise ValueError(
                f"Base class {cls.base_class} is already registered to "
                f"{mapping[cls.base_class]}, therefore it cannot be "
                f"registered to {cls}."
            )
    else:
        mapping[cls.base_class] = cls
    return cls
