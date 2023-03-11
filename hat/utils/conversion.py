import warnings
from copy import deepcopy
from typing import Any

import torch.nn as nn

from hat.exceptions import ModuleConversionWarning


def convert_to_task_dependent_module(
    module: nn.Module,
    **kwargs: Any,
) -> nn.Module:
    """Helper function to convert a base module to its task dependent version.

    This function is a best effort attempt to convert a base module to its
    task dependent version. If the module is already converted, it will be
    returned as is. If the  module is a base module to one of the registered
    task dependent modules, the `from_base_module` method of the task
    dependent module class will be called to convert the module. Otherwise,
    the function will convert the children modules recursively.

    Warnings:
        This function does not guarantee that the converted module will
        run correctly (numerically or programmatically) because it doesn't
        change the forward pass of the module. The user is responsible for
        making sure the forward pass is executable after the conversion.

    Args:
        module: The base module to be converted.
        **kwargs: The keyword arguments to be passed to the
          `from_base_module` method of the task dependent module class.

    Returns:
        The converted (partially) task dependent module.

    """
    from hat.modules.utils import base_to_task_dependent_mapping as mapping

    if module.__class__ in mapping.keys():
        return mapping[module.__class__].from_base_module(module, **kwargs)
    elif module.__class__ in mapping.values():
        return module
    else:
        # When the module is neither a base module nor a task dependent
        # module, we convert its children modules.
        if len(list(module.named_children())) == 0:
            warnings.warn(
                f"Module of class {module.__class__} has no registered "
                f"task dependent module. It will be returned as is.",
                ModuleConversionWarning,
            )
            return module
        _module = deepcopy(module)
        for __n, __m in module.named_children():
            __m = convert_to_task_dependent_module(__m, **kwargs)
            setattr(_module, __n, __m)
        return _module


def convert_to_base_module(
    module: nn.Module,
    **kwargs: Any,
) -> nn.Module:
    """Helper function to convert a task dependent module to its base version.

    This function is a best effort attempt to convert a task dependent module
    to its base version. If the module is already converted, it will be
    returned as is. If the module is a registered task dependent module, the
    `to_base_module` method of the task dependent module class will be called
    to convert the module. Otherwise, the function will convert the children
    modules recursively.

    Warnings:
        This function does not guarantee that the converted module will
        run correctly (numerically or programmatically) because it doesn't
        change the forward pass of the module. The user is responsible for
        making sure the forward pass is executable after the conversion.

    Args:
        module: The task dependent module to be converted.
        **kwargs: The keyword arguments to be passed to the
            `to_base_module` method of the task dependent module class.

    Returns:
        The converted (partially) base module.

    """

    from hat.modules.utils import base_to_task_dependent_mapping as mapping

    if module.__class__ in mapping.values():
        return module.__class__.to_base_module(module, **kwargs)
    elif module.__class__ in mapping.keys():
        return module
    else:
        # When the module is neither a base module nor a task dependent
        # module, we convert its children modules.
        if len(list(module.named_children())) == 0:
            warnings.warn(
                f"Module of class {module.__class__} is not a base "
                f"module for a registered task dependent module. "
                "It will be returned as is.",
                ModuleConversionWarning,
            )
            return module
        _module = deepcopy(module)
        for __n, __m in module.named_children():
            __m = convert_to_base_module(__m, **kwargs)
            setattr(_module, __n, __m)
        return _module
