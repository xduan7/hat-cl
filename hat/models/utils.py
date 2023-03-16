import warnings

import torch.nn as nn

from hat.exceptions import ModuleConversionWarning

# noinspection PyProtectedMember
from hat.modules._base import HATPayloadCarrierMixin
from hat.utils import convert_to_task_dependent_module


def convert_children_to_task_dependent_modules(
    module: nn.Module,
    **kwargs,
):
    """Recursively convert all children of a module to task-dependent modules.

    Note that it will not convert the module itself. This function does not
    return anything, but it will modify the module in-place.

    Args:
        module: The module whose children will be converted.
        **kwargs: The keyword arguments to be passed to the conversion
            function. See `hat.utils.convert_to_task_dependent_module` for
            more details.

    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=ModuleConversionWarning,
        )
        for __n, __c in module.named_children():
            if isinstance(__c, HATPayloadCarrierMixin):
                continue
            __c = convert_to_task_dependent_module(__c, **kwargs)
            module.__setattr__(__n, __c)
