import warnings
from typing import Iterable, Optional, Union

import torch.nn as nn

from hat.exceptions import ModuleConversionWarning

# noinspection PyProtectedMember
from hat.modules._base import HATPayloadCarrierMixin
from hat.utils import convert_to_task_dependent_module


def convert_children_to_task_dependent_modules(
    module: nn.Module,
    exclude: Optional[Union[str, Iterable[str]]] = None,
    **kwargs,
):
    """Recursively convert all children of a module to task-dependent modules.

    Note that it will not convert the module itself. This function does not
    return anything, but it will modify the module in-place.

    Args:
        module: The module whose children will be converted.
        exclude: The names of the children to be excluded from conversion,
            could be a string or an iterable of strings. Defaults to `None`.
        **kwargs: The keyword arguments to be passed to the conversion
            function. See `hat.utils.convert_to_task_dependent_module` for
            more details.

    """
    if exclude is None:
        exclude = []
    elif isinstance(exclude, str):
        exclude = [exclude]
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=ModuleConversionWarning,
        )
        for __n, __c in module.named_children():
            if __n in exclude:
                continue
            if isinstance(__c, HATPayloadCarrierMixin):
                continue
            __c = convert_to_task_dependent_module(__c, **kwargs)
            module.__setattr__(__n, __c)
