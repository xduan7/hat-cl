from copy import deepcopy

import torch.nn as nn

from hat.modules import HATMasker
from hat.types_ import ForgetResult

from .forgetting import forget_task


def prune_hat_module(
    module: nn.Module,
    task_id: int,
) -> tuple[nn.Module, ForgetResult]:
    """Prune a HAT module so that it will forget all the tasks except
    the given task.

    This function essentially calls the `forget_task` function on all
    the tasks except the given task in iteration. The forgetting result
    of each iteration will be accumulated and returned.

    Args:
        module: The HAT module to be pruned.
        task_id: The task id to be kept.

    Returns:
        A tuple of the pruned module and the forgetting result. See
        `hat.types_.ForgetResult` for more details.

    """
    _module = deepcopy(module)
    _trained_task_ids = None
    for __m in _module.modules():
        if isinstance(__m, HATMasker):
            _trained_task_ids = __m.trained_task_ids
            break
    if _trained_task_ids is None:
        raise ValueError(
            "The module does not have a HATMasker, "
            "therefore cannot be pruned by task."
        )
    if task_id not in _trained_task_ids:
        raise ValueError(
            f"The task_id {task_id} is not in the trained task ids, "
            f"therefore cannot be pruned by such task id."
        )
    forget_result = ForgetResult()
    for __task_id in _trained_task_ids:
        if __task_id != task_id:
            forget_result += forget_task(
                module=_module,
                task_id=__task_id,
                locked_task_ids=[task_id],
            )
    return _module, forget_result
