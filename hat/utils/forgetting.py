import warnings
from typing import Optional

import torch.nn as nn

from hat.exceptions import NoParameterToForgetWarning

# noinspection PyProtectedMember
from hat.modules._base import TaskDependentModuleABC
from hat.types_ import ForgetResult


def forget_task(
    module: nn.Module,
    task_id: int,
    dry_run: bool = False,
    module_name: Optional[str] = None,
    locked_task_ids: Optional[list[int]] = None,
) -> ForgetResult:
    """Forget the given tasks of a given module by calling the `forget`
    function of all `TaskDependentModuleABC` children of the module.

    Args:
        module: The module to forget the task.
        task_id: The ID of the task to be forgotten.
        dry_run: If `True`, the forgetting process will be simulated
            without actually changing the module. Defaults to `False`.
        module_name: The name of the module. If `None`, the module name
                will be inferred from the module class name.
        locked_task_ids: The list of task ids that are locked and
            cannot be forgotten. Defaults to `None`, in which case
            the module will lock all the tasks that have been trained
            except the task id to be forgotten.

    Returns:
        The forgetting result. See `hat.types_.ForgetResult` for more
        details.

    """
    if isinstance(module, TaskDependentModuleABC):
        return module.forget(
            task_id=task_id,
            dry_run=dry_run,
            module_name=module_name,
            locked_task_ids=locked_task_ids,
        )
    else:
        # When some of the children modules can forget while others cannot,
        # we still consider the task forgotten because the output will
        # change ultimately.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=NoParameterToForgetWarning
            )
            _forget_result = ForgetResult()
            _module_name = module_name or module.__class__.__name__
            for __n, __m in module.named_children():
                _forget_result += forget_task(
                    module=__m,
                    task_id=task_id,
                    dry_run=dry_run,
                    module_name=f"{_module_name}.{__n}",
                    locked_task_ids=locked_task_ids,
                )
        # If "weight" and "bias" exist and both are zero, then we presumably
        # warn the user that no parameter is forgotten.
        _fgt_num_params = 0
        try:
            _fgt_num_params += _forget_result.get_num_forgotten("weight")[0]
        except KeyError:
            pass
        try:
            _fgt_num_params += _forget_result.get_num_forgotten("bias")[0]
        except KeyError:
            pass
        if _fgt_num_params == 0:
            warnings.warn(
                NoParameterToForgetWarning(
                    f"No weights or biases are forgotten during the "
                    f"forgetting process of module `{_module_name}`."
                )
            )
        return _forget_result
