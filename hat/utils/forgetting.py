import warnings

import torch.nn as nn

from hat.exceptions import NoParameterToForgetWarning
from hat.types_ import ForgetResult


def forget_task(
    module: nn.Module,
    task_id: int,
    dry_run: bool = False,
) -> ForgetResult:
    """Forget the given tasks of a given module by calling the `forget`
    function of all `TaskDependentModuleABC` children of the module.

    Args:
        module: The module to forget the task.
        task_id: The ID of the task to be forgotten.
        dry_run: If `True`, the forgetting process will be simulated
            without actually changing the module. Defaults to `False`.

    Returns:
        The forgetting result. See `hat.types_.ForgetResult` for more
        details.

    """
    # noinspection PyProtectedMember
    from hat.modules._base import TaskDependentModuleABC

    if isinstance(module, TaskDependentModuleABC):
        return module.forget(task_id, dry_run)
    else:
        # When some of the children modules can forget while others cannot,
        # we still consider the task forgotten because the output will
        # change ultimately.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=NoParameterToForgetWarning
            )
            _forget_result = ForgetResult()
            for __m in module.children():
                _forget_result += forget_task(__m, task_id, dry_run)
        if _forget_result.num_forgotten_params == 0:
            warnings.warn(
                f"No weighted parameters are forgotten for task {task_id}.",
                NoParameterToForgetWarning,
            )
        return _forget_result
