import unittest
import warnings
from copy import deepcopy
from typing import Iterable

import torch
import torch.nn as nn

from hat.exceptions import NoParameterToForgetWarning

# noinspection PyProtectedMember
from hat.modules._base import TaskDependentModuleABC
from hat.payload import HATPayload
from hat.utils import forget_task, prune_hat_module

from .constants import (
    BATCH_SIZE,
    DEVICE,
    NUM_TASKS,
    OPTIMIZERS,
    TRN_MASK_SCALE,
)


def _compare_modules(
    module: nn.Module,
    module_ref: nn.Module,
) -> bool:
    """Check if two modules have the same parameters."""
    for __k, __v in module.named_parameters():
        __v_ref = module_ref.state_dict()[__k]
        if not torch.allclose(__v, __v_ref):
            return False
    return True


def check_remembering(
    test_case: unittest.TestCase,
    input_shape: Iterable[int],
    module: nn.Module,
    device: torch.device = DEVICE,
):
    """Check if the module remembers the output of previous tasks
    after training on the current one.
    """
    _module = module.to(device)
    for __optimizer_name, __optimizer in OPTIMIZERS.items():
        for __task_id in range(NUM_TASKS):
            _module.train()
            _optim = __optimizer(_module.parameters())
            _optim.zero_grad()
            __module_ref = deepcopy(_module)
            __pld = HATPayload(
                data=torch.rand(BATCH_SIZE, *input_shape).to(device),
                task_id=__task_id,
                mask_scale=TRN_MASK_SCALE,
            )
            __data = __pld.forward_by(_module).data
            __data.sum().backward()
            _optim.step()
            for __prev_task_id in range(__task_id):
                _module.eval()
                __module_ref.eval()
                __pld = HATPayload(
                    data=torch.rand(BATCH_SIZE, *input_shape).to(device),
                    task_id=__prev_task_id,
                    mask_scale=None,
                )
                __data = __pld.forward_by(_module).data
                __data_ref = __pld.forward_by(__module_ref).data
                test_case.assertTrue(
                    torch.allclose(__data, __data_ref),
                    f"The output of previous task {__prev_task_id} "
                    f"changed after training on task {__task_id} "
                    f"with optimizer {__optimizer_name}.",
                )


def check_locking(
    test_case: unittest.TestCase,
    input_shape: Iterable[int],
    module: nn.Module,
    device: torch.device = DEVICE,
):
    """Check if the (future tasks of) module can be locked by setting up the
    `locked_task_ids` in the payload, by comparing the output of the module
    with future tasks locked to the output of the reference module.
    """
    _module = deepcopy(module).to(device)
    for __optimizer_name, __optimizer in OPTIMIZERS.items():
        for __task_id in range(NUM_TASKS - 1):
            _module.train()
            _optim = __optimizer(_module.parameters())
            _optim.zero_grad()
            __module_ref = deepcopy(_module)
            # Lock all future tasks
            __locked_task_ids = list(range(__task_id + 1, NUM_TASKS))
            __pld = HATPayload(
                data=torch.rand(BATCH_SIZE, *input_shape).to(device),
                task_id=__task_id,
                mask_scale=1,
                locked_task_ids=__locked_task_ids,
            )
            __pld = __pld.forward_by(_module)
            __data = __pld.data
            __data.sum().backward()
            _optim.step()
            for __locked_task_id in __locked_task_ids:
                _module.eval()
                __module_ref.eval()
                __pld = HATPayload(
                    data=torch.rand(BATCH_SIZE, *input_shape).to(device),
                    task_id=__locked_task_id,
                    mask_scale=None,
                )
                __data = __pld.forward_by(_module).data
                __data_ref = __pld.forward_by(__module_ref).data
                test_case.assertTrue(
                    torch.allclose(__data, __data_ref),
                    f"The output of future task {__locked_task_id} "
                    f"changed after training on task {__task_id} "
                    f"with optimizer {__optimizer_name} even though "
                    f"the task is locked.",
                )


def check_forgetting(
    test_case: unittest.TestCase,
    input_shape: Iterable[int],
    module: nn.Module,
    device: torch.device = DEVICE,
):
    """Check if the module can forget about certain tasks without interfering
    with the other trained tasks.
    """
    # Make sure that the attention mask is not too small (nothing to forget)
    # or too large (tasks locked to each other).

    _module = module.to(device)
    _failed_forgetting_count, _total_forgetting_count = 0, 0
    for __task_id in range(NUM_TASKS):
        _module.train()
        _optim = OPTIMIZERS["SGD"](_module.parameters())
        _optim.zero_grad()
        __pld = HATPayload(
            data=torch.rand(BATCH_SIZE, *input_shape).to(device),
            task_id=__task_id,
            mask_scale=TRN_MASK_SCALE,
        )
        __data = __pld.forward_by(_module).data
        __data.sum().backward()
        _optim.step()
        for __forget_task_id in range(__task_id):
            __module = deepcopy(_module)
            __module_ref = deepcopy(_module)
            __module.eval()
            __module_ref.eval()
            _total_forgetting_count += 1
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "error", category=NoParameterToForgetWarning
                )
                try:
                    __forget_result = forget_task(__module, __forget_task_id)
                except NoParameterToForgetWarning as __e:
                    print(
                        f"Failed to forget task {__forget_task_id} "
                        f"after training on task {__task_id}. This happens "
                        f"when the forgetting task is locked behind other "
                        f"tasks."
                    )
                    _failed_forgetting_count += 1
                    continue
            # Check if all other tasks are still remembered
            __remember_task_ids = list(range(__task_id))
            __remember_task_ids.remove(__forget_task_id)
            for __remember_task_id in __remember_task_ids:
                __pld = HATPayload(
                    data=torch.rand(BATCH_SIZE, *input_shape).to(device),
                    task_id=__remember_task_id,
                    mask_scale=None,
                )
                __data = __pld.forward_by(__module).data
                __data_ref = __pld.forward_by(__module_ref).data
                test_case.assertTrue(
                    torch.allclose(__data, __data_ref),
                    f"The output of task {__remember_task_id} that should "
                    f"be remembered changed after forgetting task "
                    f"{__forget_task_id} and training on task {__task_id}.",
                )
            # Check if the forgotten task is forgotten
            # Note that the parameter changes does not equate to output
            # changes (e.g. relu zeroing out negative values). So we only
            # checks parameter changes here.
            test_case.assertFalse(
                _compare_modules(__module, __module_ref),
                f"No parameter changed after forgetting task "
                f"{__forget_task_id} and training on task {__task_id}.",
            )

    # Forgetting a single task with random masks often fails because the
    # tasks share too much mask activations, which is especially so if we
    # have more tasks or have a wider network. However, this is not a
    # problem in real applications because there won't be as much mask
    # sharing with the right regularization, and a deeper network can always
    # find a layer that might forget a task.
    test_case.assertTrue(
        (_failed_forgetting_count / _total_forgetting_count) < 1,
        f"Failed to forget all {NUM_TASKS} tasks.",
    )


def check_pruning(
    test_case: unittest.TestCase,
    input_shape: Iterable[int],
    module: nn.Module,
    device: torch.device = DEVICE,
):
    """Check if the pruned network can produce the same output as the
    original network.
    """
    _module_ref = module.to(device)
    _module_ref.train()
    _optim = OPTIMIZERS["SGD"](_module_ref.parameters())
    for __task_id in range(NUM_TASKS):
        _optim.zero_grad()
        __pld = HATPayload(
            data=torch.rand(BATCH_SIZE, *input_shape).to(device),
            task_id=__task_id,
            mask_scale=TRN_MASK_SCALE,
        )
        __data = __pld.forward_by(_module_ref).data
        __data.sum().backward()
        _optim.step()
        _optim.zero_grad()
    _module_ref.eval()
    for __task_id in range(NUM_TASKS):
        __module = deepcopy(_module_ref)
        __module, _forget_result = prune_hat_module(__module, __task_id)
        __pld = HATPayload(
            data=torch.rand(BATCH_SIZE, *input_shape).to(device),
            task_id=__task_id,
            mask_scale=None,
        )
        __data = __pld.forward_by(__module).data
        __data_ref = __pld.forward_by(_module_ref).data
        test_case.assertTrue(
            torch.allclose(__data, __data_ref),
            f"The output of task {__task_id} changed after pruning.",
        )


def check_fully_task_dependent(
    test_case: unittest.TestCase,
    module: nn.Module,
):
    """Check if a module is fully task-dependent."""
    for __n, __m in module.named_children():
        if isinstance(__m, TaskDependentModuleABC):
            return True
        else:
            test_case.assertTrue(
                check_fully_task_dependent(test_case, __m),
                f"Module {__n} is not fully task-dependent.",
            )


def check_fully_torch(
    test_case: unittest.TestCase,
    module: nn.Module,
):
    """Check if a module is fully torch."""
    for __n, __m in module.named_children():
        if isinstance(__m, TaskDependentModuleABC):
            test_case.fail(f"Module {__n} is task-dependent.")
        else:
            check_fully_torch(test_case, __m)
