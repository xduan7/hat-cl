import functools
import unittest
import warnings
from copy import deepcopy
from typing import Iterable

import torch
import numpy as np
import random

import torch.nn as nn

from hat.exceptions import (
    NoParameterToForgetWarning,
    LearningSuppressedWarning,
    ModuleConversionWarning,
)
from hat.constants import DEF_HAT_MAX_TRN_MASK_SCALE
from hat.payload import HATPayload
from hat.utils import (
    convert_to_base_module,
    convert_to_task_dependent_module,
    forget_task,
)

DEBUG = False
RANDOM_SEED = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 2 if DEBUG else 256

NUM_TASKS = 5 if DEBUG else 10
MAX_TRN_MASK_SCALE = DEF_HAT_MAX_TRN_MASK_SCALE
TRN_MASK_SCALE = 1.0

DROPOUT_RATE = 0.2

# Small learning rate to preserve random distribution of the params.
LEARNING_RATE = 1e-6
MOMENTUM = 0.9
# Weight decay must be zero to prevent forgetting over L2 regularization.
WEIGHT_DECAY = 0
OPTIMIZERS = {
    "SGD": functools.partial(
        torch.optim.SGD,
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    ),
    "Adam": functools.partial(
        torch.optim.Adam,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    ),
    "AdamW": functools.partial(
        torch.optim.AdamW,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    ),
    "RMSprop": functools.partial(
        torch.optim.RMSprop,
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    ),
}


def set_random_seed(random_seed: int = RANDOM_SEED):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)


def supress_warnings():
    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=LearningSuppressedWarning)
    warnings.filterwarnings("ignore", category=ModuleConversionWarning)


def set_up():
    set_random_seed()
    supress_warnings()


def _deactivate_dropout(module: nn.Module):
    if not module.training:
        return
    if isinstance(module, nn.Dropout):
        module.eval()
    for __m in module.children():
        _deactivate_dropout(__m)


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


def check_to_base_conversion(
    test_case: unittest.TestCase,
    input_shape: Iterable[int],
    module: nn.Module,
    device: torch.device = DEVICE,
):
    """Check if a module can be converted to a base module and produce the
    same output as the original one.
    """
    _module = module.to(device)
    # Check if the base module can produce the same output as the original
    # one with different task IDs and training modes.
    for __task_id in [None] + list(range(NUM_TASKS)):
        __training_modes = [False, True]
        for __training in __training_modes:
            __module_ref = convert_to_base_module(
                module=deepcopy(_module),
                task_id=__task_id,
                trn_mask_scale=TRN_MASK_SCALE,
            )
            _module.train(__training)
            __module_ref.train(__training)
            # Dropout needs to be disabled if we want to compare the output.
            _deactivate_dropout(_module)
            _deactivate_dropout(__module_ref)
            _pld = HATPayload(
                data=torch.rand(BATCH_SIZE, *input_shape).to(device),
                task_id=__task_id,
                mask_scale=TRN_MASK_SCALE if __training else None,
            )
            _data = _pld.forward_by(_module).data
            _data_ref = __module_ref.forward(_pld.data)
            test_case.assertTrue(
                torch.allclose(_data, _data_ref),
                f"The converted base module does not produce the same output "
                f"as the task-dependent one with task ID {__task_id} and "
                f"training mode {__training}. \nTask-dependent module: "
                f"{_module} \nBase module: {__module_ref}",
            )


def check_from_base_conversion(
    test_case: unittest.TestCase,
    input_shape: Iterable[int],
    module: nn.Module,
    device: torch.device = DEVICE,
):
    """Check if a module can be converted to a task-dependent module and
    produce the same output as the original one.
    """
    _module = module.to(device)
    # Check if the task-dependent module can produce the same output as the
    # original one with different task IDs and training modes.
    # The task ID is set to None to disable the masker
    # This is because the during the conversion, the masker is added to
    # protect the parameters (e.g. `torch.nn.Linear` to
    # `hat.modules.HATLinear`) and the presence of the masker will cause
    # the output to be different.
    _task_id = None
    _module_ref = convert_to_task_dependent_module(
        module=deepcopy(_module),
        num_tasks=NUM_TASKS,
    )
    for __training in [True, False]:
        _module.train(__training)
        _module_ref.train(__training)
        # Dropout needs to be disabled if we want to compare the output.
        _deactivate_dropout(_module)
        _deactivate_dropout(_module_ref)
        _pld = HATPayload(
            data=torch.rand(BATCH_SIZE, *input_shape).to(device),
            task_id=_task_id,
            mask_scale=TRN_MASK_SCALE if __training else None,
        )
        _data = _module.forward(_pld.data)
        _data_ref = _pld.forward_by(_module_ref).data
        test_case.assertTrue(
            torch.allclose(_data, _data_ref),
            f"The converted task-dependent module does not produce the "
            f"same output as the base one with task ID {_task_id} and "
            f"training mode {__training}. \nBase module: {_module} "
            f"\nTask-dependent module: {_module_ref}",
        )


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
                except NoParameterToForgetWarning:
                    # warnings.warn(
                    #     f"Failed to forget task {__forget_task_id} "
                    #     f"after training on task {__task_id}. If this "
                    #     f"occurs frequently, consider increasing the "
                    #     f"number of tasks or the number of epochs.",
                    # )
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
