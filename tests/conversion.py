import unittest
from copy import deepcopy
from typing import Iterable

import torch
import torch.nn as nn

from hat.payload import HATPayload
from hat.utils import convert_to_base_module, convert_to_task_dependent_module

from .constants import BATCH_SIZE, DEVICE, NUM_TASKS, TRN_MASK_SCALE
from .task import check_fully_task_dependent


def _deactivate_dropout(module: nn.Module):
    if not module.training:
        return
    if isinstance(module, nn.Dropout):
        module.eval()
    for __m in module.children():
        _deactivate_dropout(__m)


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
    ).to(device)
    check_fully_task_dependent(
        test_case=test_case,
        module=_module_ref,
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
