import unittest
from copy import deepcopy
from typing import Iterable

import torch
import torch.nn as nn

from hat.modules import HATMasker
from hat.payload import HATPayload

from .constants import BATCH_SIZE, DEVICE, MAX_TRN_MASK_SCALE, NUM_TASKS
from .utils import deactivate_dropout, deactivate_tracking_stats


def _magnify_hat_masker_attention(module: nn.Module, magnitude: float = 0.5):
    """Magnify the attention weights of the module."""
    if isinstance(module, HATMasker):
        for __attn in module.attention:
            __attn.data += magnitude * torch.sign(__attn)
    for __child in module.children():
        _magnify_hat_masker_attention(__child)


def check_trn_evl_consistency(
    test_case: unittest.TestCase,
    input_shape: Iterable[int],
    module: nn.Module,
    device: torch.device = DEVICE,
):
    """Check if the module produces very similar outputs during training and
    evaluation, when the training mask scale is set to maximum.
    """
    _module = module.to(device)
    _magnify_hat_masker_attention(_module)
    _module.train()
    deactivate_dropout(_module)
    deactivate_tracking_stats(_module)
    _module_ref = deepcopy(_module)
    _module_ref.eval()
    for __task_id in [None] + list(range(NUM_TASKS)):
        __data = torch.rand(BATCH_SIZE, *input_shape).to(device)
        __trn_pld = HATPayload(
            data=__data,
            task_id=__task_id,
            mask_scale=MAX_TRN_MASK_SCALE,
        )
        __evl_pld = HATPayload(
            data=__data,
            task_id=__task_id,
        )
        __data = __trn_pld.forward_by(_module).data
        __data_ref = __evl_pld.forward_by(_module_ref).data
        test_case.assertTrue(
            torch.allclose(__data_ref, __data),
            f"The module produces different outputs during training and "
            f"evaluation with task ID {__task_id} and training mask scale "
            f"{MAX_TRN_MASK_SCALE}. \nModule: {_module}",
        )


def check_evl_mask_scale_consistency(
    test_case: unittest.TestCase,
    input_shape: Iterable[int],
    module: nn.Module,
    device: torch.device = DEVICE,
):
    """Check if the module produces very similar outputs during evaluation
    with mask scale is set to maximum training mask scale and `None`.
    """
    _module = module.to(device)
    _magnify_hat_masker_attention(_module)
    _module.eval()
    deactivate_dropout(_module)
    deactivate_tracking_stats(_module)
    for __task_id in [None] + list(range(NUM_TASKS)):
        __data = torch.rand(BATCH_SIZE, *input_shape).to(device)
        __pld = HATPayload(
            data=__data,
            task_id=__task_id,
            mask_scale=MAX_TRN_MASK_SCALE,
        )
        __ref_pld = HATPayload(
            data=__data,
            task_id=__task_id,
        )
        __data = __pld.forward_by(_module).data
        __data_ref = __ref_pld.forward_by(_module).data
        test_case.assertTrue(
            torch.allclose(__data_ref, __data),
            f"The module produces different outputs during evaluation with "
            f"task ID {__task_id} and mask scale set to {MAX_TRN_MASK_SCALE} "
            f"and `None`. \nModule: {_module}",
        )
