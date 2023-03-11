import warnings
from typing import TYPE_CHECKING, Any

import torch.nn as nn

if TYPE_CHECKING:
    from hat.payload import HATPayload
else:
    HATPayload = Any


def _has_weight_param(module: nn.Module) -> bool:
    """Check if a module has trainable parameters with the word "weight" in
    their names.
    """
    for __n, __p in module.named_parameters():
        if "weight" in __n and __p.requires_grad:
            return True
    return False


def forward_hat_payload(
    module: nn.Module,
    hat_payload: HATPayload,
) -> HATPayload:
    """Forward the hard attention payload through the module.

    This function is meant to make `HATPayload` acceptable for ordinary
    `torch.nn.Module`.
    There are several cases depending on the module and the payload:
    1. The module is an instance of `HATPayloadCarrierMixin` and accepts
        `HATPayload` as input: forward the payload through the module.
    2. The module is a `nn.Sequential`: forward the payload through each
        submodule in the sequence.
    3. The module is not a `HATPayloadCarrierMixin` but the payload is not
        task-specific (task id is `None`): forward the payload data through
        the module and return a `HATPayload` with the returned data.
    4. The module is not a `HATPayloadCarrierMixin` and is not an instance of
        `nn.Sequential`, and the payload is task-specific: make sure that
        the module is not weighted, and perform the module forward pass on
        the unmasked data. Return a `HATPayload` with the returned data and
        everything else from the input payload.

    Args:
        module: The module to forward the payload through.
        hat_payload: The hard attention payload to forward through the module.

    Returns:
        The payload returned by the module.

    """
    # noinspection PyProtectedMember
    from hat.modules._base import HATPayloadCarrierMixin
    from hat.payload import HATPayload

    if isinstance(module, HATPayloadCarrierMixin):
        return module.forward(hat_payload)
    elif isinstance(module, nn.Sequential):
        for __m in module:
            hat_payload = forward_hat_payload(__m, hat_payload)
        return hat_payload
    elif hat_payload.task_id is None:
        return HATPayload(
            data=module.forward(hat_payload.data),
            masker=hat_payload.masker,
            task_id=hat_payload.task_id,
            mask_scale=hat_payload.mask_scale,
            prev_maskers=hat_payload.prev_maskers,
            locked_task_ids=hat_payload.locked_task_ids,
        )
    else:
        from hat.modules.utils import base_to_task_dependent_mapping as mapping

        if module.__class__ in mapping:
            warnings.warn(
                f"The module class {module.__class__} has a registered "
                f"task-dependent version {mapping[module.__class__]}. "
                f"Please use the task-dependent module instead if you "
                f"intend to use task-specific payloads."
            )
        if _has_weight_param(module):
            raise RuntimeError(
                f"The module {module.__class__.__name__} might contain "
                f"trainable parameters with 'weight' in their names. "
                f"Passing a `HATPayload` to such a module might cause "
                f"unexpected behavior."
            )
        _data = module.forward(hat_payload.data)
        return HATPayload(
            data=_data,
            masker=hat_payload.masker,
            task_id=hat_payload.task_id,
            mask_scale=hat_payload.mask_scale,
            prev_maskers=hat_payload.prev_maskers,
            locked_task_ids=hat_payload.locked_task_ids,
            mask_applied=True,
        )
