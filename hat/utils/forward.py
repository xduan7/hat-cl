import warnings
from typing import TYPE_CHECKING, Any

import torch.nn as nn

# noinspection PyProtectedMember
from hat.modules._base import HATPayloadCarrierMixin

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
    use_masked_data: bool = True,
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
        the masked data or the unmasked data depending on the value of
        `use_masked_data`. Return a `HATPayload` with the returned data and
        everything else from the input payload.

    Args:
        module: The module to forward the payload through.
        hat_payload: The hard attention payload to forward through the module.
        use_masked_data: Whether to use the masked data in the payload. This
            argument is only used when the payload is task-specific and the
            module is not a `HATPayloadCarrierMixin`. When `True`, the masked
            data will be used. When `False`, the unmasked data will be used.
            Defaults to `True`.

    Returns:
        The payload returned by the module.

    """
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
            locked_task_ids=hat_payload.locked_task_ids,
            prev_maskers=hat_payload.prev_maskers,
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
        if use_masked_data:
            _data = module.forward(hat_payload.masked_data)
        else:
            _data = module.forward(hat_payload.unmasked_data)
        return HATPayload(
            data=_data,
            masker=hat_payload.masker,
            task_id=hat_payload.task_id,
            mask_scale=hat_payload.mask_scale,
            locked_task_ids=hat_payload.locked_task_ids,
            prev_maskers=hat_payload.prev_maskers,
            mask_applied=use_masked_data,
        )
