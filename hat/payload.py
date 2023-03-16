from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

import torch
import torch.nn as nn

from .utils import forward_hat_payload

if TYPE_CHECKING:
    from .modules.maskers.hat_masker import HATMasker
else:
    HATMasker = nn.Module


class HATPayload:
    """Payload for HAT modules.

    This class encapsulates the necessary information for HAT modules
    to perform their tasks. It is used as the input and output of any
    modules that inherit from `HATPayloadCarrierMixin`.

    Args:
        data: The (masked or unmasked) data to be processed.
        masker: The masker corresponding to the data. Even if the data
            is already masked, the masker is still needed in some cases.
        task_id: The ID of the task.
        mask_scale: The scale of the mask.
        locked_task_ids: The IDs of the tasks that are locked. If set to
            `None`, all the trained tasks except the current one will be
            locked. By locking a task, we mean that the parameters
            associated with the task will not be updated. Defaults to
            `None`.
        prev_maskers: The maskers used in the previous layers that are
            associated with the generation of data. This field will
            be automatically set by the `HATMasker` instances during the
            forward pass. Defaults to `None`.
        mask_applied: Whether the mask has been applied to the data. If
            `True`, the data will be considered as masked. Defaults to
            `False`.

    """

    def __init__(
        self,
        data: torch.Tensor,
        masker: Optional[HATMasker] = None,
        task_id: Optional[int] = None,
        mask_scale: Optional[float] = None,
        locked_task_ids: Optional[Sequence[int]] = None,
        prev_maskers: Optional[list[HATMasker]] = None,
        mask_applied: bool = False,
    ):
        self._masked_data = data if mask_applied else None
        self._unmasked_data = data if not mask_applied else None
        self.masker = masker
        self.task_id = task_id
        self.mask_scale = mask_scale
        self.locked_task_ids = locked_task_ids
        self.prev_maskers = prev_maskers
        self.mask_applied = mask_applied

    @property
    def data(self) -> torch.Tensor:
        """The data to be processed. Prefer masked data if available."""
        if self.masked_data is not None:
            return self.masked_data
        else:
            return self.unmasked_data

    @property
    def masked_data(self) -> Optional[torch.Tensor]:
        """The masked data if available."""
        if self._masked_data is None:
            if self.masker is None:
                return None
            self._masked_data = self.masker.apply_mask_to(
                self._unmasked_data,
                task_id=self.task_id,
                mask_scale=self.mask_scale,
            )
        return self._masked_data

    @property
    def unmasked_data(self) -> Optional[torch.Tensor]:
        """The unmasked data if available."""
        return self._unmasked_data

    def forward_by(
        self,
        module: nn.Module,
        use_masked_data: bool = True,
    ) -> HATPayload:
        """Forward the payload through the given module.

        See `hat.utils.forward_hat_payload` for more details.

        Args:
            module: The module to be used for forwarding.
            use_masked_data: Whether to use the masked data for forwarding.
                Defaults to `True`.

        Returns:
            The forwarded payload.

        """
        return forward_hat_payload(
            module=module,
            hat_payload=self,
            use_masked_data=use_masked_data,
        )

    # TODO: operations
    # def __add__(self, other: HATPayload) -> HATPayload:
    #     # The order matters
    #     # We should add the data of self with the masked data of other
    #     # The returned payload shall have the same mask as self, and the
    #     # prev_hat_masks should be merged with the prev_hat_masks of other
    #     pass

    def __repr__(self) -> str:
        if self.prev_maskers is not None:
            _prev_maskers = [repr(__m) for __m in self.prev_maskers]
        else:
            _prev_maskers = None
        return (
            f"HATPayload(\n"
            f"  data: {self.data}\n"
            f"  masker: {repr(self.masker)}\n"
            f"  task_id: {self.task_id}\n"
            f"  mask_scale: {self.mask_scale}\n"
            f"  prev_maskers: {_prev_maskers}\n"
            f"  locked_task_ids: {self.locked_task_ids}\n"
            f"  mask_applied: {self.mask_applied}\n"
            f")"
        )

    def __str__(self) -> str:
        return self.__repr__()
