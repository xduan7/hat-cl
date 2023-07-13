from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Union

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
            forward pass. This attribute is NOT used for the masking
            process, and only serves as a reference for topology.
            Defaults to `None`.
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

    @property
    def original_data(self) -> torch.Tensor:
        """The original data (from the initialization)."""
        return self._masked_data if self.mask_applied else self._unmasked_data

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

    def to_dict(
        self,
        include_data: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """Convert the payload itself to a dictionary.

        Args:
            include_data: Whether to include the data in the dictionary.
                Defaults to `True`.
            **kwargs: The additional fields to be included in the
                dictionary.

        Returns:
            The dictionary representation of the payload.

        """
        _ret = {
            "masker": self.masker,
            "task_id": self.task_id,
            "mask_scale": self.mask_scale,
            "locked_task_ids": self.locked_task_ids,
            "prev_maskers": self.prev_maskers,
            "mask_applied": self.mask_applied,
        }
        _ret.update(kwargs)
        if include_data:
            _ret["data"] = self.original_data
        return _ret

    def apply_mask(self) -> HATPayload:
        """Apply the mask to the data and return a new payload.

        Returns:
            A new payload with masked data.

        """
        return HATPayload(
            data=self.masked_data,
            **self.to_dict(include_data=False, mask_applied=True),
        )

    def _merge_prev_masks(
        self,
        prev_maskers: Optional[list[HATMasker]],
    ) -> Optional[list[HATMasker]]:
        """Merge the previous maskers (from another payload) with the
        previous maskers of this payload.

        Args:
            prev_maskers: The previous maskers.

        Returns:
            The merged maskers.

        """
        if prev_maskers is None:
            return self.prev_maskers
        else:
            if self.prev_maskers:
                return prev_maskers + self.prev_maskers
            else:
                return prev_maskers

    def reshape(self, *args, **kwargs) -> HATPayload:
        """Reshape the data.

        Note that reshaping the data could make the masker non-applicable,
        so this method will automatically apply the mask before reshaping, if
        applicable.

        Args:
            *args: The arguments to be passed to `torch.Tensor.reshape`.
            **kwargs: The keyword arguments to be passed to
                `torch.Tensor.reshape`.

        Returns:
            The reshaped payload.

        """
        _pld = self.apply_mask() if self.masker is not None else self
        return HATPayload(
            data=_pld.original_data.reshape(*args, **kwargs),
            **_pld.to_dict(include_data=False),
        )

    def permute(self, *args, **kwargs) -> HATPayload:
        """Permute the data.

        Note that reshaping the data could make the masker non-applicable,
        so this method will automatically apply the mask before reshaping, if
        applicable.

        Args:
            *args: The arguments to be passed to `torch.Tensor.permute`.
            **kwargs: The keyword arguments to be passed to
                `torch.Tensor.permute`.

        Returns:
            The permuted payload.

        """
        return HATPayload(
            data=self.original_data.permute(*args, **kwargs),
            **self.to_dict(include_data=False),
        )

    def transpose(self, *args, **kwargs) -> HATPayload:
        """Transpose the data.

        Args:
            *args: The arguments to be passed to `torch.Tensor.transpose`.
            **kwargs: The keyword arguments to be passed to
                `torch.Tensor.transpose`.

        Returns:
            The transposed payload.

        """
        return HATPayload(
            data=self.original_data.transpose(*args, **kwargs),
            **self.to_dict(include_data=False),
        )

    def __op__(
        self,
        op_name: str,
        other: Union[HATPayload, torch.Tensor, float],
        use_other_masker: bool = False,
        reapply_mask: bool = False,
    ) -> HATPayload:
        """Apply an operation to the data.

        This is a helper method for implementing the operators.

        Args:
            op_name: The name of the operation.
            other: The other operand, which could be a `HATPayload` instance,
                a `torch.Tensor` instance, or a `float` instance.
            use_other_masker: Whether to use the masker of the other operand.
                If `True`, the other operand must be a `HATPayload`
                instance, and the masker of the other operand will be used
                for the new payload. Defaults to `False`.
            reapply_mask: Whether to reapply the mask to the new payload.
                Defaults to `False`.

        """
        _other_data = other.data if isinstance(other, HATPayload) else other
        _data = getattr(self.data, op_name)(_other_data)
        if use_other_masker:
            if not isinstance(other, HATPayload):
                raise ValueError(
                    "If `use_other_masker` of an operation is set to True, "
                    "the `other` argument must be an instance of `HATPayload`."
                )
            _masker = other.masker
        else:
            _masker = self.masker
        _prev_maskers = (
            self._merge_prev_masks(other.prev_maskers)
            if isinstance(other, HATPayload)
            else self.prev_maskers
        )
        _mask_applied = self.mask_applied and (not reapply_mask)
        return HATPayload(
            data=_data,
            **self.to_dict(
                include_data=False,
                masker=_masker,
                prev_maskers=_prev_maskers,
                mask_applied=_mask_applied,
            ),
        )

    def __add__(
        self,
        other: Union[HATPayload, torch.Tensor, float],
    ) -> HATPayload:
        """Add the data from other payload or tensor or number.

        Note that the order matters. The returned payload will inherit the
        attributes from `self` except for the following:
        - data: The data will be the element-wise sum between the original
            data of `self` and the preferably masked data of `other`.
        - prev_maskers: The `prev_maskers` will be the merged maskers from
            `self` and `other`.
        - mask_applied: The `mask_applied` will be set to `False` to enforce
            the mask to be applied again after the addition.

        """
        return self.__op__(
            op_name="__add__",
            other=other,
            reapply_mask=True,
        )

    def __mul__(
        self,
        other: Union[HATPayload, torch.Tensor, float],
    ) -> HATPayload:
        """Multiply the data from other payload or tensor or number.

        Note that the order matters. The returned payload will inherit the
        attributes from `self` except for the following:
        - data: The data will be the element-wise product between the original
            data of `self` and the preferably masked data of `other`.
        - prev_maskers: The `prev_maskers` will be the merged maskers from
            `self` and `other`.

        """
        return self.__op__(
            op_name="__mul__",
            other=other,
        )

    def __matmul__(
        self,
        other: Union[HATPayload, torch.Tensor],
    ) -> HATPayload:
        """Matrix multiply the data from other payload or tensor.

        Note that the order matters. The returned payload will inherit the
        attributes from `self` except for the following:
        - data: The data will be the matrix product between the original
            data of `self` and the preferably masked data of `other`.
        - masker: The `masker` will be the masker of `other` if available,
            otherwise the masker of `self`.
        - prev_maskers: The `prev_maskers` will be the merged maskers from
            `self` and `other`.

        """
        return self.__op__(
            op_name="__matmul__",
            other=other,
            use_other_masker=True,
        )

    def __repr__(self) -> str:
        """Return the string representation of the payload."""
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
        """Return the string representation of the payload."""
        return self.__repr__()
