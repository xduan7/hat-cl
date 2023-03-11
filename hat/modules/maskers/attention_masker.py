from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn

from hat.types_ import Mask

from ._base import VectorMaskerABC


class AttentionMasker(VectorMaskerABC):
    """A trainable vector masker that mimics the attention mechanism in hard
    attention to the task.

    This class serves as a base class for the `HATMasker` module. It can
    also function as a snapshot of the `HATMasker` module for a specific
    task, which could be useful for testing and shipping purposes.

    Args:
        num_features: The number of features in the data tensor.
        trn_mask_scale: The scale of the mask in training mode. If None,
            the mask must be given explicitly in the `get_mask` method during
            training. Please refer to `self.infer_mask_scale` for more
            details.
        gate: The gating function to apply to the scaled attention vector. If
            None, `torch.sigmoid` function will be used.

    """

    def __init__(
        self,
        num_features: Optional[int] = None,
        trn_mask_scale: Optional[float] = None,
        gate: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        mask_dim: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(mask_dim)
        self.attention = nn.Parameter(
            torch.randn(num_features, device=device, dtype=dtype)
        )
        if trn_mask_scale is not None:
            self.register_buffer(
                "trn_mask_scale",
                torch.as_tensor(trn_mask_scale),
            )
        else:
            self.trn_mask_scale = None
        self.gate = gate or torch.sigmoid

    def infer_mask_scale(self) -> Optional[float]:
        """Infer the mask scale to use in the `get_mask` method.

        It returns the `trn_mask_scale` attribute if the module is in
        training mode. Otherwise, it returns `None` so that the masks will
        be binary during inference.

        """
        return self.trn_mask_scale if self.training else None

    def get_mask(self, mask_scale: Optional[float] = None) -> Mask:
        """Get the mask tensor.

        Args:
            mask_scale: The scale of the mask. If None, the mask scale will
                be inferred from the `self.infer_mask_scale` method. Note
                that training requires a valid mask scale to be given,
                either by setting the `trn_mask_scale` attribute or passing
                the `mask_scale` as an argument. Defaults to `None`.

        Returns
            The mask tensor.

        """
        mask_scale = mask_scale or self.infer_mask_scale()
        if mask_scale is None:
            if self.training:
                raise ValueError(
                    "The mask scale of `AttentionMasker` is not set. "
                    "Please set the `trn_mask_scale` attribute or "
                    "pass the `mask_scale` as an argument."
                )
            return torch.BoolTensor(self.attention > 0)
        else:
            return self.gate(self.attention * mask_scale)

    def forward(
        self,
        data: torch.Tensor,
        mask_dim: Optional[int] = None,
        mask_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """Forward the data by applying the mask to it. The mask is
        generated by the `self.get_mask` method.

        Args:
            data: The data tensor.
            mask_dim: The dimension of the mask. If None, the dimension
                will be inferred by method `self.infer_mask_dim`. Defaults
                to `None`.
            mask_scale: The scale of the mask. See `self.get_mask` for
                more details. Defaults to `None`.

        Returns:
            The masked data tensor.

        """
        return self.apply_mask_to(
            data=data,
            mask_dim=mask_dim,
            mask_scale=mask_scale,
        )

    def __str__(self) -> str:
        return (
            f"AttentionMasker("
            f"num_features={self.attention.shape[0]}, "
            f"trn_mask_scale={self.trn_mask_scale}, "
            f"gate={self.gate.__name__})"
        )

    def __repr__(self) -> str:
        return self.__str__()
