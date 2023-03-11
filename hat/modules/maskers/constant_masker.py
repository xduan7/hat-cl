from __future__ import annotations

from copy import deepcopy
from typing import Optional

import torch

from hat.types_ import Mask

from ._base import VectorMaskerABC


class ConstantMasker(VectorMaskerABC):
    """A constant (non-trainable) vector masker.

    This class is useful to save/load a tensor mask with a model.

    Args:
        mask: The mask tensor. If None, a random mask will be generated of
            shape `(num_features,)`.
        num_features: The number of features of the mask. If `mask` is
            specified, this argument will be ignored.
        mask_dim: The dimension of the data tensor to apply the mask to.
            Please refer to `VectorMaskerABC` for more details.

    """

    def __init__(
        self,
        mask: Optional[torch.Tensor],
        num_features: Optional[int],
        mask_dim: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(mask_dim)
        if mask is None:
            if num_features is None:
                raise ValueError(
                    "Either `num_features` or `mask` must be specified."
                )
            mask = torch.randn(num_features, device=device, dtype=dtype)
        elif mask.dim() == 1:
            mask = deepcopy(mask).to(device=device, dtype=dtype)
        elif mask.dim() != 1:
            raise ValueError(
                f"Vector mask has to be a 1D tensor. "
                f"Got mask of shape {mask.shape}."
            )
        self.register_buffer("mask", mask)

    def get_mask(self) -> Mask:
        """Get the mask tensor."""
        return self.mask

    def forward(
        self,
        data: torch.Tensor,
        mask_dim: Optional[int] = None,
    ) -> torch.Tensor:
        """Forward the data by applying the constant mask to it.

        Args:
            data: The data tensor.
            mask_dim: The dimension of the mask. If None, the dimension
                will be inferred by method `self.infer_mask_dim`. Defaults
                to `None`.

        Returns:
            The masked data tensor.

        """
        return self.apply_mask_to(data=data, mask_dim=mask_dim)
