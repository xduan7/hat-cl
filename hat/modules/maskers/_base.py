from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Any, Optional

import torch
import torch.nn as nn

from hat.exceptions import (
    MaskDimensionInferenceWarning,
    MaskDimensionMismatchError,
)
from hat.types_ import Mask


class VectorMaskerABC(nn.Module, ABC):
    """Abstract class for vector maskers.

    A vector masker is a module that can mask the a data tensor with a
    1D mask tensor. The masking process is done by element-wise
    multiplication of the data tensor and the expanded mask tensor.

    Args:
        mask_dim: The dimension of the data tensor to apply the mask to.
            For example, if the data tensor is of shape `(batch_size,
            num_features)`, and `mask_dim=1`, the mask tensor must be of
            shape `(num_features,)` and will be expanded to shape
            `(batch_size, num_features)` before applying to the data tensor.
            If `None`, the mask dimension will be inferred from the data
            tensor. Please refer to `infer_mask_dim` for more details.

    """

    def __init__(self, mask_dim: Optional[int], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_dim = mask_dim

    @abstractmethod
    def get_mask(self, *args, **kwargs) -> Mask:
        """Get the mask tensor."""
        raise NotImplementedError

    def apply_mask_to(
        self,
        data: torch.Tensor,
        mask_dim: Optional[int] = None,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Mask the data tensor with the mask tensor.

        Args:
            data: The data tensor to be masked.
            mask_dim: The dimension of the data tensor to apply the mask to
                that overrides `self.mask_dim`.

        Returns:
            The masked data tensor.

        """
        return data * self.reshape_mask(
            data=data,
            mask=self.get_mask(*args, **kwargs),
            mask_dim=mask_dim or self.mask_dim,
            expand_as_data=True,
        )

    @staticmethod
    def infer_mask_dim(
        data: torch.Tensor,
        mask: torch.Tensor,
    ) -> int:
        """Infer which dimension of the data tensor the mask should be
        applied to.

        It should be the first dimension of the data tensor that matches the
        dimension of the mask, that is not the batch dimension.

        Args:
            data: The data tensor to be masked.
            mask: The mask tensor to be applied.

        Returns:
            The dimension of the mask.

        """
        _mask_dim = None
        for __dim, __size in enumerate(data.shape[1:]):
            if __size == mask.shape[0]:
                if _mask_dim is None:
                    _mask_dim = __dim + 1
                else:
                    warnings.warn(
                        f"Vector mask of shape {mask.shape} could be "
                        f"applied to multiple dimensions of data of "
                        f"shape {data.shape}. Please specify the "
                        f"dimension to apply the mask to explicitly.",
                        MaskDimensionInferenceWarning,
                    )
        if _mask_dim is None:
            raise MaskDimensionMismatchError(
                f"Vector mask of shape {mask.shape} is not "
                f"applicable to data of shape {data.shape}."
            )
        return _mask_dim

    @staticmethod
    def reshape_mask(
        data: torch.Tensor,
        mask: torch.Tensor,
        mask_dim: Optional[int] = None,
        expand_as_data: bool = True,
    ) -> torch.Tensor:
        """Reshape and expand the mask tensor to match the data tensor.

        Args:
            data: The data tensor to be masked.
            mask: The mask tensor to be applied.
            mask_dim: The dimension of the data tensor to apply the mask to.
                If None, the dimension will be inferred from the data tensor
                and the mask tensor. Please refer to `self.infer_mask_dim`
                for more details.
            expand_as_data: Whether to expand the mask tensor to match the
                shape of the data tensor.

        Returns:
            The reshaped and expanded mask tensor.

        """
        _mask_dim = (
            VectorMaskerABC.infer_mask_dim(data, mask)
            if mask_dim is None
            else mask_dim
        )
        if _mask_dim < 0:
            _mask_dim += data.dim()
        mask = mask.reshape(
            *([1] * _mask_dim),
            mask.size(0),
            *([1] * (data.dim() - _mask_dim - 1)),
        )
        if expand_as_data:
            mask = mask.expand_as(data)
        return mask

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        """Forward the data (usually by applying mask to it)."""
        raise NotImplementedError
