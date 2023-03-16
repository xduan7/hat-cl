from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
from torch import classproperty

from hat.types_ import HATConfig

from ._base import HATMaskedModuleABC
from .utils import register_mapping


@register_mapping
class HATLinear(
    HATMaskedModuleABC,
    nn.Linear,
):
    """HAT linear layer.

    This class is a wrapper for `torch.nn.Linear` that adds the HAT masker
    to the linear layer, with HAT mechanism implemented in the
    `forward` method.

    Args:
        in_features: see `torch.nn.Linear`.
        out_features: see `torch.nn.Linear`.
        hat_config: The configuration for `HATMasker`
        bias: see `torch.nn.Linear`.
        device: see `torch.nn.Linear`.
        dtype: see `torch.nn.Linear`.

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hat_config: HATConfig,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            num_features=out_features,
            hat_config=hat_config,
            bias=bias,
            device=device,
            dtype=dtype,
        )

    @classproperty
    def base_class(self) -> type[nn.Module]:
        """Base class of the HAT linear layer."""
        return nn.Linear  # type: ignore

    def to_base_module(
        self,
        task_id: Optional[int] = None,
        **kwargs: Any,
    ) -> nn.Sequential:
        """Convert the HAT linear layer to a sequential module consisting of
        the base linear layer and the base masker, which shall produce the
        same output as the HAT linear layer on the given task.

        Args:
            task_id: The ID of the task to be converted. It's irrelevant to
                the linear layer and only required to convert the masker.
                See `HATMasker.to_base_module` for more details. Defaults
                to `None`.
            **kwargs: Additional keyword arguments for
                `HATMasker.to_base_module` method.

        Returns:
            A sequential module consisting of the base linear layer and
            the base masker.

        """
        _linear = self.base_class(
            in_features=self.in_features,
            out_features=self.out_features,
            bias=self.bias is not None,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        _linear.weight.data.copy_(self.weight.data)
        _linear.bias.data.copy_(self.bias.data)
        _masker = self.masker.to_base_module(
            task_id=task_id,
            **kwargs,
        )
        return nn.Sequential(_linear, _masker)

    @classmethod
    def from_base_module(
        cls: type[HATLinear],
        base_module: nn.Linear,
        **kwargs: Any,
    ) -> HATLinear:
        """Create a HAT linear layer from a PyTorch linear layer by copying
        the weights and the bias.

        Args:
            base_module: The base linear layer.
            **kwargs: Additional keyword arguments for the `HATConfig`
                constructor.

        Returns:
            A HAT linear layer with the same weights and bias as the base
            linear layer.

        """
        _hat_config = HATConfig(**kwargs)
        _hat_linear = cls(
            in_features=base_module.weight.shape[1],
            out_features=base_module.weight.shape[0],
            hat_config=_hat_config,
            bias=base_module.bias is not None,
            device=base_module.weight.device,
            dtype=base_module.weight.dtype,
        )
        _hat_linear.load_from_base_module(base_module=base_module)
        return _hat_linear
