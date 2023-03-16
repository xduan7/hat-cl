from __future__ import annotations

from abc import ABC
from itertools import repeat
from typing import Any, Iterable, Optional, Union

import torch
import torch.nn as nn
from torch import classproperty

# noinspection PyProtectedMember
from torch.nn.modules.conv import _ConvNd

from hat.types_ import HATConfig

from ._base import HATMaskedModuleABC
from .utils import register_mapping


def _ntuple(n, name="parse"):
    """Helper function that converts a value to a tuple of size n."""

    def parse(x):
        if isinstance(x, Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    parse.__name__ = name
    return parse


_single = _ntuple(1, "_single")
_pair = _ntuple(2, "_pair")
_triple = _ntuple(3, "_triple")


class _HATConvNdABC(HATMaskedModuleABC, ABC):
    """Abstract class for HAT convolutional layers.

    It implements all the methods for the HAT convolutional layers except for
    the `__init__` and `base_class` property, which differs for different
    convolutional layers.

    """

    def to_base_module(
        self,
        task_id: Optional[int] = None,
        **kwargs: Any,
    ) -> nn.Sequential:
        """Convert the HAT convolutional layer to a sequential module
        consisting of the base convolutional layer and the base masker,
        which shall produce the same output as the HAT convolutional layer
        on the given task.

        Args:
            task_id: The ID of the task to be converted. It's irrelevant to
                the convolution layer and only required to convert the
                masker. See `HATMasker.to_base_module` for more details.
                Defaults to `None`.
            **kwargs: Additional keyword arguments for
                `HATMasker.to_base_module` method.

        Returns:
            The sequential module consisting of the base convolutional layer
            and the base masker.

        """
        _convnd: _ConvNd = self.base_class(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=self.bias is not None,
            padding_mode=self.padding_mode,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        _convnd.weight.data.copy_(self.weight.data)
        if self.bias is not None:
            _convnd.bias.data.copy_(self.bias.data)
        _masker = self.masker.to_base_module(
            task_id=task_id,
            **kwargs,
        )
        return nn.Sequential(_convnd, _masker)

    @classmethod
    def from_base_module(
        cls: Union[type[HATConv1d], type[HATConv2d], type[HATConv3d]],
        base_module: _ConvNd,
        **kwargs: Any,
    ) -> _HATConvNdABC:
        """Create a HAT convolutional layer from a PyTorch convolutional
        layer by copying the weights and the bias.

        Args:
            base_module: The base convolutional layer.
            **kwargs: Additional keyword arguments for the `HATConfig`
                constructor.

        Returns:
            The created HAT convolutional layer.

        """
        _hat_config = HATConfig(**kwargs)
        _hat_convnd = cls(
            in_channels=base_module.in_channels,
            out_channels=base_module.out_channels,
            kernel_size=base_module.kernel_size,
            stride=base_module.stride,
            padding=base_module.padding,
            dilation=base_module.dilation,
            groups=base_module.groups,
            bias=base_module.bias is not None,
            padding_mode=base_module.padding_mode,
            hat_config=_hat_config,
            device=base_module.weight.device,
            dtype=base_module.weight.dtype,
        )
        _hat_convnd.load_from_base_module(base_module=base_module)
        return _hat_convnd


@register_mapping
class HATConv1d(_HATConvNdABC, nn.Conv1d):
    """HAT convolutional layer for 1D inputs.

    This class is a wrapper for `torch.nn.Conv1d` that adds the HAT masker
    to the convolutional layer, with HAT mechanism implemented in the
    `forward` method.

    Args:
        in_channels: see `torch.nn.Conv1d`.
        out_channels: see `torch.nn.Conv1d`.
        kernel_size: see `torch.nn.Conv1d`.
        hat_config: The configuration for `HATMasker`
        stride: see `torch.nn.Conv1d`.
        padding: see `torch.nn.Conv1d`.
        dilation: see `torch.nn.Conv1d`.
        groups: see `torch.nn.Conv1d`.
        bias: see `torch.nn.Conv1d`.
        padding_mode: see `torch.nn.Conv1d`.
        device: see `torch.nn.Conv1d`.
        dtype: see `torch.nn.Conv1d`.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple[int, ...]],
        hat_config: HATConfig,
        stride: Union[int, tuple[int, ...]] = 1,
        padding: Union[str, int, tuple[int, ...]] = 0,
        dilation: Union[int, tuple[int, ...]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        kernel_size_ = _single(kernel_size)
        stride_ = _single(stride)
        padding_ = padding if isinstance(padding, str) else _single(padding)
        dilation_ = _single(dilation)
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_features=out_channels,
            hat_config=hat_config,
            kernel_size=kernel_size_,
            stride=stride_,
            padding=padding_,
            dilation=dilation_,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

    @classproperty
    def base_class(self) -> type[nn.Module]:
        """Base class of the HAT convolutional 1D layer."""
        return nn.Conv1d  # type: ignore


@register_mapping
class HATConv2d(_HATConvNdABC, nn.Conv2d):
    """HAT convolutional layer for 2D inputs.

    This class is a wrapper for `torch.nn.Conv2d` that adds the HAT masker
    to the convolutional layer, with HAT mechanism implemented in the
    `forward` method.

    Args:
        in_channels: see `torch.nn.Conv2d`.
        out_channels: see `torch.nn.Conv2d`.
        kernel_size: see `torch.nn.Conv2d`.
        hat_config: The configuration for `HATMasker`
        stride: see `torch.nn.Conv2d`.
        padding: see `torch.nn.Conv2d`.
        dilation: see `torch.nn.Conv2d`.
        groups: see `torch.nn.Conv2d`.
        bias: see `torch.nn.Conv2d`.
        padding_mode: see `torch.nn.Conv2d`.
        device: see `torch.nn.Conv2d`.
        dtype: see `torch.nn.Conv2d`.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple[int, ...]],
        hat_config: HATConfig,
        stride: Union[int, tuple[int, ...]] = 1,
        padding: Union[str, int, tuple[int, ...]] = 0,
        dilation: Union[int, tuple[int, ...]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_features=out_channels,
            hat_config=hat_config,
            kernel_size=kernel_size_,
            stride=stride_,
            padding=padding_,
            dilation=dilation_,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

    @classproperty
    def base_class(self) -> type[nn.Module]:
        """Base class of the HAT convolutional 2D layer."""
        return nn.Conv2d  # type: ignore


@register_mapping
class HATConv3d(_HATConvNdABC, nn.Conv3d):
    """HAT convolutional layer for 3D inputs.

    This class is a wrapper for `torch.nn.Conv3d` that adds the HAT masker
    to the convolutional layer, with HAT mechanism implemented in the
    `forward` method.

    Args:
        in_channels: see `torch.nn.Conv3d`.
        out_channels: see `torch.nn.Conv3d`.
        kernel_size: see `torch.nn.Conv3d`.
        hat_config: The configuration for `HATMasker`
        stride: see `torch.nn.Conv3d`.
        padding: see `torch.nn.Conv3d`.
        dilation: see `torch.nn.Conv3d`.
        groups: see `torch.nn.Conv3d`.
        bias: see `torch.nn.Conv3d`.
        padding_mode: see `torch.nn.Conv3d`.
        device: see `torch.nn.Conv3d`.
        dtype: see `torch.nn.Conv3d`.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple[int, ...]],
        hat_config: HATConfig,
        stride: Union[int, tuple[int, ...]] = 1,
        padding: Union[str, int, tuple[int, ...]] = 0,
        dilation: Union[int, tuple[int, ...]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        kernel_size_ = _triple(kernel_size)
        stride_ = _triple(stride)
        padding_ = padding if isinstance(padding, str) else _triple(padding)
        dilation_ = _triple(dilation)
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_features=out_channels,
            hat_config=hat_config,
            kernel_size=kernel_size_,
            stride=stride_,
            padding=padding_,
            dilation=dilation_,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

    @classproperty
    def base_class(self) -> type[nn.Module]:
        """Base class of the HAT convolutional 3D layer."""
        return nn.Conv3d  # type: ignore
