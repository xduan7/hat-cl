from __future__ import annotations

import warnings
from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Callable, Optional, Union

import torch

from .constants import (
    DEF_HAT_ATTN_CLAMP,
    DEF_HAT_GRAD_COMP_CLAMP,
    DEF_HAT_GRAD_COMP_FACTOR,
    DEF_HAT_INIT_STRAT,
    DEF_HAT_MAX_TRN_MASK_SCALE,
)

Mask = Union[torch.FloatTensor, torch.BoolTensor]


class HATConfig(Mapping):
    """Configuration for HAT (Hard attention to the task) modules.

    Args:
        num_tasks: Number of tasks.
        mask_dim: The dimension of the data tensor to apply the mask to.
            See `VectorMaskerABC` for more details. Defaults to `None`.
        max_trn_mask_scale: The maximum scale of the trainable mask.
            Necessary for gradient compensation. Defaults to
            `hat.constants.DEF_HAT_MAX_TRN_MASK_SCALE`.
        init_strat: The initialization strategy for the trainable mask.
            Defaults to `hat.constants.DEF_HAT_INIT_STRAT`.
        grad_comp_clamp: The maximum value of the gradient during gradient
            compensation. Defaults to `hat.constants.DEF_HAT_GRAD_COMP_CLAMP`.
        grad_comp_factor: The factor to multiply the gradient with during
            gradient compensation. Defaults to
            `hat.constants.DEF_HAT_GRAD_COMP_FACTOR`.
        gate: The gating function to apply to the scaled attention vector.
            Defaults to `torch.sigmoid`.
        **kwargs: For polymorphism reasons. Should be empty.

    """

    def __init__(
        self,
        num_tasks: int,
        mask_dim: Optional[int] = None,
        init_strat: str = DEF_HAT_INIT_STRAT,
        max_trn_mask_scale: float = DEF_HAT_MAX_TRN_MASK_SCALE,
        # TODO: make the clamps optional
        attn_clamp: float = DEF_HAT_ATTN_CLAMP,
        grad_comp_clamp: float = DEF_HAT_GRAD_COMP_CLAMP,
        grad_comp_factor: float = DEF_HAT_GRAD_COMP_FACTOR,
        gate: Callable[[torch.Tensor], torch.Tensor] = torch.sigmoid,
        **kwargs: Any,  # For polymorphism reasons.
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.mask_dim = mask_dim
        self.max_trn_mask_scale = max_trn_mask_scale
        self.init_strat = init_strat
        self.attn_clamp = attn_clamp
        self.grad_comp_clamp = grad_comp_clamp
        self.grad_comp_factor = grad_comp_factor
        self.gate = gate
        if kwargs != {}:
            warnings.warn(f"Unrecognized kwargs: {kwargs}.")

    def __len__(self) -> int:
        """Return the length of the config. Useful for unpacking."""
        return len(self.__dict__)

    def __iter__(self):
        """Return the iterator of the config. Useful for unpacking."""
        return iter(self.__dict__)

    def __getitem__(self, item):
        """Return the value of the config. Useful for unpacking."""
        return self.__dict__[item]

    def __repr__(self) -> str:
        return (
            f"HATConfig(num_tasks={self.num_tasks}, "
            f"mask_dim={self.mask_dim}, "
            f"max_trn_mask_scale={self.max_trn_mask_scale}, "
            f"grad_comp_clamp={self.grad_comp_clamp}, "
            f"gate={self.gate})"
        )

    def __str__(self) -> str:
        return self.__repr__()


class ForgetResult(dict[str, torch.BoolTensor]):
    """Result of forgetting a task of a task dependent module.

    This class is essentially a dictionary that maps from the name of a
    parameter (e.g., 'l0.weight' and 'layer[2].conv1.bias'.) to
    `torch.BoolTensor`, which indicates whether the corresponding parameter
    is forgotten or not (1 for forgotten).

    Args:
        **kwargs: Forget result content as key-value pairs.

    """

    def __init__(
        self,
        **kwargs: Any,
    ):
        super().__init__()
        for __k, __v in kwargs.items():
            if (
                isinstance(__k, str)
                and isinstance(__v, torch.Tensor)
                and __v.dtype == torch.bool
            ):
                self[__k] = __v
            else:
                raise TypeError(
                    f"Expected key to be a `str` and value to be a "
                    f"`torch.BoolTensor`, but got {__k} and {__v}."
                )

    def get_num_forgotten(self, keyword: str) -> tuple[int, int]:
        """Get the number of forgotten parameters and the total number of
        parameters that contain the given keyword.

        Args:
            keyword: The keyword to search for, e.g., 'weight'.

        Returns:
            A tuple consisting of the number of forgotten parameters and the
            total number of parameters that contain the given keyword.

        """
        if keyword in self:
            __t = self[keyword]
            return __t.sum().item(), __t.numel()
        else:
            # Return a tuple consisting of the sum of the values of the
            # keys that contain the given key.
            _sum = (0, 0)
            for __k, __v in self.items():
                if keyword in __k.split("."):
                    _sum = (_sum[0] + __v.sum().item(), _sum[1] + __v.numel())
            if _sum == (0, 0):
                raise KeyError(f"Keyword {keyword} not found.")
            return _sum

    def __add__(self, other: ForgetResult) -> ForgetResult:
        _sum = deepcopy(self)
        for __k, __v in other.items():
            if __k in _sum:
                _sum[__k] = torch.logical_or(_sum[__k], __v)
            else:
                _sum[__k] = __v
        return _sum

    def __repr__(self):
        # print dict items with sorted names and tab
        _repr = "ForgetResult(\n"
        for __k, __v in sorted(self.items()):
            _repr += f"  {__k}: {__v.sum().item()}/{__v.numel()}\n"
        _repr = _repr + ")"
        return _repr
