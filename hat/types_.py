from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any, Callable, Optional, Union

import torch

from .constants import DEF_HAT_GRAD_COMP_CLAMP, DEF_HAT_MAX_TRN_MASK_SCALE

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
        grad_comp_clamp: The maximum value of the gradient during gradient
            compensation. Defaults to `hat.constants.DEF_HAT_GRAD_COMP_CLAMP`.
        gate: The gating function to apply to the scaled attention vector.
            Defaults to `torch.sigmoid`.
        **kwargs: For polymorphism reasons. Should be empty.

    """

    def __init__(
        self,
        num_tasks: int,
        mask_dim: Optional[int] = None,
        max_trn_mask_scale: Optional[float] = DEF_HAT_MAX_TRN_MASK_SCALE,
        grad_comp_clamp: float = DEF_HAT_GRAD_COMP_CLAMP,
        gate: Callable[[torch.Tensor], torch.Tensor] = torch.sigmoid,
        **kwargs: Any,  # For polymorphism reasons.
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.mask_dim = mask_dim
        self.max_trn_mask_scale = max_trn_mask_scale
        self.grad_comp_clamp = grad_comp_clamp
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


class ForgetResult(dict[str, tuple[int, int]]):
    """Result of forgetting a task of a task dependent module.

    The result is a dictionary of the number of forgotten parameters and
    the number of total parameters for each child module of the task
    dependent module. The keys are the names of the parameters, e.g.,
    'l0.weight' and 'layer[2].conv1.bias'.

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
                and isinstance(__v, tuple)
                and len(__v) == 2
                and isinstance(__v[0], int)
                and isinstance(__v[1], int)
            ):
                self[__k] = __v  # type: ignore
            else:
                raise TypeError(
                    f"Expected key to be a `str` and value to be a tuple "
                    f"of two `int`s, but got {__k} and {__v}."
                )

    def __add__(self, other: ForgetResult) -> ForgetResult:
        _keys = set(self.keys()) & set(other.keys())
        if set(self.keys()) & set(other.keys()):
            raise ValueError(
                f"Cannot add two `ForgetResult` "
                f"with the same keys: {_keys}."
            )
        return ForgetResult(**{**self, **other})

    def __getitem__(self, key: str) -> tuple[int, int]:
        if key in self:
            return super().__getitem__(key)
        else:
            # Return a tuple consisting of the sum of the values of the
            # keys that contain the given key.
            _sum = (0, 0)
            for __k, __v in self.items():
                if key in __k.split("."):
                    _sum = (_sum[0] + __v[0], _sum[1] + __v[1])
            if _sum == (0, 0):
                raise KeyError(f"Key {key} not found.")
            return _sum

    def __repr__(self):
        # print dict items with sorted names and tab
        _repr = "ForgetResult(\n"
        for __k, __v in sorted(self.items()):
            _repr += f"  {__k}: {__v[0]}/{__v[1]}\n"
        _repr = _repr + ")"
        return _repr
