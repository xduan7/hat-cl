from __future__ import annotations

import warnings
from typing import Any, Callable, Optional, Union

import torch

from .constants import DEF_HAT_GRAD_COMP_CLAMP, DEF_HAT_MAX_TRN_MASK_SCALE

Mask = Union[torch.FloatTensor, torch.BoolTensor]


class HATConfig:
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
        self.num_tasks = num_tasks
        self.mask_dim = mask_dim
        self.max_trn_mask_scale = max_trn_mask_scale
        self.grad_comp_clamp = grad_comp_clamp
        self.gate = gate
        if kwargs != {}:
            warnings.warn(f"Unrecognized kwargs: {kwargs}.")

    def keys(self):
        """Return the keys of the config. Useful for unpacking."""
        return self.__dict__.keys()

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


class ForgetResult:
    """Result of forgetting a task of a task dependent module.

    Args:
        num_forgotten_params: Number of (non-mask) parameters forgotten.
        num_trainable_params: Number of (non-mask) trainable parameters.
        num_forgotten_mask_params: Number of mask parameters forgotten.
        num_trainable_mask_params: Number of trainable mask parameters.

    """

    def __init__(
        self,
        num_forgotten_params: int = 0,
        num_trainable_params: int = 0,
        num_forgotten_mask_params: int = 0,
        num_trainable_mask_params: int = 0,
    ):
        self.num_forgotten_params = num_forgotten_params
        self.num_trainable_params = num_trainable_params
        self.num_forgotten_mask_params = num_forgotten_mask_params
        self.num_trainable_mask_params = num_trainable_mask_params

    def __add__(self, other: ForgetResult) -> ForgetResult:
        return ForgetResult(
            num_forgotten_params=self.num_forgotten_params
            + other.num_forgotten_params,
            num_trainable_params=self.num_trainable_params
            + other.num_trainable_params,
            num_forgotten_mask_params=self.num_forgotten_mask_params
            + other.num_forgotten_mask_params,
            num_trainable_mask_params=self.num_trainable_mask_params
            + other.num_trainable_mask_params,
        )

    def __repr__(self):
        return (
            f"Forgotten parameters: {self.num_forgotten_params} + "
            f"{self.num_forgotten_mask_params} (mask) / "
            f"Total trainable parameters: {self.num_trainable_params} + "
            f"{self.num_trainable_mask_params} (mask)"
        )
