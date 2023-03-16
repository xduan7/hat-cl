from __future__ import annotations

from abc import ABC
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Optional

import torch
from torch import classproperty
from torch import nn as nn

# noinspection PyProtectedMember
from torch.nn.modules.batchnorm import _BatchNorm

from ._base import ForgetResult, TaskIndexedModuleListABC
from .utils import register_mapping

if TYPE_CHECKING:
    from hat.payload import HATPayload
else:
    HATPayload = Any


class _TaskIndexedBatchNorm(TaskIndexedModuleListABC, ABC):
    """Abstract class for task-indexed batch normalization layers.

    It implements all the methods for the task-indexed batch normalization
    except for the `base_class` property, which differs for different
    batch normalization layers.

    """

    def forward(self, pld: HATPayload) -> HATPayload:
        """Forward the payload by applying batch normalization of the given
        task to the unmasked data.

        Args:
            pld: The payload to be forwarded.

        Returns:
            The forwarded payload.

        """
        return self.forward_(pld, use_masked_data=False)

    def forget(
        self,
        task_id: int,
        dry_run: bool = False,
        module_name: Optional[str] = None,
        locked_task_ids: Optional[list[int]] = None,
    ) -> ForgetResult:
        """Forget the given tasks by resetting the parameters of the
        batch normalization module of the given task.

        Args:
            task_id: The ID of the task to be forgotten. Cannot be `None`
                even if the module accepts `None` as a task id.
            dry_run: If `True`, the forgetting process will be simulated
                without actually changing the module. Defaults to `False`.
            module_name: The name of the module. If `None`, the module name
                will be inferred from the module class name.
            locked_task_ids: The list of task ids that are locked and
                cannot be forgotten. This is ignored here, as forgetting
                of a task does not affect the other tasks.

        Returns:
            The forgetting result. See `hat.types_.ForgetResult` for more
            details.

        """
        if not dry_run:
            self[task_id].reset_parameters()
        # `running_mean` and `running_var` are not parameters, as they are
        # not included in `self.parameters()`, but `weight` and `bias` are.
        _weight_forget_result = torch.zeros(
            (len(self), self[task_id].weight.numel()), dtype=torch.bool
        )
        _weight_forget_result[task_id, :] = True
        _bias_forget_result = torch.zeros(
            (len(self), self[task_id].bias.numel()), dtype=torch.bool
        )
        _bias_forget_result[task_id, :] = True
        _module_name = module_name or self.__class__.__name__
        _forget_result = {
            f"{_module_name}.weight": _weight_forget_result,
            f"{_module_name}.bias": _bias_forget_result,
        }
        return ForgetResult(**_forget_result)

    def to_base_module(
        self,
        task_id: Optional[int] = None,
        **kwargs: Any,
    ) -> _BatchNorm:
        """Convert the task-indexed batch normalization layer to a
        PyTorch batch normalization layer by deep copying the module of
        the given task.

        Args:
            task_id: The ID of the task to be converted. If `None`, the
                module that corresponds to the non-task-specific case will
                be converted.
            **kwargs: For compatibility with other modules' `to_base_module`
                methods. Will be ignored here.

        Returns:
            The converted PyTorch batch normalization layer.

        """
        return deepcopy(self[task_id])

    @classmethod
    def from_base_module(
        cls: type[_TaskIndexedBatchNorm],
        base_module: _BatchNorm,
        num_tasks: Optional[int] = None,
        **kwargs: Any,
    ) -> _TaskIndexedBatchNorm:
        """Create a task-indexed batch normalization layer from a PyTorch
        batch normalization layer by copying the parameters of the given
        module to the batch normalization layers of all the tasks.

        Args:
            base_module: The PyTorch batch normalization layer to be
                converted.
            num_tasks: The number of tasks. Defaults to `None` for
                compatibility  with other modules' `from_base_module`
                methods. If `None`, an error will be raised.
            **kwargs: For compatibility with other modules' `from_base_module`
                methods. Will be ignored here.

        Returns:
            The created task-indexed batch normalization layer.

        """
        if num_tasks is None:
            raise ValueError(
                "The number of tasks must be explicitly specified when "
                "creating a task-dependent batch normalization layer "
                "from a base module."
            )
        if base_module.affine:
            _device = base_module.weight.device
            _dtype = base_module.weight.dtype
        elif base_module.track_running_stats:
            _device = base_module.running_mean.device
            _dtype = base_module.running_mean.dtype
        else:
            _device = None
            _dtype = None
        _ti_bn = cls(
            num_tasks=num_tasks,
            num_features=base_module.num_features,
            eps=base_module.eps,
            momentum=base_module.momentum,
            affine=base_module.affine,
            track_running_stats=base_module.track_running_stats,
            device=_device,
            dtype=_dtype,
        )
        _ti_bn.load_from_base_module(base_module)
        return _ti_bn

    def load_from_base_module(self, base_module: nn.Module):
        """Load the parameters of the given module to all the modules of
        the task-indexed batch normalization layer.

        Args:
            base_module: The module from which the parameters will be
                loaded.

        """
        for __task_id in range(self.num_tasks):
            self[__task_id].load_state_dict(base_module.state_dict())


@register_mapping
class TaskIndexedBatchNorm1d(_TaskIndexedBatchNorm):
    """Task-indexed 1D batch normalization layer."""

    @classproperty
    def base_class(self) -> type[nn.Module]:
        """Base class of task-indexed batch normalization 1D."""
        return nn.BatchNorm1d  # type: ignore


@register_mapping
class TaskIndexedBatchNorm2d(_TaskIndexedBatchNorm):
    """Task-indexed 2D batch normalization layer."""

    @classproperty
    def base_class(self) -> type[nn.Module]:
        """Base class of task-indexed batch normalization 2D."""
        return nn.BatchNorm2d  # type: ignore


@register_mapping
class TaskIndexedBatchNorm3d(_TaskIndexedBatchNorm):
    """Task-indexed 3D batch normalization layer."""

    @classproperty
    def base_class(self) -> type[nn.Module]:
        """Base class of task-indexed batch normalization 3D."""
        return nn.BatchNorm3d  # type: ignore
