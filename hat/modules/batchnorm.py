from __future__ import annotations

from abc import ABC
from copy import deepcopy
from typing import Any, Optional

from torch import nn as nn

# noinspection PyProtectedMember
from torch.nn.modules.batchnorm import _BatchNorm

from hat.payload import HATPayload

from ._base import ForgetResult, TaskIndexedModuleListABC
from .utils import get_num_params, register_mapping


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
    ) -> ForgetResult:
        """Forget the given tasks by resetting the parameters of the
        batch normalization module of the given task.

        Args:
            task_id: The ID of the task to be forgotten. Cannot be `None`
                even if the module accepts `None` as a task id.
            dry_run: If `True`, the forgetting process will be simulated
                without actually changing the module. Defaults to `False`.

        Returns:
            The forgetting result. See `hat.types_.ForgetResult` for more
            details.

        """
        _num_forgotten_params = 0
        _num_trainable_params = get_num_params(self)
        if not dry_run:
            self[task_id].reset_parameters()
        _num_forgotten_params += get_num_params(self[task_id])
        # `running_mean` and `running_var` are not parameters, as they are
        # not included in `self.parameters()`, but `weight` and `bias` are.
        return ForgetResult(
            num_forgotten_params=_num_forgotten_params,
            num_trainable_params=_num_trainable_params,
        )

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
        _ti_bn = cls(num_tasks=num_tasks, **kwargs)
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

    def base_class(self) -> type[nn.Module]:
        """Base class of task-indexed batch normalization 1D."""
        return nn.BatchNorm1d  # type: ignore


@register_mapping
class TaskIndexedBatchNorm2d(_TaskIndexedBatchNorm):
    """Task-indexed 2D batch normalization layer."""

    def base_class(self) -> type[nn.Module]:
        """Base class of task-indexed batch normalization 2D."""
        return nn.BatchNorm2d  # type: ignore


@register_mapping
class TaskIndexedBatchNorm3d(_TaskIndexedBatchNorm):
    """Task-indexed 3D batch normalization layer."""

    def base_class(self) -> type[nn.Module]:
        """Base class of task-indexed batch normalization 3D."""
        return nn.BatchNorm3d  # type: ignore
