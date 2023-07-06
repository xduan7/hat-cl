from __future__ import annotations

from typing import Any, Optional

import torch
from torch import classproperty
from torch import nn as nn

from ._base import ForgetResult, HATPayload, TaskIndexedModuleListABC
from .utils import register_mapping


@register_mapping
class TaskIndexedLayerNorm(TaskIndexedModuleListABC):
    """Task-indexed layer normalization layer.

    Task-indexed layer normalization layer s a wrapper class of the
    :class:`torch.nn.LayerNorm` layer. It has a list of normalization
    layers, each of which corresponds to a task, to prevent interference
    between tasks.

    """

    @classproperty
    def base_class(self) -> type[torch.nn.Module]:
        """Base class of task-indexed layer normalization."""
        return torch.nn.LayerNorm  # type: ignore

    def forward(self, pld: HATPayload) -> HATPayload:
        """Forward the payload by applying layer normalization of the given
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
        layer normalization module of the given task.

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
        return self._get_forget_result(
            task_id=task_id,
            module_name=module_name,
            locked_task_ids=locked_task_ids,
        )

    @classmethod
    def from_base_module(
        cls,
        base_module: nn.LayerNorm,
        num_tasks: Optional[int] = None,
        **kwargs: Any,
    ) -> TaskIndexedLayerNorm:
        """Create a task-indexed layer normalization layer from a PyTorch
        layer normalization layer by copying the parameters of the given
        module to the layer normalization layers of all the tasks.

        Args:
            base_module: The PyTorch layer normalization layer to be
                converted.
            num_tasks: The number of tasks. Defaults to `None` for
                compatibility  with other modules' `from_base_module`
                methods. If `None`, an error will be raised.
            **kwargs: For compatibility with other modules' `from_base_module`
                methods. Will be ignored here.

        Returns:
            The created task-indexed layer normalization layer.

        """
        if num_tasks is None:
            raise ValueError(
                "The number of tasks must be explicitly specified when "
                "creating a task-dependent layer normalization layer "
                "from a base module."
            )
        if base_module.elementwise_affine:
            _device = base_module.weight.device
            _dtype = base_module.weight.dtype
        else:
            _device = None
            _dtype = None

        _ti_ln = cls(
            num_tasks=num_tasks,
            normalized_shape=base_module.normalized_shape,
            eps=base_module.eps,
            elementwise_affine=base_module.elementwise_affine,
            device=_device,
            dtype=_dtype,
        )
        _ti_ln.load_from_base_module(base_module)
        return _ti_ln
