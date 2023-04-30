"""Contains the base classes and mixins for task dependent modules.

"""
from __future__ import annotations

import functools
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional, Union

import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle

from hat._base import TaskDependentMixin
from hat.exceptions import (
    LearningSuppressedWarning,
    NoParameterToForgetWarning,
)
from hat.types_ import ForgetResult, HATConfig

if TYPE_CHECKING:
    from hat.payload import HATPayload
else:
    HATPayload = Any


class HATPayloadCarrierMixin(nn.Module, ABC):
    """Mixin class for PyTorch module that implements a `forward` method
    that accepts `HATPayload` as the first argument and returns a another
    `HATPayload` instance.

    """

    @abstractmethod
    def forward(self, pld: HATPayload) -> HATPayload:
        """Forward the payload."""
        raise NotImplementedError


class TaskDependentModuleABC(
    TaskDependentMixin,
    HATPayloadCarrierMixin,
    ABC,
):
    """Abstract class for task dependent modules.

    ANY MODULE that behaves differently depending on the task id should
    inherit from this class or its children.

    """

    @abstractmethod
    def forget(
        self,
        task_id: int,
        dry_run: bool = False,
        module_name: Optional[str] = None,
        locked_task_ids: Optional[list[int]] = None,
    ) -> ForgetResult:
        """Forget the given tasks by resetting the parameters that are
        solely associated with the given task.

        Warnings:
            Changing the module parameters DOES NOT necessarily change the
            module output. There are many cases where the module output
            remains the same even after the parameters are changed. For
            example, with hard attention masks might zero out the changed
            output values, or dropout during training, or relu activation
            on negative values, etc.

        Args:
            task_id: The ID of the task to be forgotten. Cannot be `None`
                even if the module accepts `None` as a task id.
            dry_run: If `True`, the forgetting process will be simulated
                without actually changing the module. Defaults to `False`.
            module_name: The name of the module. If `None`, the module name
                will be inferred from the module class name.
            locked_task_ids: The list of task ids that are locked and
                cannot be forgotten. Defaults to `None`, in which case
                the module will lock all the tasks that have been trained
                except the task id to be forgotten.

        Returns:
            The forgetting result. See `hat.types_.ForgetResult` for more
            details.

        """
        raise NotImplementedError

    @abstractmethod
    def to_base_module(self, **kwargs: Any) -> nn.Module:
        """Convert the module to a base module for the given task id.

        Note that the base modules is not necessarily a `base_class`
        instance. It's a functionally equivalent module with the given
        parameters (e.g. task id, etc.) that might contain the `base_class`
        instance as a submodule.

        Returns:
            The base module.

        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_base_module(
        cls,
        base_module: nn.Module,
        **kwargs: Any,
    ) -> TaskDependentModuleABC:
        """Initialize a task dependent module from a base module.

        Args:
            base_module: The base module to initialize from.
            kwargs: The key word arguments for the conversion to a task
                dependent module.

        """
        raise NotImplementedError

    @abstractmethod
    def load_from_base_module(self, base_module: nn.Module):
        """Load the parameters from the given base module.

        Args:
            base_module: The base module to load the parameters from.

        """
        raise NotImplementedError

    @abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError

    def __str__(self) -> str:
        return self.__repr__()


class TaskIndexedModuleListABC(
    nn.ModuleList,
    TaskDependentModuleABC,
    ABC,
):
    """Abstract class for task dispatcher modules.

    The module is essentially a list of modules that are associated with
    different tasks. Upon receiving a payload, the module will dispatch the
    payload data to the corresponding module and return the result.

    Args:
        num_tasks: The number of tasks to dispatch.

    """

    def __init__(self, num_tasks: int, *args: Any, **kwargs: Any):
        super().__init__()
        for _ in range(num_tasks + 1):
            self.append(self.base_class(*args, **kwargs))

    @property
    def num_tasks(self) -> int:
        """The max number of tasks accepted by the module.

        For compatibility with other task dependent modules, the number of
        tasks is one less than the length of the module list. This is because
        `None` is also accepted as a task ID, but it's not an actual ID of
        any task. It's a special ID that is used to indicate that the module
        is not in task-dependent mode.

        Returns:
            The number of tasks accepted by the module.

        """
        return len(self) - 1

    @abstractmethod
    def forward(self, pld: HATPayload) -> HATPayload:
        """Forward the payload.

        In most cases, the user could use `forward_` to fill in the forward
        method with the `prefer_masked_data` argument set to their preference.
        See `_forward` for more details.

        Args:
            pld: The payload to be forwarded.

        Returns:
            The forwarded payload.

        """
        raise NotImplementedError

    def forward_(
        self,
        pld: HATPayload,
        use_masked_data: bool,
    ) -> HATPayload:
        """Forward the payload with mask application ordering specified.

        This method is intended to be used by the `forward` method of the
        children classes. It determines whether to apply the mask before or
        after the forward pass.

        The mask application ordering is crucial for some modules. Consider
        a `torch.nn.LayerNorm` module. If the mask is applied before the
        layer normalization, the supposedly masked value might become
        non-zero as the normalization is performed on all the features. In
        this case, we would like to apply the mask after the normalization.

        Args:
            pld: The payload to be forwarded.
            use_masked_data: If `True`, the mask will be applied before the
                forward pass. Otherwise, the mask will be applied after the
                forward pass.

        Returns:
            The forwarded payload.

        """
        from hat.payload import HATPayload

        if use_masked_data or pld.unmasked_data is None:
            _mask_applied = True
            _data = pld.masked_data
        else:
            _mask_applied = False
            _data = pld.unmasked_data
        _data = self[pld].forward(_data)
        return HATPayload(
            data=_data,
            masker=pld.masker,
            task_id=pld.task_id,
            mask_scale=pld.mask_scale,
            locked_task_ids=pld.locked_task_ids,
            prev_maskers=pld.prev_maskers,
            mask_applied=_mask_applied,
        )

    def __getitem__(self, key: Union[Optional[int], HATPayload]) -> nn.Module:
        from hat.payload import HATPayload

        if isinstance(key, HATPayload):
            key = key.task_id
        if key is None:
            return super().__getitem__(-1)
        return super().__getitem__(key)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  num_tasks: {self.num_tasks}\n"
            f"  base_module: {self[0]}\n"
            f")"
        )


class HATModuleABC(
    TaskDependentModuleABC,
    ABC,
):
    """Abstract class for modules that implements HAT (hard attention
    to the task) mechanism.

    The HAT mechanism is essentially a gradient manipulation trick that
    prevents the change of the parameters that are associated with other
    tasks, thereby preventing catastrophic forgetting. We use hooks to
    implement such gradient modification.

    The method `_register_grad_mod_hooks` should be called in the `forward`
    method to register the gradient modification hooks.

    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._grad_mod_hook_handles: list[RemovableHandle] = []

    def remove_grad_mod_hooks(self):
        """Remove the gradient modification hooks."""
        for handle in self._grad_mod_hook_handles:
            handle.remove()
        self._grad_mod_hook_handles = []

    @abstractmethod
    def _register_grad_mod_hooks(self, *args: Any, **kwargs: Any):
        """Register the gradient modification hooks.

        This method should be called in the `forward` method.

        """
        raise NotImplementedError


class HATMaskedModuleABC(
    HATModuleABC,
    HATPayloadCarrierMixin,
    ABC,
):
    """Internal abstract class for any module that (1) inherent from a
    weighted `HATModuleABC` modules, and (2) has a built-in `HATMasker`
    instance.

    This class encapsulates the common logic of HAT mechanism for modules
    that combines a PyTorch module with `weight` and `bias` parameters and
    a HAT masker, e.g. `HATLinear` and `HATConv2d`.
    Note that the `_condition_weight_grad` determines how masks are applied
    to the gradients of the `weight` parameter, which should be overloaded
    if the dimension of `weight` is not in the form of `[masker.num_features,
    prev_masker.num_features, *]`.

    Args:
        num_features: The number of features of the output data along the
            masked dimension.
        hat_config: The HAT configuration.
        device: The device to store the parameters.
        dtype: The data type of the parameters.
        kwargs: The key word arguments for the base `HATModuleABC` class.

    """

    def __init__(
        self,
        num_features: int,
        hat_config: HATConfig,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ):
        from .maskers import HATMasker

        super().__init__(**kwargs)
        self.masker = HATMasker(
            num_features=num_features,
            hat_config=hat_config,
            device=device,
            dtype=dtype,
        )

    @property
    def num_tasks(self) -> int:
        """The max number of tasks accepted by the module."""
        return self.masker.num_tasks

    def forward(self, pld: HATPayload) -> HATPayload:
        """Forward the payload through the module.

        The payload will go through the following steps:
        (1) the data will be processed by the base class forward method;
        (2) the gradient modification hooks will be registered if the task
        id is not `None` and the module is in training mode with gradient
        enabled;
        (3) the payload will be passed through the masker;

        Args:
            pld: The payload to be forwarded.

        Returns:
            The forwarded payload.

        """
        from hat.payload import HATPayload

        _pld = HATPayload(
            data=self.base_class.forward(self, pld.data),
            masker=pld.masker,
            task_id=pld.task_id,
            mask_scale=pld.mask_scale,
            locked_task_ids=pld.locked_task_ids,
            prev_maskers=pld.prev_maskers,
            mask_applied=True,
        )
        if (
            self.training
            and torch.is_grad_enabled()
            and _pld.task_id is not None
        ):
            _prev_maskers = _pld.masker
            _prev_locked_mask = (
                _prev_maskers.get_locked_mask(
                    task_id=_pld.task_id,
                    locked_task_ids=_pld.locked_task_ids,
                )
                if _prev_maskers is not None
                else None
            )
            _curr_locked_mask = self.masker.get_locked_mask(
                task_id=_pld.task_id,
                locked_task_ids=_pld.locked_task_ids,
            )
            self._register_grad_mod_hooks(
                prev_locked_mask=_prev_locked_mask,
                curr_locked_mask=_curr_locked_mask,
            )
        return self.masker.forward(_pld)

    def forget(
        self,
        task_id: int,
        dry_run: bool = False,
        module_name: Optional[str] = None,
        locked_task_ids: Optional[list[int]] = None,
    ) -> ForgetResult:
        """Forget the task with the given task id.

        This function will find the weights/biases that are associated
        only with the given task id and reset them to random values if
        the execution is not dry run. The forgettable weights/biases are
        determined by the masker of this module, and the maskers that are
        immediately before this module and generate the input data.

        This function assumes that the `weight` parameter is of shape
        `[masker.num_features, prev_masker.num_features, *]`. Overload
        this function if the assumption does not hold.

        Args:
            task_id: The ID of the task to be forgotten. Cannot be `None`
                even if the module accepts `None` as a task id.
            dry_run: If `True`, the forgetting process will be simulated
                without actually changing the module. Defaults to `False`.
            module_name: The name of the module. If `None`, the module name
                will be inferred from the module class name.
            locked_task_ids: The list of task ids that are locked and
                cannot be forgotten. Defaults to `None`, in which case
                the module will lock all the tasks that have been trained
                except the task id to be forgotten.

        Returns:
            The forgetting result. See `hat.types_.ForgetResult` for more
            details.

        """
        _forgettable_mask, _, _ = self.masker.get_forgettable_mask(
            task_id=task_id,
            locked_task_ids=locked_task_ids,
        )
        _prev_forgettable_mask = self.masker.get_prev_forgettable_mask(
            task_id=task_id,
            locked_task_ids=locked_task_ids,
        ).expand(self.weight.shape[1])
        _weight_change_pos = torch.outer(
            _forgettable_mask, _prev_forgettable_mask
        )
        _bias_change_pos = _forgettable_mask
        if not dry_run:
            # TODO: there are probably better way to reset the parameters
            #  other than using normal distribution
            # Assume that weight is of shape [out, in, *]
            self.weight.data[_weight_change_pos] = self.weight.data[
                _weight_change_pos
            ].normal_()
            if self.bias is not None:
                self.bias.data[_bias_change_pos] = self.bias.data[
                    _bias_change_pos
                ].normal_()
        _module_name = module_name or self.__class__.__name__
        _forget_result = {f"{_module_name}.weight": _weight_change_pos}
        _forget_num_params = _weight_change_pos.sum().item()
        if self.bias is not None:
            _forget_result[f"{_module_name}.bias"] = _bias_change_pos
            _forget_num_params += _bias_change_pos.sum().item()
        _forget_result = ForgetResult(**_forget_result)
        if _forget_num_params == 0:
            warnings.warn(
                f"No parameters are forgotten for task {task_id}. "
                f"This task might be locked behind other tasks.",
                NoParameterToForgetWarning,
            )
        # Masker must forget last because the masking information is
        # required for the base module to forget.
        _masker_forget_result = self.masker.forget(
            task_id=task_id,
            dry_run=dry_run,
            module_name=f"{_module_name}.masker",
        )
        return _forget_result + _masker_forget_result

    def load_from_base_module(self, base_module: nn.Module):
        """Load the weight/bias from the base module."""
        self.weight.data.copy_(base_module.weight.data)
        if self.bias is not None:
            self.bias.data.copy_(base_module.bias.data)

    def _register_grad_mod_hooks(
        self,
        prev_locked_mask: Optional[torch.BoolTensor],
        curr_locked_mask: torch.BoolTensor,
    ):
        """Register the gradient conditioning hooks.

        This function is a helper function that registers the gradient
        conditioning functions `_condition_weight_grad` and
        `_condition_bias_grad` to the weight and bias respectively.

        Args:
            prev_locked_mask: The binary mask generated by the previous
                maskers that indicates which weights are locked along the
                second dimension of the weight.
            curr_locked_mask: The binary mask generated by the current
                masker that indicates which weights/biases are locked
                along the first dimension of the weight/bias.

        """
        self.remove_grad_mod_hooks()
        if self.weight.requires_grad:
            _w_grad_cond_hook = functools.partial(
                self._condition_weight_grad,
                prev_locked_mask=prev_locked_mask,
                curr_locked_mask=curr_locked_mask,
            )
            _w_grad_cond_hook.__torch_unserializable__ = True  # type: ignore
            _w_grad_cond_hook_handle = self.weight.register_hook(
                _w_grad_cond_hook
            )
            self._grad_mod_hook_handles.append(_w_grad_cond_hook_handle)
        if self.bias is not None and self.bias.requires_grad:
            _b_grad_cond_hook = functools.partial(
                self._condition_bias_grad,
                curr_locked_mask=curr_locked_mask,
            )
            _b_grad_cond_hook.__torch_unserializable__ = True  # type: ignore
            _b_grad_cond_hook_handle = self.bias.register_hook(
                _b_grad_cond_hook
            )
            self._grad_mod_hook_handles.append(_b_grad_cond_hook_handle)

    @staticmethod
    def _condition_weight_grad(
        weight_grad: torch.Tensor,
        prev_locked_mask: Optional[torch.BoolTensor],
        curr_locked_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        """Condition the gradient of the weight by zeroing out the gradient
        of the biases that are locked by other tasks.

        References:
            [1] https://arxiv.org/abs/1801.01423 (Chapter 2.3)

        Args:
            weight_grad: The gradient of the weight.
            prev_locked_mask: See `_register_grad_mod_hooks`.
            curr_locked_mask: See `_register_grad_mod_hooks`.

        """
        curr_locked_mask = curr_locked_mask.reshape(
            -1, *([1] * (weight_grad.dim() - 1))
        ).expand_as(weight_grad)
        if prev_locked_mask is None:
            _weight_grad_coef = 1 - curr_locked_mask.float()
        else:
            if torch.all(prev_locked_mask):
                warnings.warn(
                    "The learning of new featured is suppressed because "
                    "the input HAT mask is depleted by the previous tasks.",
                    LearningSuppressedWarning,
                )
            prev_locked_mask = prev_locked_mask.reshape(
                1, -1, *([1] * (weight_grad.dim() - 2))
            ).expand_as(weight_grad)
            _weight_grad_coef = (
                1.0
                - torch.logical_and(
                    curr_locked_mask.expand_as(weight_grad),
                    prev_locked_mask.expand_as(weight_grad),
                ).float()
            )
        if torch.all(_weight_grad_coef == 0):
            warnings.warn(
                "The learning is completely suppressed because both input "
                "and output HAT masks are depleted by the previous tasks.",
                LearningSuppressedWarning,
            )
        return weight_grad * _weight_grad_coef

    @staticmethod
    def _condition_bias_grad(
        bias_grad: torch.Tensor,
        curr_locked_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        """Condition the gradient of the bias by zeroing out the gradient
        of the biases that are locked by other tasks.

        References:
            [1] https://arxiv.org/abs/1801.01423 (Chapter 2.3)

        Args:
            bias_grad: The gradient of the bias.
            curr_locked_mask: See `_register_grad_mod_hooks`.

        """
        _bias_grad_coef = 1.0 - curr_locked_mask.float()
        return bias_grad * _bias_grad_coef

    def __repr__(self) -> str:
        return self.base_class.__repr__(self)  # type: ignore
