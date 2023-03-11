from __future__ import annotations

import functools
import warnings
from typing import Any, Literal, Optional, Sequence, Union

import torch
import torch.nn as nn
from torch import classproperty

from hat.exceptions import HATInitializationError, MaskerLockedException
from hat.modules._base import HATModuleABC, HATPayloadCarrierMixin
from hat.modules.utils import register_mapping
from hat.payload import HATPayload
from hat.types_ import ForgetResult, HATConfig, Mask

from ._base import VectorMaskerABC
from .attention_masker import AttentionMasker


class _HATMakerRegulator:
    """A helper class that keeps track of the variables that are used to
    generate the regularization term for the `HATMaker` module.

    """

    def __init__(self, marker: HATMasker):
        self.marker = marker
        self.depth: Optional[int] = None
        self.desired_utils: Optional[dict[int, float]] = None
        self.desired_masks: Optional[list[dict[int, torch.Tensor]]] = None

    def get_reg_term(
        self,
        reg_strategy: Literal["uniform", "heuristic"],
        *args: Any,
        **kwargs: Any,
    ) -> float:
        if reg_strategy == "uniform":
            return self.get_uniform_reg_term(*args, **kwargs)
        # elif reg_strategy == "heuristic":
        #     return self.get_heuristic_reg_term(*args, **kwargs)
        else:
            raise ValueError(
                f"Unknown regularization strategy: {reg_strategy}"
            )

    def get_uniform_reg_term(
        self,
        task_id: int,
        mask_scale: Optional[float],
        locked_task_ids: Optional[Sequence[int]] = None,
    ) -> float:
        """Get the uniform regularization term.

        This regularization term penalizes usage of the mask elements of the
        current task that are not activated for the "locked" tasks,
        in a uniform manner.

        The current implementation differs from the original paper [1] in
        the way that the regularization term is calculated by individual
        layer, instead of the whole network.

        Args:
            task_id: The ID of the current task.
            mask_scale: The scale of the mask.
            locked_task_ids: The IDs of the tasks that are locked. Check
                `HATMaker.get_locked_mask` for more details.

        References:
            [1] https://arxiv.org/abs/1801.01423 (Equation 5)

        """
        _masks = self.marker.get_mask(task_id=task_id, mask_scale=mask_scale)
        _locked_mask = self.marker.get_locked_mask(
            task_id=task_id,
            locked_task_ids=locked_task_ids,
        )
        _aux = 1.0 - _locked_mask
        _reg = (_masks * _aux).sum()
        _ttl = _aux.sum()
        return float(_reg / _ttl)

    # def get_heuristic_reg_term(self) -> float:
    #     pass


@register_mapping
class HATMasker(
    VectorMaskerABC,
    HATModuleABC,
    HATPayloadCarrierMixin,
):
    """The HAT masker module.

    This module is used to generate the mask for the HAT module. It also
    keeps track of the maskers of the previous layers, and provides the
    information for weighted modules (e.g., `HATLinear`) to find the
    parameters that are associated only to a certain task. This enables
    training or forgetting of a task without affecting the other tasks.

    Args:
        num_features: The number of features in the input.
        hat_config: The configuration of the HAT module.
        device: The device of the module.
        dtype: The data type of the module.

    """

    def __init__(
        self,
        num_features: int,
        hat_config: HATConfig,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(hat_config.mask_dim)
        _num_tasks = hat_config.num_tasks
        self.attention = nn.ParameterList(
            [
                nn.Parameter(
                    torch.randn(num_features, device=device, dtype=dtype)
                )
                for _ in range(_num_tasks)
            ]
        )
        self.regulator = _HATMakerRegulator(marker=self)

        self._max_trn_mask_scale = hat_config.max_trn_mask_scale
        self._grad_comp_clamp = hat_config.grad_comp_clamp
        self._gate = hat_config.gate

        self._task_trained = nn.Parameter(
            torch.zeros(_num_tasks, dtype=torch.bool, device=device),
            requires_grad=False,
        )
        self._cached_binary_mask: dict[int, torch.Tensor] = {}
        self._prev_maskers: Optional[list[HATMasker]] = None

    @classproperty
    def base_class(self) -> type[nn.Module]:
        """Base class of the HAT masker module."""
        return AttentionMasker  # type: ignore

    @property
    def num_tasks(self) -> int:
        """The max number of tasks accepted by the module."""
        return len(self.attention)

    @property
    def trained_task_ids(self) -> list[int]:
        """The IDs of the tasks that have been trained."""
        _trained_task_ids = torch.nonzero(self._task_trained).flatten()
        return _trained_task_ids.tolist()  # type: ignore

    def forward(self, pld: HATPayload) -> HATPayload:
        """Forward the payload by appending the masker to the payload.

        Note that this function does not generate nor apply the mask. It
        register the gradient compensation hook as if the mask is applied.
        Then it returns the payload with itself (and its previous maskers)
        for context. The mask will be applied whenever the masked data is
        accessed.

        Args:
            pld: The payload to be forwarded.

        Returns:
            The forwarded payload.

        """
        if self.training:
            if pld.mask_scale is None:
                raise ValueError(
                    "The mask scale must be specified during training."
                )
            elif pld.mask_scale > self._max_trn_mask_scale:  # type: ignore
                warnings.warn(
                    "The mask scale is greater than the maximum training "
                    "mask scale. This might slow down the training of mask "
                    "due to insufficient gradient compensation.",
                    RuntimeWarning,
                )
            if torch.is_grad_enabled():
                if pld.task_id is not None:
                    self._task_trained[pld.task_id] = True
                    self._cached_binary_mask.pop(pld.task_id, None)
                    self._register_grad_mod_hooks(
                        task_id=pld.task_id,
                        mask_scale=pld.mask_scale,
                    )
        _prev_maskers = self._get_prev_maskers(pld=pld)
        return HATPayload(
            # The mask only applies to the data when requested, not here.
            data=pld.data,
            masker=self,
            task_id=pld.task_id,
            mask_scale=pld.mask_scale,
            prev_maskers=_prev_maskers,
            locked_task_ids=pld.locked_task_ids,
        )

    def get_mask(
        self,
        task_id: Optional[int],
        mask_scale: Optional[float],
    ) -> Mask:
        """Get the mask for the given task ID and mask scale.

        Args:
            task_id: The ID of the task. If set to `None`, the mask will
                be disabled, meaning that the mask will be all ones.
            mask_scale: The scale of the attention. If set to `None`, the
                mask will be binary. See `get_binary_mask()` for more details.

        Returns:
            The mask tensor.

        """
        if task_id is None:
            return torch.ones_like(self.attention[0])
        if mask_scale is None:
            return self.get_binary_mask(task_id=task_id)
        else:
            return self._gate(self.attention[task_id] * mask_scale)

    def get_binary_mask(
        self,
        task_id: int,
    ) -> Mask:
        """Get the binary mask for the given task ID.

        Note that this method will not generate gradient, and it shall only
        be used to calculate the context of the mask for locating parameters
        that are associated with a certain task.

        Args:
            task_id: The ID of the task.

        Returns:
            The binary mask tensor.
        """
        if task_id not in self._cached_binary_mask:
            self._cached_binary_mask[task_id] = torch.BoolTensor(
                self.attention[task_id] > 0
            )
        return self._cached_binary_mask[task_id]

    def get_locked_mask(
        self,
        task_id: int,
        locked_task_ids: Optional[Sequence[int]] = None,
    ) -> Mask:
        """Get the mask where activation means that the position is locked
        by the given locked task IDs, during the training of the given task.

        Args:
            task_id: The ID of the (current training) task.
            locked_task_ids: The IDs of the tasks that are locked. If set
                to `None`, the locked task IDs will be inferred from the
                trained task IDs. See `_infer_locked_task_ids()` for more
                details.

        Returns:
            The locked mask tensor.

        """
        if locked_task_ids is None:
            locked_task_ids = self._infer_locked_task_ids(task_id)
        else:
            if task_id in locked_task_ids:
                raise MaskerLockedException(
                    f"Current task ID {task_id} is in the "
                    f"locked task ID list {locked_task_ids}."
                )
        if len(locked_task_ids) == 0:
            return torch.zeros_like(self.attention[0]).bool()
        _locked_mask = torch.stack(
            [self.get_binary_mask(__task_id) for __task_id in locked_task_ids],
        ).any(dim=0)
        return _locked_mask

    def forget(
        self,
        task_id: int,
        dry_run: bool = False,
    ) -> ForgetResult:
        """Forget the given task ID by reinitializing the attention.

        Args:
            task_id: The ID of the task to be forgotten. Cannot be `None`.
            dry_run: If `True`, the forgetting process will be simulated
                without actually changing the module. Defaults to `False`.

        Returns:
            The forgetting result. See `hat.types_.ForgetResult` for more
            details.

        """
        _num_forgotten_params = self.attention[task_id].numel()
        if task_id not in self.trained_task_ids:
            raise IndexError(
                f"Untrained task ID {task_id} cannot be forgotten."
            )
        if not dry_run:
            self.attention[task_id].data.normal_()
        _num_trainable_params = sum(
            __p.numel() for __p in self.attention if __p.requires_grad
        )
        return ForgetResult(
            num_forgotten_mask_params=_num_forgotten_params,
            num_trainable_mask_params=_num_trainable_params,
        )

    def get_forgettable_mask(
        self,
        task_id: int,
    ) -> tuple[Mask, Mask, Mask]:
        """Helper method to get the masks that are used to locate the
        parameters that solely belong to the given task ID by checking
        the itself.

        Definition:
            - Forgettable mask: The mask where activation means that the
                position can be safely forgotten.
            - Inclusive mask: The mask where activation means that the
                position is associated with the given task ID.
            - Exclusive mask: The mask where activation means that the
                position is associated with any of the locked task IDs.

        Essentially, the forgettable mask is defined as `F = I & (not E)`,
        where `I` is the inclusive mask and `E` is the exclusive mask.

        Args:
            task_id: The ID of the task.

        Returns:
            The forgettable mask, inclusive mask, and exclusive mask.

        """

        _inclusive_mask = self.get_binary_mask(task_id)
        # All the tasks other than the ones to be forgotten are locked.
        _locked_task_ids = self._infer_locked_task_ids(task_id)
        if len(_locked_task_ids) == 0:
            _exclusive_mask = torch.zeros_like(self.attention[0]).bool()
        else:
            _exclusive_mask = torch.stack(
                [
                    self.get_binary_mask(__task_id)
                    for __task_id in _locked_task_ids
                ],
            ).any(dim=0)
        _forgettable_mask = _inclusive_mask & (~_exclusive_mask)
        return _forgettable_mask, _inclusive_mask, _exclusive_mask

    def get_prev_forgettable_mask(
        self,
        task_id: int,
    ) -> Mask:
        """Helper method to get the mask that is used to locate the
        parameters that are associated with the given task ID by checking
        its previous maskers.

        This function follows the same definition as `get_forgettable_mask(
        )`. However, there might be more than one previous maskers, and the
        forgettable mask is defined as
        `F = I_1 | I_2 | ... | I_n & (not E_1 | E_2 | ... | E_n)`, where
        `I_i` is the inclusive mask of the `i`-th previous masker and
        `E_i` is the exclusive mask of the `i`-th previous masker.

        Args:
            task_id: The ID of the task.

        Returns:
            The forgettable mask of the previous maskers.

        """
        if self._prev_maskers is None:
            raise HATInitializationError(
                "The previous maskers of a `HATMasker` are not specified. "
                "Please run the forward pass of the network for at least "
                "once."
            )
        if len(self._prev_maskers) == 0:
            return torch.ones(1).bool()
        _prev_incl_masks, _prev_excl_masks = [], []
        for __prev_masker in self._prev_maskers:
            (
                _,
                __prev_incl_mask,
                __prev_excl_mask,
            ) = __prev_masker.get_forgettable_mask(task_id)
            _prev_incl_masks.append(__prev_incl_mask)
            _prev_excl_masks.append(__prev_excl_mask)

        _prev_incl_mask = torch.stack(_prev_incl_masks).any(dim=0)
        _prev_excl_mask = torch.stack(_prev_excl_masks).any(dim=0)
        _prev_forgettable_mask = _prev_incl_mask & (~_prev_excl_mask)
        return _prev_forgettable_mask

    def to_base_module(
        self,
        task_id: Optional[int] = None,
        trn_mask_scale: Optional[float] = None,
        **kwargs: Any,
    ) -> Union[AttentionMasker, nn.Identity]:
        """Convert the HAT masker to a simple attention masker or an
        identity module, based on the given task ID. The returned module
        will have the same parameters as the HAT masker with the given
        task ID.

        Args:
            task_id: The ID of the task. If `None`, an identity module
                will be returned.
            trn_mask_scale: The scale of the training mask. If `None`,
                the scale of the masker will be used.
            **kwargs: Additional arguments to be passed to the base
                module.

        Returns:
            A attention masker with the same parameters as the HAT
            masker with the given task ID, or an identity module if the
            task ID is `None`.

        """
        if task_id is None:
            return nn.Identity()
        _attention_masker = AttentionMasker(
            num_features=self.attention[task_id].numel(),
            trn_mask_scale=trn_mask_scale,
            gate=self._gate,
            mask_dim=self.mask_dim,
            device=self.attention[task_id].device,
            dtype=self.attention[task_id].dtype,
        )
        _attention_masker.attention.data = self.attention[task_id].data.clone()
        return _attention_masker

    @classmethod
    def from_base_module(
        cls: type[HATMasker],
        base_module: AttentionMasker,
        **kwargs: Any,
    ) -> HATMasker:
        """Convert a simple attention masker to a HAT masker by copying
        the attention to all the tasks.

        Args:
            base_module: The base attention masker.
            **kwargs: Additional keyword arguments for the `HATConfig`
                constructor.

        Returns:
            A HAT masker with the same parameters as the base attention
            masker for all the tasks.

        """
        _hat_config = HATConfig(**kwargs)
        _hat_masker = cls(
            num_features=base_module.attention.numel(),
            hat_config=_hat_config,
            device=base_module.attention.device,
            dtype=base_module.attention.dtype,
        )
        _hat_masker.load_from_base_module(base_module)
        return _hat_masker

    def load_from_base_module(self, base_module: AttentionMasker):
        """Load the parameters from the base attention masker to all the
        tasks.

        Args:
            base_module: The base attention masker.

        """
        for __attn in self.attention:
            __attn.data.copy_(base_module.attention.data)

    def _register_grad_mod_hooks(
        self,
        task_id: Optional[int],
        mask_scale: Optional[float],
    ):
        """Register the gradient compensation hook.

        This function is a helper function that registers the gradient
        compensation function `_compensate_attention_grad()` to the attention.

        Args:
            task_id: The ID of the current training task.
            mask_scale: The scale of the attention.

        """
        self.remove_grad_mod_hooks()
        _grad_comp_hook_handle = self.attention[task_id].register_hook(
            functools.partial(
                self._compensate_attention_grad,
                attention=self.attention[task_id],
                mask_scale=mask_scale,
                max_trn_mask_scale=self._max_trn_mask_scale,
                grad_comp_clamp=self._grad_comp_clamp,
            )
        )
        self._grad_mod_hook_handles.append(_grad_comp_hook_handle)

    @staticmethod
    def _compensate_attention_grad(
        attention_grad: torch.Tensor,
        attention: torch.Tensor,
        mask_scale: Optional[float],
        max_trn_mask_scale: float,
        grad_comp_clamp: float,
    ) -> torch.Tensor:
        """Compensate the gradient of hard attention to make attention masks
        easier to train.

        References:
            [1] https://arxiv.org/abs/1801.01423 (Chapter 2.5)

        """
        _num = torch.cosh(
            torch.clamp(
                mask_scale * attention, -grad_comp_clamp, grad_comp_clamp
            )
        )
        _num = (_num + 1) * max_trn_mask_scale
        _den = torch.cosh(attention)
        _den = (_den + 1) * mask_scale
        attention_grad *= _num / _den
        return attention_grad

    def _get_prev_maskers(
        self,
        pld: HATPayload,
    ) -> Optional[list[HATMasker]]:
        """Assign the previous maskers and return the current masker if needed.

        We only need to assign the previous maskers once. If the previous
        maskers are already assigned, we will return `None` to indicate that
        the current masker is not needed.
        During the initial assignment, if the previous maskers in the
        payload is `None`, it indicates that the current masker is the first
        masker in the chain. In this case, we assign an empty list to the
        previous maskers. Otherwise, we assign the previous maskers in the
        payload to the current masker.

        Args:
            pld: The payload to forward.

        Returns:
            A list of previous maskers for the payload that passes on to
            the next module, or `None` if the previous maskers are already
            assigned.

        """
        if self._prev_maskers is None:
            if pld.prev_maskers is None:
                self._prev_maskers = []
            else:
                self._prev_maskers = pld.prev_maskers
            return [self]
        else:
            assert pld.prev_maskers is None
            return None

    def _infer_locked_task_ids(
        self,
        task_ids: Optional[Union[int, Sequence[int]]],
    ) -> Sequence[int]:
        """Infer the task IDs that are supposed to be locked, which are all
        the task IDs that are trained except the given task IDs.

        Args:
            task_ids: The task IDs that current in use (e.g. the current
                task ID for training).

        Returns:
            A list of task IDs that are supposed to be locked.

        """
        # By default, all trained tasks are locked except the current task.
        _trained_task_ids = torch.nonzero(self._task_trained).flatten()
        _locked_task_ids: list[int] = _trained_task_ids.tolist()
        # Remove all the task_ids from the locked task ids.
        if task_ids is not None:
            if isinstance(task_ids, int):
                task_ids = [task_ids]
            for __task_id in task_ids:
                if __task_id in _locked_task_ids:
                    _locked_task_ids.remove(__task_id)
        return _locked_task_ids

    def __str__(self) -> str:
        return (
            f"HATMasker("
            f"num_features={self.attention[0].numel()}, "
            f"num_tasks={self.num_tasks}, "
            f"gate={self._gate.__name__})"
        )

    def __repr__(self) -> str:
        return self.__str__()
