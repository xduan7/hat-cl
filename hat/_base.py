"""Common base classes for the whole framework."""
from abc import ABC, abstractmethod

from torch import classproperty


class TaskDependentMixin(ABC):
    """Mixin class for all things task dependent (e.g. parameter, module)."""

    @classproperty
    @abstractmethod
    def base_class(self) -> type:
        """Get the base class of the task dependent class/instance.

        For example, the base class of `HardAttentionLinear` is `nn.Linear`,
        and the base class of `TaskDependentBatchNorm1d` is `nn.BatchNorm1d`.

        """
        raise NotImplementedError

    @property
    @abstractmethod
    def num_tasks(self) -> int:
        """The max number of tasks accepted by the module."""
        raise NotImplementedError
