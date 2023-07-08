import torch.nn as nn
from torch import classproperty

from ._base import TaskDependentMixin


class TaskIndexedParameter(TaskDependentMixin):
    """A task-dependent parameter.

    This class is a wrapper of `torch.nn.ParameterList` that supports the
    interface defined in `hat._base.TaskDependentMixin`. The list is indexed
    by task ids, and each element of the list is a `torch.nn.Parameter`
    instance.

    Args:
        num_tasks: The number of tasks.
        *args: The positional arguments to be passed to the constructor of
            `torch.nn.Parameter`.
        **kwargs: The keyword arguments to be passed to the constructor of
            `torch.nn.Parameter`.

    """

    def __init__(self, num_tasks: int, *args, **kwargs):
        super().__init__()
        self._parameters = nn.ParameterList(
            [self.base_class(*args, **kwargs) for _ in range(num_tasks)]
        )

    @property
    def num_tasks(self) -> int:
        """The number of tasks."""
        return len(self._parameters)

    @classproperty
    def base_class(self) -> type:
        """The base class of the task-dependent parameter."""
        return nn.Parameter  # type: ignore

    def __len__(self) -> int:
        """The number of tasks."""
        return self.num_tasks

    def __getitem__(self, task_id: int) -> nn.Parameter:
        """Get the parameter of the given task.

        Args:
            task_id: The ID of the task.

        Returns:
            The parameter of the given task.

        """
        return self._parameters[task_id]
