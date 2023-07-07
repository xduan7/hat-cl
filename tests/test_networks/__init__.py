from abc import ABC, abstractmethod
from typing import Iterable

import timm
import torch.nn as nn

from tests.constants import DEBUG
from tests.task import (
    check_forgetting,
    check_forward,
    check_fully_task_dependent,
    check_locking,
    check_pruning,
    check_remembering,
)
from tests.utils import set_up

# The shape of the input are the minimum size that can produces features of
# size (B, C, 2, 2) for different models, in order to save time.
RESNET_INPUT_SHAPE = (3, 4, 4) if DEBUG else (3, 33, 33)
RESNET_S_INPUT_SHAPE = (3, 4, 4) if DEBUG else (3, 17, 17)


class _TestNetworkABC(ABC):
    # noinspection PyPep8Naming,PyMethodMayBeStatic
    def setUp(self):
        set_up()

    @property
    @abstractmethod
    def network_name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def network_kwargs(self) -> dict:
        raise NotImplementedError

    @property
    @abstractmethod
    def input_shape(self) -> Iterable[int]:
        raise NotImplementedError

    @property
    def network(self) -> nn.Module:
        return timm.create_model(
            self.network_name,
            **self.network_kwargs,
        )

    @property
    def is_task_dependent(self) -> bool:
        return False

    def test_forward(self):
        check_forward(
            test_case=self,
            input_shape=self.input_shape,
            module=self.network,
            is_task_dependent=self.is_task_dependent,
        )


class _TestHATNetworkABC(_TestNetworkABC, ABC):
    @property
    def is_task_dependent(self) -> bool:
        return True

    def test_fully_task_dependent(self):
        check_fully_task_dependent(
            test_case=self,
            module=self.network,
        )

    def test_remembering(self):
        check_remembering(
            test_case=self,
            input_shape=self.input_shape,
            module=self.network,
        )

    def test_locking(self):
        check_locking(
            test_case=self,
            input_shape=self.input_shape,
            module=self.network,
        )

    def test_forgetting(self):
        check_forgetting(
            test_case=self,
            input_shape=self.input_shape,
            module=self.network,
        )

    def test_pruning(self):
        check_pruning(
            test_case=self,
            input_shape=self.input_shape,
            module=self.network,
        )
