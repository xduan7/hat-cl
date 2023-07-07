import unittest
from abc import ABC, abstractmethod
from typing import Iterable

import torch.nn as nn

from hat.modules import (
    TaskIndexedBatchNorm1d,
    TaskIndexedBatchNorm2d,
    TaskIndexedBatchNorm3d,
)
from tests.constants import DEBUG, NUM_TASKS

from . import _TestTIModule

BN1D_INPUT_SHAPE = (2,) if DEBUG else (64,)
BN2D_INPUT_SHAPE = (2, 2) if DEBUG else (8, 8)
BN3D_INPUT_SHAPE = (2, 2, 2) if DEBUG else (4, 4, 4)

INPUT_CHANNELS = 6 if DEBUG else 32


INPUT_DIM_TO_BN_CLS: dict[int, type[nn.Module]] = {
    1: nn.BatchNorm1d,
    2: nn.BatchNorm2d,
    3: nn.BatchNorm3d,
}
INPUT_DIM_TO_TI_BN_CLS: dict[int, type[nn.Module]] = {
    1: TaskIndexedBatchNorm1d,
    2: TaskIndexedBatchNorm2d,
    3: TaskIndexedBatchNorm3d,
}


class _TestTIBatchNormndABC(_TestTIModule, ABC):
    @property
    @abstractmethod
    def input_dim(self) -> int:
        raise NotImplementedError

    def get_single_layer_module(self) -> nn.Module:
        _cls = INPUT_DIM_TO_BN_CLS[self.input_dim]
        # noinspection PyArgumentList
        return _cls(num_features=INPUT_CHANNELS)

    def get_single_layer_ti_module(self) -> nn.Module:
        _cls = INPUT_DIM_TO_TI_BN_CLS[self.input_dim]
        # noinspection PyArgumentList
        return _cls(num_features=INPUT_CHANNELS, num_tasks=NUM_TASKS)


class TestTIBatchNorm1d(unittest.TestCase, _TestTIBatchNormndABC):
    @property
    def input_dim(self) -> int:
        return 1

    @property
    def input_shape(self) -> Iterable[int]:
        # noinspection PyRedundantParentheses
        return (INPUT_CHANNELS, *BN1D_INPUT_SHAPE)


class TestTIBatchNorm2d(unittest.TestCase, _TestTIBatchNormndABC):
    @property
    def input_dim(self) -> int:
        return 2

    @property
    def input_shape(self) -> Iterable[int]:
        # noinspection PyRedundantParentheses
        return (INPUT_CHANNELS, *BN2D_INPUT_SHAPE)


class TestTIBatchNorm3d(unittest.TestCase, _TestTIBatchNormndABC):
    @property
    def input_dim(self) -> int:
        return 3

    @property
    def input_shape(self) -> Iterable[int]:
        # noinspection PyRedundantParentheses
        return (INPUT_CHANNELS, *BN3D_INPUT_SHAPE)
