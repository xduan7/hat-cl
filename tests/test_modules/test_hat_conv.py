import unittest
from abc import ABC, abstractmethod
from typing import Iterable, Optional

from torch import nn

from hat.modules import (
    HATConv1d,
    HATConv2d,
    HATConv3d,
    TaskIndexedBatchNorm1d,
    TaskIndexedBatchNorm2d,
    TaskIndexedBatchNorm3d,
)
from hat.types_ import HATConfig
from tests.constants import BATCH_SIZE, DEBUG, DROPOUT_RATE, NUM_TASKS

from . import _TestHATModuleABC

CONV1D_INPUT_SHAPE = (2,) if DEBUG else (64,)
CONV2D_INPUT_SHAPE = (2, 2) if DEBUG else (8, 8)
CONV3D_INPUT_SHAPE = (2, 2, 2) if DEBUG else (4, 4, 4)

INPUT_CHANNELS = 6 if DEBUG else 32
OUTPUT_CHANNELS = 3 if DEBUG else 16

CONV1D_KERNEL_SIZE = (1,) if DEBUG else (3,)
CONV2D_KERNEL_SIZE = (1, 1) if DEBUG else (2, 2)
CONV3D_KERNEL_SIZE = (1, 1, 1) if DEBUG else (1, 1, 1)

KERNEL_DIM_TO_KERNEL_SIZE: dict[int, tuple[int, ...]] = {
    1: CONV1D_KERNEL_SIZE,
    2: CONV2D_KERNEL_SIZE,
    3: CONV3D_KERNEL_SIZE,
}
KERNEL_DIM_TO_CONVND_CLS: dict[int, type[nn.Module]] = {
    1: nn.Conv1d,
    2: nn.Conv2d,
    3: nn.Conv3d,
}
KERNEL_DIM_TO_HAT_CONVND_CLS: dict[int, type[nn.Module]] = {
    1: HATConv1d,
    2: HATConv2d,
    3: HATConv3d,
}
KERNEL_DIM_TO_BN_CLS: dict[int, type[nn.Module]] = {
    1: nn.BatchNorm1d,
    2: nn.BatchNorm2d,
    3: nn.BatchNorm3d,
}
KERNEL_DIM_TO_TI_BN_CLS: dict[int, type[nn.Module]] = {
    1: TaskIndexedBatchNorm1d,
    2: TaskIndexedBatchNorm2d,
    3: TaskIndexedBatchNorm3d,
}


class _TestHATConvndABC(_TestHATModuleABC, ABC):
    @property
    @abstractmethod
    def kernel_dim(self) -> int:
        raise NotImplementedError

    def get_single_layer_module(self) -> nn.Module:
        _cls = KERNEL_DIM_TO_CONVND_CLS[self.kernel_dim]
        _kernel_size = KERNEL_DIM_TO_KERNEL_SIZE[self.kernel_dim]
        # noinspection PyArgumentList
        return _cls(
            in_channels=INPUT_CHANNELS,
            out_channels=OUTPUT_CHANNELS,
            kernel_size=_kernel_size,
        )

    def get_single_layer_hat_module(self) -> nn.Module:
        _cls = KERNEL_DIM_TO_HAT_CONVND_CLS[self.kernel_dim]
        _kernel_size = KERNEL_DIM_TO_KERNEL_SIZE[self.kernel_dim]
        _hat_config = HATConfig(num_tasks=NUM_TASKS)
        # noinspection PyArgumentList
        return _cls(
            in_channels=INPUT_CHANNELS,
            out_channels=OUTPUT_CHANNELS,
            kernel_size=_kernel_size,
            hat_config=_hat_config,
        )

    def get_multi_layer_module(
        self,
        normalization: bool = True,
        activation: Optional[nn.Module] = nn.ReLU(),
        dropout: bool = True,
    ) -> nn.Module:
        _cls = KERNEL_DIM_TO_CONVND_CLS[self.kernel_dim]
        _kernel_size = KERNEL_DIM_TO_KERNEL_SIZE[self.kernel_dim]
        _batchnorm = KERNEL_DIM_TO_BN_CLS[self.kernel_dim]
        _activation = activation if activation is not None else nn.Identity()
        _dropout = nn.Dropout(DROPOUT_RATE) if dropout else nn.Identity()
        # noinspection PyArgumentList
        return nn.Sequential(
            _cls(
                in_channels=INPUT_CHANNELS,
                out_channels=INPUT_CHANNELS,
                kernel_size=_kernel_size,
            ),
            _batchnorm(num_features=INPUT_CHANNELS)
            if normalization
            else nn.Identity(),
            _activation,
            _dropout,
            _cls(
                in_channels=INPUT_CHANNELS,
                out_channels=OUTPUT_CHANNELS,
                kernel_size=_kernel_size,
            ),
            _batchnorm(num_features=OUTPUT_CHANNELS)
            if normalization
            else nn.Identity(),
            _activation,
            _dropout,
            _cls(
                in_channels=OUTPUT_CHANNELS,
                out_channels=OUTPUT_CHANNELS,
                kernel_size=_kernel_size,
            ),
        )

    def get_multi_layer_hat_module(
        self,
        normalization: bool = True,
        activation: Optional[nn.Module] = nn.ReLU(),
        dropout: bool = True,
    ) -> nn.Module:
        _cls = KERNEL_DIM_TO_HAT_CONVND_CLS[self.kernel_dim]
        _kernel_size = KERNEL_DIM_TO_KERNEL_SIZE[self.kernel_dim]
        _ti_batchnorm = KERNEL_DIM_TO_TI_BN_CLS[self.kernel_dim]
        _activation = activation if activation is not None else nn.Identity()
        _dropout = nn.Dropout(DROPOUT_RATE) if dropout else nn.Identity()
        _hat_config = HATConfig(num_tasks=NUM_TASKS)
        _bi_config = {"num_tasks": NUM_TASKS}
        # noinspection PyArgumentList
        return nn.Sequential(
            _cls(
                in_channels=INPUT_CHANNELS,
                out_channels=INPUT_CHANNELS,
                kernel_size=_kernel_size,
                hat_config=_hat_config,
            ),
            _ti_batchnorm(num_features=INPUT_CHANNELS, **_bi_config)
            if normalization
            else nn.Identity(),
            _activation,
            _dropout,
            _cls(
                in_channels=INPUT_CHANNELS,
                out_channels=OUTPUT_CHANNELS,
                kernel_size=_kernel_size,
                hat_config=_hat_config,
            ),
            _ti_batchnorm(num_features=OUTPUT_CHANNELS, **_bi_config)
            if normalization
            else nn.Identity(),
            _activation,
            _dropout,
            _cls(
                in_channels=OUTPUT_CHANNELS,
                out_channels=OUTPUT_CHANNELS,
                kernel_size=_kernel_size,
                hat_config=_hat_config,
            ),
        )


class TestHATConv1d(unittest.TestCase, _TestHATConvndABC):
    @property
    def kernel_dim(self) -> int:
        return 1

    @property
    def input_shape(self) -> Iterable[int]:
        # noinspection PyRedundantParentheses
        return (INPUT_CHANNELS, *CONV1D_INPUT_SHAPE)


class TestHATConv2d(unittest.TestCase, _TestHATConvndABC):
    @property
    def kernel_dim(self) -> int:
        return 2

    @property
    def input_shape(self) -> Iterable[int]:
        # noinspection PyRedundantParentheses
        return (INPUT_CHANNELS, *CONV2D_INPUT_SHAPE)


class TestHATConv3d(unittest.TestCase, _TestHATConvndABC):
    @property
    def kernel_dim(self) -> int:
        return 3

    @property
    def input_shape(self) -> Iterable[int]:
        # noinspection PyRedundantParentheses
        return (INPUT_CHANNELS, *CONV3D_INPUT_SHAPE)


if __name__ == "__main__":
    unittest.main()
