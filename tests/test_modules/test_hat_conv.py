import unittest
from typing import Optional, Union

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
from tests.constants import DEBUG, DROPOUT_RATE, NUM_TASKS
from tests.conversion import (
    check_from_base_conversion,
    check_to_base_conversion,
)
from tests.task import (
    check_forgetting,
    check_locking,
    check_pruning,
    check_remembering,
)
from tests.utils import set_up

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


def _get_single_layer_conv_nd(kernel_dim: int):
    _cls = KERNEL_DIM_TO_CONVND_CLS[kernel_dim]
    _kernel_size = KERNEL_DIM_TO_KERNEL_SIZE[kernel_dim]
    # noinspection PyArgumentList
    return _cls(
        in_channels=INPUT_CHANNELS,
        out_channels=OUTPUT_CHANNELS,
        kernel_size=_kernel_size,
    )


def _get_single_layer_hat_conv_nd(kernel_dim: int):
    _cls = KERNEL_DIM_TO_HAT_CONVND_CLS[kernel_dim]
    _kernel_size = KERNEL_DIM_TO_KERNEL_SIZE[kernel_dim]
    _hat_config = HATConfig(num_tasks=NUM_TASKS)
    # noinspection PyArgumentList
    return _cls(
        in_channels=INPUT_CHANNELS,
        out_channels=OUTPUT_CHANNELS,
        kernel_size=_kernel_size,
        hat_config=_hat_config,
    )


def _get_multi_layer_conv_nd(
    kernel_dim: int,
    normalization: bool = True,
    activation: Optional[nn.Module] = nn.ReLU(),
    dropout: bool = True,
):
    _cls = KERNEL_DIM_TO_CONVND_CLS[kernel_dim]
    _kernel_size = KERNEL_DIM_TO_KERNEL_SIZE[kernel_dim]
    _batchnorm = KERNEL_DIM_TO_BN_CLS[kernel_dim]
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


def _get_multi_layer_hat_conv_nd(
    kernel_dim: int,
    normalization: bool = True,
    activation: Optional[nn.Module] = nn.ReLU(),
    dropout: bool = True,
):
    _cls = KERNEL_DIM_TO_HAT_CONVND_CLS[kernel_dim]
    _kernel_size = KERNEL_DIM_TO_KERNEL_SIZE[kernel_dim]
    _ti_batchnorm = KERNEL_DIM_TO_TI_BN_CLS[kernel_dim]
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


class _TestHATConvnd:
    _kernel_dim: int
    _input_shape: Union[int, tuple[int, ...]]

    # noinspection PyMethodMayBeStatic, PyPep8Naming
    def SetUp(self):
        set_up()

    def test_single_layer_to_base_conversion(self):
        check_to_base_conversion(
            test_case=self,
            input_shape=self._input_shape,
            module=_get_single_layer_hat_conv_nd(self._kernel_dim),
        )

    def test_single_layer_from_base_conversion(self):
        check_from_base_conversion(
            test_case=self,
            input_shape=self._input_shape,
            module=_get_single_layer_conv_nd(self._kernel_dim),
        )

    def test_single_layer_remembering(self):
        check_remembering(
            test_case=self,
            input_shape=self._input_shape,
            module=_get_single_layer_hat_conv_nd(self._kernel_dim),
        )

    def test_single_layer_locking(self):
        check_locking(
            test_case=self,
            input_shape=self._input_shape,
            module=_get_single_layer_hat_conv_nd(self._kernel_dim),
        )

    def test_single_layer_forgetting(self):
        check_locking(
            test_case=self,
            input_shape=self._input_shape,
            module=_get_single_layer_hat_conv_nd(self._kernel_dim),
        )

    def test_single_layer_pruning(self):
        check_pruning(
            test_case=self,
            input_shape=self._input_shape,
            module=_get_single_layer_hat_conv_nd(self._kernel_dim),
        )

    def test_multi_layer_to_base_conversion(self):
        check_to_base_conversion(
            test_case=self,
            input_shape=self._input_shape,
            module=_get_multi_layer_hat_conv_nd(
                kernel_dim=self._kernel_dim,
                # Normalization must be disabled for this test.
                # The test will go through all task IDs, and during the
                # conversion, the order of execution will change. For example,
                # a `HATLinear` module followed by a `TaskIndexedBatchNorm1d`
                # module will be converted to a `nn.Linear` module followed by
                # mask, then a `nn.BatchNorm1d` module. In the first case
                # scenario, the mask will be applied after the batch norm,
                # which is the opposite of the second case scenario.
                normalization=False,
            ),
        )

    def test_multi_layer_from_base_conversion(self):
        check_from_base_conversion(
            test_case=self,
            input_shape=self._input_shape,
            module=_get_multi_layer_conv_nd(
                kernel_dim=self._kernel_dim,
            ),
        )

    def test_multi_layer_remembering(self):
        check_remembering(
            test_case=self,
            input_shape=self._input_shape,
            module=_get_multi_layer_hat_conv_nd(
                kernel_dim=self._kernel_dim,
            ),
        )

    def test_multi_layer_locking(self):
        check_locking(
            test_case=self,
            input_shape=self._input_shape,
            module=_get_multi_layer_hat_conv_nd(
                kernel_dim=self._kernel_dim,
            ),
        )

    def test_multi_layer_forgetting(self):
        check_forgetting(
            test_case=self,
            input_shape=self._input_shape,
            module=_get_multi_layer_hat_conv_nd(
                kernel_dim=self._kernel_dim,
            ),
        )

    def test_multi_layer_pruning(self):
        check_pruning(
            test_case=self,
            input_shape=self._input_shape,
            module=_get_multi_layer_hat_conv_nd(
                kernel_dim=self._kernel_dim,
            ),
        )


class TestHATConv1d(unittest.TestCase, _TestHATConvnd):
    _kernel_dim = 1
    _input_shape = (INPUT_CHANNELS, *CONV1D_INPUT_SHAPE)


class TestHATConv2d(unittest.TestCase, _TestHATConvnd):
    _kernel_dim = 2
    _input_shape = (INPUT_CHANNELS, *CONV2D_INPUT_SHAPE)


class TestHATConv3d(unittest.TestCase, _TestHATConvnd):
    _kernel_dim = 3
    _input_shape = (INPUT_CHANNELS, *CONV3D_INPUT_SHAPE)


if __name__ == "__main__":
    unittest.main()
