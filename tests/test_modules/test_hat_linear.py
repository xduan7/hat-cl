import unittest
from typing import Optional

from torch import nn

from hat.modules import HATLinear, TaskIndexedBatchNorm1d
from hat.types_ import HATConfig
from tests.constants import DEBUG, DROPOUT_RATE, NUM_TASKS
from tests.conversion import (
    check_from_base_conversion,
    check_to_base_conversion,
)
from tests.task import check_forgetting, check_locking, check_remembering
from tests.utils import set_up

INPUT_DIM = 8 if DEBUG else 128
OUTPUT_DIM = 6 if DEBUG else 32


def _get_single_layer_linear():
    return nn.Linear(
        in_features=INPUT_DIM,
        out_features=OUTPUT_DIM,
    )


def _get_single_layer_hat_linear():
    _hat_config = HATConfig(num_tasks=NUM_TASKS)
    return HATLinear(
        in_features=INPUT_DIM,
        out_features=OUTPUT_DIM,
        hat_config=_hat_config,
    )


def _get_multi_layer_linear(
    normalization: bool = True,
    activation: Optional[nn.Module] = nn.ReLU(),
    dropout: bool = True,
):
    _activation = activation if activation is not None else nn.Identity()
    _dropout = nn.Dropout(DROPOUT_RATE) if dropout else nn.Identity()
    return nn.Sequential(
        nn.Linear(
            in_features=INPUT_DIM,
            out_features=INPUT_DIM,
        ),
        nn.BatchNorm1d(num_features=INPUT_DIM)
        if normalization
        else nn.Identity(),
        _activation,
        _dropout,
        nn.Linear(
            in_features=INPUT_DIM,
            out_features=OUTPUT_DIM,
        ),
        nn.BatchNorm1d(num_features=OUTPUT_DIM)
        if normalization
        else nn.Identity(),
        _activation,
        _dropout,
        nn.Linear(
            in_features=OUTPUT_DIM,
            out_features=OUTPUT_DIM,
        ),
    )


def _get_multi_layer_hat_linear(
    normalization: bool = True,
    activation: Optional[nn.Module] = nn.ReLU(),
    dropout: bool = True,
):
    _hat_config = HATConfig(num_tasks=NUM_TASKS)
    _activation = activation if activation is not None else nn.Identity()
    _dropout = nn.Dropout(DROPOUT_RATE) if dropout else nn.Identity()
    return nn.Sequential(
        HATLinear(
            in_features=INPUT_DIM,
            out_features=INPUT_DIM,
            hat_config=_hat_config,
        ),
        TaskIndexedBatchNorm1d(
            num_features=INPUT_DIM,
            num_tasks=NUM_TASKS,
        )
        if normalization
        else nn.Identity(),
        _activation,
        _dropout,
        HATLinear(
            in_features=INPUT_DIM,
            out_features=OUTPUT_DIM,
            hat_config=_hat_config,
        ),
        TaskIndexedBatchNorm1d(
            num_features=OUTPUT_DIM,
            num_tasks=NUM_TASKS,
        )
        if normalization
        else nn.Identity(),
        _activation,
        _dropout,
        HATLinear(
            in_features=OUTPUT_DIM,
            out_features=OUTPUT_DIM,
            hat_config=_hat_config,
        ),
    )


class TestHATLinear(unittest.TestCase):
    def setUp(self):
        set_up()

    def test_single_layer_to_base_conversion(self):
        check_to_base_conversion(
            test_case=self,
            input_shape=(INPUT_DIM,),
            module=_get_single_layer_hat_linear(),
        )

    def test_single_layer_from_base_conversion(self):
        check_from_base_conversion(
            test_case=self,
            input_shape=(INPUT_DIM,),
            module=_get_single_layer_linear(),
        )

    def test_single_layer_remembering(self):
        check_remembering(
            test_case=self,
            input_shape=(INPUT_DIM,),
            module=_get_single_layer_hat_linear(),
        )

    def test_single_layer_locking(self):
        check_locking(
            test_case=self,
            input_shape=(INPUT_DIM,),
            module=_get_single_layer_hat_linear(),
        )

    def test_single_layer_forgetting(self):
        check_forgetting(
            test_case=self,
            input_shape=(INPUT_DIM,),
            module=_get_single_layer_hat_linear(),
        )

    def test_multi_layer_to_base_conversion(self):
        check_to_base_conversion(
            test_case=self,
            input_shape=(INPUT_DIM,),
            module=_get_multi_layer_hat_linear(
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
            input_shape=(INPUT_DIM,),
            module=_get_multi_layer_linear(),
        )

    def test_multi_layer_remembering(self):
        check_remembering(
            test_case=self,
            input_shape=(INPUT_DIM,),
            module=_get_multi_layer_hat_linear(),
        )

    def test_multi_layer_locking(self):
        check_locking(
            test_case=self,
            input_shape=(INPUT_DIM,),
            module=_get_multi_layer_hat_linear(),
        )

    def test_multi_layer_forgetting(self):
        check_forgetting(
            test_case=self,
            input_shape=(INPUT_DIM,),
            module=_get_multi_layer_hat_linear(),
        )


if __name__ == "__main__":
    unittest.main()
