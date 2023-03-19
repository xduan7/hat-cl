import unittest
from typing import Iterable, Optional

import torch.nn as nn

from hat.modules import HATLinear, TaskIndexedBatchNorm1d
from hat.types_ import HATConfig
from tests.constants import DEBUG, DROPOUT_RATE, NUM_TASKS

from . import _TestHATModule

INPUT_DIM = 8 if DEBUG else 128
OUTPUT_DIM = 6 if DEBUG else 32


class TestHATLinear(unittest.TestCase, _TestHATModule):
    @property
    def input_shape(self) -> Iterable[int]:
        # noinspection PyRedundantParentheses
        return (INPUT_DIM,)

    def get_single_layer_module(self) -> nn.Module:
        return nn.Linear(
            in_features=INPUT_DIM,
            out_features=OUTPUT_DIM,
        )

    def get_single_layer_hat_module(self) -> nn.Module:
        _hat_config = HATConfig(num_tasks=NUM_TASKS)
        return HATLinear(
            in_features=INPUT_DIM,
            out_features=OUTPUT_DIM,
            hat_config=_hat_config,
        )

    def get_multi_layer_module(
        self,
        normalization: bool = True,
        activation: Optional[nn.Module] = nn.ReLU(),
        dropout: bool = True,
    ) -> nn.Module:
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

    def get_multi_layer_hat_module(
        self,
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


if __name__ == "__main__":
    unittest.main()
