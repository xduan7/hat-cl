import unittest
from typing import Iterable

from torch import nn as nn

from hat.modules import TaskIndexedLayerNorm
from tests.constants import DEBUG, NUM_TASKS

from . import _TestTIModule


class TestTILayerNorm(unittest.TestCase, _TestTIModule):
    @property
    def input_shape(self) -> Iterable[int]:
        return (2, 2) if DEBUG else (8, 8)

    def get_single_layer_module(self) -> nn.Module:
        return nn.LayerNorm(normalized_shape=self.input_shape)

    def get_single_layer_ti_module(self) -> nn.Module:
        return TaskIndexedLayerNorm(
            normalized_shape=self.input_shape,
            num_tasks=NUM_TASKS,
        )
