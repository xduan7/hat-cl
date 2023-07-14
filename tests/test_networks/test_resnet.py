import unittest
from typing import Iterable

from tests.constants import DEBUG
from . import _TestNetworkABC

RESNET_S_INPUT_SHAPE = (3, 4, 4) if DEBUG else (3, 17, 17)


class TestResNet18s(unittest.TestCase, _TestNetworkABC):
    @property
    def network_name(self) -> str:
        return "resnet18s"

    @property
    def network_kwargs(self) -> dict:
        return {}

    @property
    def input_shape(self) -> Iterable[int]:
        return RESNET_S_INPUT_SHAPE


class TestResNet34s(unittest.TestCase, _TestNetworkABC):
    @property
    def network_name(self) -> str:
        return "resnet34s"

    @property
    def network_kwargs(self) -> dict:
        return {}

    @property
    def input_shape(self) -> Iterable[int]:
        return RESNET_S_INPUT_SHAPE
