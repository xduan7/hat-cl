import unittest
from typing import Iterable

# noinspection PyUnresolvedReferences
import hat.networks
from hat.types_ import HATConfig
from tests.constants import NUM_TASKS

from . import RESNET_INPUT_SHAPE, RESNET_S_INPUT_SHAPE, _TestHATNetworkABC

HAT_CONFIG = HATConfig(num_tasks=NUM_TASKS)


class TestHATResNet18(unittest.TestCase, _TestHATNetworkABC):
    @property
    def network_name(self) -> str:
        return "hat_resnet18"

    @property
    def network_kwargs(self) -> dict:
        return {"hat_config": HAT_CONFIG}

    @property
    def input_shape(self) -> Iterable[int]:
        return RESNET_INPUT_SHAPE


class TestHATResNet18s(unittest.TestCase, _TestHATNetworkABC):
    @property
    def network_name(self) -> str:
        return "hat_resnet18s"

    @property
    def network_kwargs(self) -> dict:
        return {"hat_config": HAT_CONFIG}

    @property
    def input_shape(self) -> Iterable[int]:
        return RESNET_S_INPUT_SHAPE


class TestHATResNet34(unittest.TestCase, _TestHATNetworkABC):
    @property
    def network_name(self) -> str:
        return "hat_resnet34"

    @property
    def network_kwargs(self) -> dict:
        return {"hat_config": HAT_CONFIG}

    @property
    def input_shape(self) -> Iterable[int]:
        return RESNET_INPUT_SHAPE


class TestHATResNet34s(unittest.TestCase, _TestHATNetworkABC):
    @property
    def network_name(self) -> str:
        return "hat_resnet34s"

    @property
    def network_kwargs(self) -> dict:
        return {"hat_config": HAT_CONFIG}

    @property
    def input_shape(self) -> Iterable[int]:
        return RESNET_S_INPUT_SHAPE
