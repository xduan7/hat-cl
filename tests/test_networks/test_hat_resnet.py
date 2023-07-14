import unittest
from typing import Iterable

# noinspection PyUnresolvedReferences
import hat.timm_models
from hat.types_ import HATConfig
from tests.constants import DEBUG, NUM_TASKS

from . import _TestHATNetworkABC

HAT_CONFIG = HATConfig(num_tasks=NUM_TASKS)

# The shape of the input are the minimum size that can produces features of
# size (B, C, 2, 2) for different models, in order to save time.
HAT_RESNET_INPUT_SHAPE = (3, 4, 4) if DEBUG else (3, 33, 33)
HAT_RESNET_S_INPUT_SHAPE = (3, 4, 4) if DEBUG else (3, 17, 17)


class TestHATResNet18(unittest.TestCase, _TestHATNetworkABC):
    @property
    def network_name(self) -> str:
        return "hat_resnet18"

    @property
    def non_hat_network_name(self) -> str:
        return "resnet18"

    @property
    def network_kwargs(self) -> dict:
        return {"hat_config": HAT_CONFIG}

    @property
    def input_shape(self) -> Iterable[int]:
        return HAT_RESNET_INPUT_SHAPE


class TestHATResNet18s(unittest.TestCase, _TestHATNetworkABC):
    @property
    def network_name(self) -> str:
        return "hat_resnet18s"

    @property
    def non_hat_network_name(self) -> str:
        raise "resnet18s"

    @property
    def network_kwargs(self) -> dict:
        return {"hat_config": HAT_CONFIG}

    @property
    def input_shape(self) -> Iterable[int]:
        return HAT_RESNET_S_INPUT_SHAPE


class TestHATResNet34(unittest.TestCase, _TestHATNetworkABC):
    @property
    def network_name(self) -> str:
        return "hat_resnet34"

    @property
    def non_hat_network_name(self) -> str:
        return "resnet34"

    @property
    def network_kwargs(self) -> dict:
        return {"hat_config": HAT_CONFIG}

    @property
    def input_shape(self) -> Iterable[int]:
        return HAT_RESNET_INPUT_SHAPE


class TestHATResNet34s(unittest.TestCase, _TestHATNetworkABC):
    @property
    def network_name(self) -> str:
        return "hat_resnet34s"

    @property
    def non_hat_network_name(self) -> str:
        raise "resnet34s"

    @property
    def network_kwargs(self) -> dict:
        return {"hat_config": HAT_CONFIG}

    @property
    def input_shape(self) -> Iterable[int]:
        return HAT_RESNET_S_INPUT_SHAPE
