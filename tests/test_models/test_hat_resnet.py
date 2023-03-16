import unittest

import timm

# noinspection PyUnresolvedReferences
import hat.models
from hat.types_ import HATConfig
from tests.constants import DEBUG, DROPOUT_RATE, NUM_TASKS
from tests.task import (
    check_forgetting,
    check_fully_task_dependent,
    check_locking,
    check_pruning,
    check_remembering,
)
from tests.utils import set_up

HAT_CONFIG = HATConfig(num_tasks=NUM_TASKS)


# The shape of the input are the minimum size that can produces features of
# size (B, C, 2, 2) for different models, in order to save time.
RESNET_INPUT_SHAPE = (3, 4, 4) if DEBUG else (3, 33, 33)
RESNET_S_INPUT_SHAPE = (3, 4, 4) if DEBUG else (3, 17, 17)


class TestHATResNet(unittest.TestCase):
    def setUp(self):
        set_up()

    def test_hat_resnet18_fully_task_dependent(self):
        check_fully_task_dependent(
            test_case=self,
            module=timm.create_model("hat_resnet18", hat_config=HAT_CONFIG),
        )

    def test_hat_resnet18_remembering(self):
        check_remembering(
            test_case=self,
            input_shape=RESNET_INPUT_SHAPE,
            module=timm.create_model("hat_resnet18", hat_config=HAT_CONFIG),
        )

    def test_hat_resnet18_locking(self):
        check_locking(
            test_case=self,
            input_shape=RESNET_INPUT_SHAPE,
            module=timm.create_model("hat_resnet18", hat_config=HAT_CONFIG),
        )

    def test_hat_resnet18_forgetting(self):
        check_forgetting(
            test_case=self,
            input_shape=RESNET_INPUT_SHAPE,
            module=timm.create_model("hat_resnet18", hat_config=HAT_CONFIG),
        )

    def test_hat_resnet18_pruning(self):
        check_pruning(
            test_case=self,
            input_shape=RESNET_INPUT_SHAPE,
            module=timm.create_model("hat_resnet18", hat_config=HAT_CONFIG),
        )

    def test_hat_resnet18s_fully_task_dependent(self):
        check_fully_task_dependent(
            test_case=self,
            module=timm.create_model("hat_resnet18s", hat_config=HAT_CONFIG),
        )

    def test_hat_resnet18s_remembering(self):
        check_remembering(
            test_case=self,
            input_shape=RESNET_S_INPUT_SHAPE,
            module=timm.create_model("hat_resnet18s", hat_config=HAT_CONFIG),
        )

    def test_hat_resnet18s_locking(self):
        check_locking(
            test_case=self,
            input_shape=RESNET_S_INPUT_SHAPE,
            module=timm.create_model("hat_resnet18s", hat_config=HAT_CONFIG),
        )

    def test_hat_resnet18s_forgetting(self):
        check_forgetting(
            test_case=self,
            input_shape=RESNET_S_INPUT_SHAPE,
            module=timm.create_model("hat_resnet18s", hat_config=HAT_CONFIG),
        )

    def test_hat_resnet18s_pruning(self):
        check_pruning(
            test_case=self,
            input_shape=RESNET_INPUT_SHAPE,
            module=timm.create_model("hat_resnet18s", hat_config=HAT_CONFIG),
        )
