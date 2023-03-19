from abc import ABC, abstractmethod
from typing import Iterable, Optional

import torch.nn as nn

from tests.consistency import (
    check_evl_mask_scale_consistency,
    check_trn_evl_consistency,
)
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


class _TestHATModule(ABC):
    # noinspection PyPep8Naming,PyMethodMayBeStatic
    def setUp(self):
        set_up()

    @property
    @abstractmethod
    def input_shape(self) -> Iterable[int]:
        raise NotImplementedError

    @abstractmethod
    def get_single_layer_module(self) -> nn.Module:
        raise NotImplementedError

    @abstractmethod
    def get_single_layer_hat_module(self) -> nn.Module:
        raise NotImplementedError

    @abstractmethod
    def get_multi_layer_module(
        self,
        normalization: bool = True,
        activation: Optional[nn.Module] = nn.ReLU(),
        dropout: bool = True,
    ) -> nn.Module:
        raise NotImplementedError

    @abstractmethod
    def get_multi_layer_hat_module(
        self,
        normalization: bool = True,
        activation: Optional[nn.Module] = nn.ReLU(),
        dropout: bool = True,
    ) -> nn.Module:
        raise NotImplementedError

    def test_single_layer_to_base_conversion(self):
        check_to_base_conversion(
            test_case=self,
            input_shape=self.input_shape,
            module=self.get_single_layer_hat_module(),
        )

    def test_single_layer_from_base_conversion(self):
        check_from_base_conversion(
            test_case=self,
            input_shape=self.input_shape,
            module=self.get_single_layer_module(),
        )

    def test_single_layer_remembering(self):
        check_remembering(
            test_case=self,
            input_shape=self.input_shape,
            module=self.get_single_layer_hat_module(),
        )

    def test_single_layer_locking(self):
        check_locking(
            test_case=self,
            input_shape=self.input_shape,
            module=self.get_single_layer_hat_module(),
        )

    def test_single_layer_forgetting(self):
        check_forgetting(
            test_case=self,
            input_shape=self.input_shape,
            module=self.get_single_layer_hat_module(),
        )

    def test_single_layer_pruning(self):
        check_pruning(
            test_case=self,
            input_shape=self.input_shape,
            module=self.get_single_layer_hat_module(),
        )

    def test_single_layer_trn_evl_consistency(self):
        check_trn_evl_consistency(
            test_case=self,
            input_shape=self.input_shape,
            module=self.get_single_layer_hat_module(),
        )

    def check_single_layer_evl_mask_scale_consistency(self):
        check_evl_mask_scale_consistency(
            test_case=self,
            input_shape=self.input_shape,
            module=self.get_single_layer_hat_module(),
        )

    def test_multi_layer_to_base_conversion(self):
        check_to_base_conversion(
            test_case=self,
            input_shape=self.input_shape,
            module=self.get_multi_layer_hat_module(
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
            input_shape=self.input_shape,
            module=self.get_multi_layer_module(),
        )

    def test_multi_layer_remembering(self):
        check_remembering(
            test_case=self,
            input_shape=self.input_shape,
            module=self.get_multi_layer_hat_module(),
        )

    def test_multi_layer_locking(self):
        check_locking(
            test_case=self,
            input_shape=self.input_shape,
            module=self.get_multi_layer_hat_module(),
        )

    def test_multi_layer_forgetting(self):
        check_forgetting(
            test_case=self,
            input_shape=self.input_shape,
            module=self.get_multi_layer_hat_module(),
        )

    def test_multi_layer_pruning(self):
        check_pruning(
            test_case=self,
            input_shape=self.input_shape,
            module=self.get_multi_layer_hat_module(),
        )

    def test_multi_layer_trn_evl_consistency(self):
        check_trn_evl_consistency(
            test_case=self,
            input_shape=self.input_shape,
            module=self.get_multi_layer_hat_module(),
        )

    def test_multi_layer_evl_mask_scale_consistency(self):
        check_evl_mask_scale_consistency(
            test_case=self,
            input_shape=self.input_shape,
            module=self.get_multi_layer_hat_module(),
        )
