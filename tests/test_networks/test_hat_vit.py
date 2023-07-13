import unittest
from typing import Iterable

import torch.nn as nn

# noinspection PyUnresolvedReferences
import hat.timm_models
from hat.timm_models.layers.mlp import HATMlp
from hat.timm_models.layers.patch_embed import HATPatchEmbed
from hat.timm_models.vision_transformer import HATAttention, HATBlock
from tests.constants import HAT_CONFIG

from . import DEBUG, _TestHATNetworkABC

VIT_NUM_PATCHES = 4 if DEBUG else 16
VIT_FEATURE_DIM = 6 if DEBUG else 24
VIT_NUM_HEADS = 3 if DEBUG else 12
VIT_IMG_SIZE = 224
VIT_IMG_NUM_CHANNELS = 3
VIT_HIDDEN_SHAPE = (VIT_NUM_PATCHES, VIT_FEATURE_DIM)
VIT_INPUT_SHAPE = (VIT_IMG_NUM_CHANNELS, VIT_IMG_SIZE, VIT_IMG_SIZE)


class TestHATPatchEmbed(unittest.TestCase, _TestHATNetworkABC):
    @property
    def network_name(self) -> str:
        return "hat.timm_models.layers.HATPatchEmbed"

    @property
    def network_kwargs(self) -> dict:
        return {
            "hat_config": HAT_CONFIG,
            "img_size": VIT_IMG_SIZE,
            "in_chans": VIT_IMG_NUM_CHANNELS,
            "embed_dim": VIT_FEATURE_DIM,
        }

    @property
    def network(self) -> nn.Module:
        return HATPatchEmbed(**self.network_kwargs)

    @property
    def input_shape(self) -> Iterable[int]:
        return VIT_INPUT_SHAPE


class TestHATAttention(unittest.TestCase, _TestHATNetworkABC):
    @property
    def network_name(self) -> str:
        return "hat.timm_models.vision_transformer.HATAttention"

    @property
    def network_kwargs(self) -> dict:
        return {
            "dim": VIT_FEATURE_DIM,
            "hat_config": HAT_CONFIG,
            "num_heads": VIT_NUM_HEADS,
        }

    @property
    def network(self) -> nn.Module:
        return HATAttention(**self.network_kwargs)

    @property
    def input_shape(self) -> Iterable[int]:
        return VIT_HIDDEN_SHAPE


class TestHATMlp(unittest.TestCase, _TestHATNetworkABC):
    @property
    def network_name(self) -> str:
        return "hat.timm_models.layers.HATMlp"

    @property
    def network_kwargs(self) -> dict:
        return {
            "in_features": VIT_FEATURE_DIM,
            "hat_config": HAT_CONFIG,
        }

    @property
    def network(self) -> nn.Module:
        return HATMlp(**self.network_kwargs)

    @property
    def input_shape(self) -> Iterable[int]:
        return VIT_HIDDEN_SHAPE


class TestHATBlock(unittest.TestCase, _TestHATNetworkABC):
    @property
    def network_name(self) -> str:
        return "hat.timm_models.vision_transformer.HATBlock"

    @property
    def network_kwargs(self) -> dict:
        return {
            "dim": VIT_FEATURE_DIM,
            "hat_config": HAT_CONFIG,
            "num_heads": VIT_NUM_HEADS,
        }

    @property
    def network(self) -> nn.Module:
        return HATBlock(**self.network_kwargs)

    @property
    def input_shape(self) -> Iterable[int]:
        return VIT_HIDDEN_SHAPE


class TestHATViTTinyPatch16x224(unittest.TestCase, _TestHATNetworkABC):
    @property
    def network_name(self) -> str:
        return "hat_vit_tiny_patch16_224"

    @property
    def network_kwargs(self) -> dict:
        return {"hat_config": HAT_CONFIG}

    @property
    def input_shape(self) -> Iterable[int]:
        return VIT_INPUT_SHAPE
