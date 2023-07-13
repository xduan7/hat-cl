from typing import Optional

import torch.nn as nn
from timm.models.layers.helpers import to_2tuple

from hat import HATConfig, HATPayload
from hat.modules import HATLinear

# noinspection PyProtectedMember
from hat.modules._base import HATPayloadCarrierMixin


class HATMlp(HATPayloadCarrierMixin):
    """HAT-MLP as used in HAT Vision Transformer, and related networks.

    Please refer to the original implementation for more details:
    :class:`timm.models.layers.mlp`.

    """

    def __init__(
        self,
        in_features: int,
        hat_config: HATConfig,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        bias: bool = True,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        biases = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = HATLinear(
            in_features=in_features,
            out_features=hidden_features,
            hat_config=hat_config,
            bias=biases[0],
        )
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = HATLinear(
            in_features=hidden_features,
            out_features=out_features,
            hat_config=hat_config,
            bias=biases[1],
        )
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, pld: HATPayload) -> HATPayload:
        """Forward the payload through MLP players."""
        pld = self.fc1(pld)
        pld = pld.forward_by(self.act)
        pld = pld.forward_by(self.drop1)
        pld = self.fc2(pld)
        pld = pld.forward_by(self.drop2)
        return pld
