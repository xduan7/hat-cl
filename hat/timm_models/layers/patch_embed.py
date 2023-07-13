from typing import Optional

import torch.nn as nn
from timm.models.layers.helpers import to_2tuple

# noinspection PyProtectedMember
from timm.models.layers.trace_utils import _assert

from hat import HATConfig, HATPayload
from hat.modules import HATConv2d

# noinspection PyProtectedMember
from hat.modules._base import HATPayloadCarrierMixin


class HATPatchEmbed(HATPayloadCarrierMixin):
    """HAT 2D image to patch embedding for vision transformers and related
    networks.

    Please refer to the original implementation for more details:
    :class:`timm.models.layers.patch_embed`.

    """

    def __init__(
        self,
        hat_config: HATConfig,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[nn.Module] = None,
        flatten: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.grid_size = (
            self.img_size[0] // self.patch_size[0],
            self.img_size[1] // self.patch_size[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = HATConv2d(
            in_channels=in_chans,
            out_channels=embed_dim,
            hat_config=hat_config,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=bias,
        )
        if self.flatten:
            self.proj.masker.mask_dim = 2
        if norm_layer:
            if not isinstance(norm_layer, HATPayloadCarrierMixin):
                raise ValueError(
                    "`norm_layer` must be a `HATPayloadCarrierMixin` "
                    "instance, got {type(norm_layer)}."
                )
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = nn.Identity()

    def forward(self, pld: HATPayload) -> HATPayload:
        """Forward the payload through patch embedding layer."""
        B, C, H, W = pld.data.shape
        _assert(
            H == self.img_size[0],
            f"Input image height ({H}) "
            f"doesn't match model ({self.img_size[0]}).",
        )
        _assert(
            W == self.img_size[1],
            f"Input image width ({W}) "
            f"doesn't match model ({self.img_size[1]}).",
        )
        pld = self.proj(pld)
        if self.flatten:
            pld = HATPayload(
                # Change the shape from (B, C, H, W) to (B, H * W, C)
                data=pld.original_data.flatten(2).transpose(1, 2),
                **pld.to_dict(include_data=False),
            )
        pld = self.norm(pld)
        return pld
