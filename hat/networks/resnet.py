from typing import Optional

import torch
import torch.nn as nn
from timm.models import register_model
from timm.models.helpers import build_model_with_cfg, checkpoint_seq

# noinspection PyProtectedMember
from timm.models.resnest import _cfg
from timm.models.resnet import BasicBlock, ResNet

# noinspection PyProtectedMember
from hat.modules._base import HATPayloadCarrierMixin
from hat.payload import HATPayload
from hat.types_ import HATConfig

from .utils import convert_children_to_task_dependent_modules

__all__ = ["HATResNet"]


default_cfgs = {
    "hat_resnet18": _cfg(),
    "hat_resnet18s": _cfg(
        input_size=(3, 32, 32),
    ),
}


class HATBasicBlock(BasicBlock, HATPayloadCarrierMixin):
    """Basic ResNet block with HAT modules.

    This class is a modified version of `timm.models.resnet.BasicBlock`,
    whose submodules are converted to their task-dependent counterparts.
    See `hat.models.utils.convert_children_to_task_dependent_modules` for
    more details.

    """

    # Do not change the orderings of the arguments as the `make_blocks`
    # function specifically depends on the order.
    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        cardinality=1,
        base_width=64,
        reduce_first=1,
        dilation=1,
        first_dilation=None,
        act_layer=nn.ReLU,
        norm_layer=nn.BatchNorm2d,
        attn_layer=None,
        aa_layer=None,
        drop_block=None,
        drop_path=None,
        # Not actually optional. But the `make_blocks` function call in
        # `timm` was based on the argument order. So we have to make it
        # compatible by adding this argument at the end.
        hat_config: Optional[HATConfig] = None,
    ):
        assert hat_config is not None
        super().__init__(
            inplanes=inplanes,
            planes=planes,
            stride=stride,
            downsample=downsample,
            cardinality=cardinality,
            base_width=base_width,
            reduce_first=reduce_first,
            dilation=dilation,
            first_dilation=first_dilation,
            act_layer=act_layer,
            norm_layer=norm_layer,
            attn_layer=attn_layer,
            aa_layer=aa_layer,
            drop_block=drop_block,
            drop_path=drop_path,
        )
        # Replace the all layers with their task-dependent versions.
        convert_children_to_task_dependent_modules(self, **hat_config)

    def zero_init_last(self):
        """Initialize the last batch norm layer with zeros.

        Used when initializing the ResNet with `zero_init_last` set to True.

        """
        for __bn in self.bn2:
            nn.init.zeros_(__bn.weight)

    def forward(self, pld: HATPayload) -> HATPayload:
        """Forward pass of the block.

        Note that the output, which is the sum of the shortcut and the
        processed input, is masked by the last masker (i.e.
        `self.conv2.masker`).

        """
        shortcut = pld

        pld = pld.forward_by(self.conv1)
        pld = pld.forward_by(self.bn1)
        pld = pld.forward_by(self.drop_block)
        pld = pld.forward_by(self.act1)
        pld = pld.forward_by(self.aa)

        pld = pld.forward_by(self.conv2)
        pld = pld.forward_by(self.bn2)

        if self.se is not None:
            pld = pld.forward_by(self.se)

        if self.drop_path is not None:
            pld = pld.forward_by(self.drop_path)

        if self.downsample is not None:
            shortcut = shortcut.forward_by(
                self.downsample,
                use_masked_data=False,
            )

        assert pld.unmasked_data is not None
        pld = HATPayload(
            data=pld.unmasked_data + shortcut.data,
            masker=pld.masker,
            task_id=pld.task_id,
            mask_scale=pld.mask_scale,
            locked_task_ids=pld.locked_task_ids,
            prev_maskers=pld.prev_maskers,
        )
        pld = pld.forward_by(self.act2)
        return pld


# TODO: HardAttentionBottleneck for ResNet variants.


# noinspection PyPep8Naming
class ResNet_(ResNet):
    """ResNet from `timm` with an additional option that reduces the kernel
    size of the first conv layer to 3 and stride to 1, which enables the
    model to process images with a smaller resolution (e.g. 32x32).

    """

    def __init__(
        self,
        layers,
        block=HATBasicBlock,
        num_classes=1000,
        in_chans=3,
        output_stride=32,
        global_pool="avg",
        cardinality=1,
        base_width=64,
        stem_width=64,
        stem_type="",
        replace_stem_pool=False,
        block_reduce_first=1,
        down_kernel_size=1,
        avg_down=False,
        act_layer=nn.ReLU,
        norm_layer=nn.BatchNorm2d,
        aa_layer=None,
        drop_rate=0.0,
        drop_path_rate=0.0,
        drop_block_rate=0.0,
        zero_init_last=True,
        block_args=None,
        reduced_conv1_kernel=False,
    ):
        super().__init__(
            block=block,
            layers=layers,
            num_classes=num_classes,
            in_chans=in_chans,
            output_stride=output_stride,
            global_pool=global_pool,
            cardinality=cardinality,
            base_width=base_width,
            stem_width=stem_width,
            stem_type=stem_type,
            replace_stem_pool=replace_stem_pool,
            block_reduce_first=block_reduce_first,
            down_kernel_size=down_kernel_size,
            avg_down=avg_down,
            act_layer=act_layer,
            norm_layer=norm_layer,
            aa_layer=aa_layer,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            drop_block_rate=drop_block_rate,
            zero_init_last=zero_init_last,
            block_args=block_args,
        )

        if reduced_conv1_kernel:
            if "deep" in stem_type:
                # In this case, `conv1` is an `nn.Sequential` object,
                # which requires further inspection.
                raise NotImplementedError(
                    f"`small_img` is currently not supported "
                    f"for ResNet with stem_type {stem_type}."
                )
            else:
                self.conv1 = nn.Conv2d(
                    in_chans,
                    self.conv1.out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                )


class HATResNet(ResNet_, HATPayloadCarrierMixin):
    """HAT- ResNet / ResNeXt / SE-ResNeXt / SE-Net

    This class is a modified version of `timm.models.ResNet` that supports
    HAT (hard attention to the task). See `timm.models.ResNet` for more
    details.

    """

    def __init__(
        self,
        layers,
        hat_config: HATConfig,
        block=HATBasicBlock,
        num_classes=1000,
        in_chans=3,
        output_stride=32,
        global_pool="avg",
        cardinality=1,
        base_width=64,
        stem_width=64,
        stem_type="",
        replace_stem_pool=False,
        block_reduce_first=1,
        down_kernel_size=1,
        avg_down=False,
        act_layer=nn.ReLU,
        norm_layer=nn.BatchNorm2d,
        aa_layer=None,
        drop_rate=0.0,
        drop_path_rate=0.0,
        drop_block_rate=0.0,
        zero_init_last=True,
        block_args=None,
        reduced_conv1_kernel=False,
    ):
        # Must prepare the `block_args` before calling the super constructor
        # so that the `block` can be correctly initialized.
        block_args = block_args or {}
        block_args["hat_config"] = hat_config
        super().__init__(
            block=block,
            layers=layers,
            num_classes=num_classes,
            in_chans=in_chans,
            output_stride=output_stride,
            global_pool=global_pool,
            cardinality=cardinality,
            base_width=base_width,
            stem_width=stem_width,
            stem_type=stem_type,
            replace_stem_pool=replace_stem_pool,
            block_reduce_first=block_reduce_first,
            down_kernel_size=down_kernel_size,
            avg_down=avg_down,
            act_layer=act_layer,
            norm_layer=norm_layer,
            aa_layer=aa_layer,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            drop_block_rate=drop_block_rate,
            zero_init_last=zero_init_last,
            block_args=block_args,
            reduced_conv1_kernel=reduced_conv1_kernel,
        )
        # Replace the all layers with their task-dependent versions.
        convert_children_to_task_dependent_modules(self, **hat_config)
        # Dropout layer before the final fc layer.
        self.drop_fc = (
            nn.Dropout(p=self.drop_rate) if drop_rate > 0.0 else nn.Identity()
        )

    def forward_features(self, pld: HATPayload) -> HATPayload:
        """Extract the features from the input payload."""
        pld = pld.forward_by(self.conv1)
        pld = pld.forward_by(self.bn1)
        pld = pld.forward_by(self.act1)
        pld = pld.forward_by(self.maxpool)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            # This is not tested with TorchScript
            pld = checkpoint_seq(
                [self.layer1, self.layer2, self.layer3, self.layer4],
                pld,
                flatten=True,
            )
        else:
            pld = self.layer1(pld)
            pld = self.layer2(pld)
            pld = self.layer3(pld)
            pld = self.layer4(pld)
        return pld

    def forward_head(
        self,
        pld: HATPayload,
        pre_logits: bool = False,
    ) -> HATPayload:
        """Produce the logits from the feature payload."""
        pld = pld.forward_by(self.global_pool)
        pld = pld.forward_by(self.drop_fc)
        return pld if pre_logits else pld.forward_by(self.fc)

    def forward(self, pld: HATPayload) -> HATPayload:
        """Generate the logits from the input payload."""
        pld = self.forward_features(pld)
        pld = self.forward_head(pld)
        return pld


def _create_resnet(variant, pretrained=False, **kwargs):
    """Helper method to create a ResNet model with a given variant"""
    return build_model_with_cfg(ResNet_, variant, pretrained, **kwargs)


def _create_resnet_hat(variant, pretrained=False, **kwargs):
    """Helper method to create a HAT-ResNet model with a given variant"""
    return build_model_with_cfg(HATResNet, variant, pretrained, **kwargs)


@register_model
def hat_resnet18(pretrained=False, **kwargs):
    """Constructs a HAT-ResNet-18 model."""
    model_kwargs = dict(
        layers=[2, 2, 2, 2],
        **kwargs,
    )
    return _create_resnet_hat(
        variant="hat_resnet18",
        pretrained=pretrained,
        **model_kwargs,
    )


@register_model
def hat_resnet18s(pretrained=False, **kwargs):
    """Constructs a HAT-ResNet-18-S model."""
    model_kwargs = dict(
        layers=[2, 2, 2, 2],
        reduced_conv1_kernel=True,
        **kwargs,
    )
    return _create_resnet_hat(
        variant="hat_resnet18s",
        pretrained=pretrained,
        **model_kwargs,
    )
