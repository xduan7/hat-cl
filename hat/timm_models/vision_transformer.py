from __future__ import annotations

import math
from functools import partial
from typing import Any, Optional

import torch
import torch.nn as nn
from timm.models import register_model
from timm.models.helpers import (
    adapt_input_conv,
    build_model_with_cfg,
    checkpoint_seq,
    named_apply,
    resolve_pretrained_cfg,
)
from timm.models.layers.drop import DropPath
from timm.models.layers.weight_init import trunc_normal_
from timm.models.vision_transformer import (
    LayerScale,
    VisionTransformer,
    checkpoint_filter_fn,
)
from timm.models.vision_transformer import default_cfgs as timm_default_cfgs
from timm.models.vision_transformer import (
    init_weights_vit_jax,
    init_weights_vit_timm,
    resize_pos_embed,
)
from torch import classproperty

from hat import TaskIndexedParameter
from hat.modules import HATLinear, TaskIndexedLayerNorm

# noinspection PyProtectedMember
from hat.modules._base import HATPayloadCarrierMixin, TaskIndexedModuleListABC
from hat.payload import HATPayload
from hat.types_ import ForgetResult, HATConfig

from .layers import HATMlp, HATPatchEmbed

default_cfgs = {
    "hat_vit_tiny_patch16_224": timm_default_cfgs["vit_tiny_patch16_224"],
    "hat_vit_tiny_patch16_224_in21k": timm_default_cfgs[
        "vit_tiny_patch16_224_in21k"
    ],
}


class HATAttention(HATPayloadCarrierMixin):
    """HAT attention module for Vision Transformer.

    Please refer to the original implementation for more details:
    :class:`timm.models.vision_transformer.Attention`.

    Args:
        dim: Number of input dimensions/channels.
        hat_config: The HAT configuration.
        num_heads: Number of attention heads.
        qkv_bias: Whether to add bias to the query, key, and value
            projections.
        attn_drop: Dropout rate for the attention weights.
        proj_drop: Dropout rate for the output.

    """

    def __init__(
        self,
        dim: int,
        hat_config: HATConfig,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.hat_config = hat_config

        # Name `qkv_*` to find them with the key word `qkv`.
        self.qkv_q = HATLinear(dim, dim, hat_config, bias=qkv_bias)
        self.qkv_k = HATLinear(dim, dim, hat_config, bias=qkv_bias)
        self.qkv_v = HATLinear(dim, dim, hat_config, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = HATLinear(dim, dim, hat_config)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, pld: HATPayload) -> HATPayload:
        """Forward the payload by applying multi-head attention to the
        unmasked data and then projecting the result.
        """
        B, N, C = pld.data.shape
        _qkv_shape = (B, N, self.num_heads, self.head_dim)
        q = self.qkv_q(pld).reshape(*_qkv_shape).permute(0, 2, 1, 3)
        k = self.qkv_k(pld).reshape(*_qkv_shape).permute(0, 2, 3, 1)
        v = self.qkv_v(pld).reshape(*_qkv_shape).permute(0, 2, 1, 3)

        attn = (q @ k) * self.scale
        attn.apply_mask()
        attn = HATPayload(
            data=self.attn_drop(attn.data.softmax(dim=-1)),
            **attn.to_dict(include_data=False),
        )

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = x.forward_by(self.proj_drop)
        return x  # type: ignore


class TaskIndexedLayerScale(TaskIndexedModuleListABC):
    """Task-indexed layer scale module for Vision Transformer.

    Please refer to the original implementation for more details:
    :class:`timm.models.vision_transformer.LayerScale`.

    """

    def __init__(
        self,
        init_values: float = 1e-5,
        **kwargs: Any,
    ):
        super().__init__(init_values=init_values, **kwargs)
        self.init_values = init_values

    @classproperty
    def base_class(self) -> type[torch.nn.Module]:
        """Base class of task-indexed layer scale."""
        return LayerScale  # type: ignore

    def forward(self, pld: HATPayload) -> HATPayload:
        """Forward the payload by applying layer scale of the given
        task to the unmasked data.

        Args:
            pld: The payload to be forwarded.

        Returns:
            The forwarded payload.

        """
        return self.forward_(pld, use_masked_data=False)

    def forget(
        self,
        task_id: int,
        dry_run: bool = False,
        module_name: Optional[str] = None,
        locked_task_ids: Optional[list[int]] = None,
    ) -> ForgetResult:
        """Forget the given tasks by resetting the parameters of the
        layer scale module of the given task.

        Args:
            task_id: The ID of the task to be forgotten. Cannot be `None`
                even if the module accepts `None` as a task id.
            dry_run: If `True`, the forgetting process will be simulated
                without actually changing the module. Defaults to `False`.
            module_name: The name of the module. If `None`, the module name
                will be inferred from the module class name.
            locked_task_ids: The list of task ids that are locked and
                cannot be forgotten. This is ignored here, as forgetting
                of a task does not affect the other tasks.

        Returns:
            The forgetting result. See `hat.types_.ForgetResult` for more
            details.

        """
        if not dry_run:
            self[task_id].gamma.data.fill_(self.init_values)
        return self._get_forget_result(
            task_id=task_id,
            module_name=module_name,
            locked_task_ids=locked_task_ids,
        )

    @classmethod
    def from_base_module(
        cls: type[TaskIndexedLayerScale],
        base_module: LayerScale,
        num_tasks: Optional[int] = None,
        **kwargs: Any,
    ) -> TaskIndexedLayerScale:
        """Create a task-indexed layer scale module from a `LayerScale`
        module by copying the parameters of the given module to the layer
        scale modules of all the tasks.

        Args:
            base_module: The `LayerScale` module to be converted.
            num_tasks: The number of tasks. Defaults to `None` for
                compatibility  with other modules' `from_base_module`
                methods. If `None`, an error will be raised.
            **kwargs: For compatibility with other modules' `from_base_module`
                methods. Will be ignored here.

        Returns:
            A task-indexed layer scale module with the same parameters as
            the given `LayerScale` module.

        """
        if num_tasks is None:
            raise ValueError(
                "The number of tasks must be explicitly specified when "
                "creating a task-dependent layer scale module from a base "
                "module."
            )
        _device = base_module.gamma.device
        _ti_ls: TaskIndexedLayerScale = cls(
            num_tasks=num_tasks,
            dim=base_module.gamma.numel(),
            inplace=base_module.inplace,
        ).to(_device)
        _ti_ls.load_from_base_module(base_module)
        return _ti_ls


class HATBlock(HATPayloadCarrierMixin):
    """HATBlock module for Vision Transformer.

    Please refer to the original implementation for more details:
    :class:`timm.models.vision_transformer.Block`.

    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        hat_config: HATConfig,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = TaskIndexedLayerNorm,
    ):
        super().__init__()
        # noinspection PyArgumentList
        self.norm1 = norm_layer(hat_config.num_tasks, dim)
        self.attn = HATAttention(
            dim,
            hat_config=hat_config,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.ls1 = (
            TaskIndexedLayerScale(
                dim=dim,
                hat_config=hat_config,
                init_values=init_values,
            )
            if init_values is not None
            else nn.Identity()
        )
        self.drop_path1 = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

        # noinspection PyArgumentList
        self.norm2 = norm_layer(hat_config.num_tasks, dim)
        self.mlp = HATMlp(
            in_features=dim,
            hat_config=hat_config,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.ls2 = (
            TaskIndexedLayerScale(
                dim=dim,
                hat_config=hat_config,
                init_values=init_values,
            )
            if init_values is not None
            else nn.Identity()
        )
        self.drop_path2 = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

    def forward(self, pld: HATPayload) -> HATPayload:
        """Forward the payload through the vision transformer block."""
        shortcut = pld
        pld = pld.forward_by(self.norm1)
        pld = self.attn(pld)
        pld = pld.forward_by(self.ls1)
        pld = pld.forward_by(self.drop_path1)
        pld = shortcut + pld

        shortcut = pld
        pld = pld.forward_by(self.norm2)
        pld = self.mlp(pld)
        pld = pld.forward_by(self.ls2)
        pld = pld.forward_by(self.drop_path2)
        pld += shortcut
        return pld


class HATVisionTransformer(HATPayloadCarrierMixin):
    """HAT Vision Transformer.

    Please refer to the original implementation for more details:
    :class:`timm.models.vision_transformer.VisionTransformer`.

    """

    def __init__(
        self,
        hat_config: HATConfig,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        global_pool: str = "token",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        init_values: Optional[float] = None,
        class_token: bool = True,
        no_embed_class: bool = False,
        pre_norm: bool = False,
        fc_norm: Optional[bool] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        weight_init: str = "",
        embed_layer: type[HATPayloadCarrierMixin] = HATPatchEmbed,
        norm_layer: Optional[type[HATPayloadCarrierMixin]] = None,
        act_layer: nn.Module = None,
        block_fn: type[HATPayloadCarrierMixin] = HATBlock,
    ):
        super().__init__()
        assert global_pool in ("", "avg", "token")
        assert class_token or global_pool != "token"
        assert issubclass(embed_layer, HATPayloadCarrierMixin)
        if norm_layer:
            assert issubclass(norm_layer, HATPayloadCarrierMixin)
        if issubclass(block_fn, nn.Module):
            assert issubclass(block_fn, HATPayloadCarrierMixin)

        use_fc_norm = global_pool == "avg" if fc_norm is None else fc_norm
        norm_layer_partial = norm_layer or partial(
            TaskIndexedLayerNorm, eps=1e-6
        )
        act_layer = act_layer or nn.GELU

        self.hat_config = hat_config
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim
        self.num_prefix_tokens = 1 if class_token else 0
        self.no_embed_class = no_embed_class
        self.grad_checkpointing = False

        # noinspection PyArgumentList
        self.patch_embed = embed_layer(
            hat_config=hat_config,
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = (
            TaskIndexedParameter(
                num_tasks=hat_config.num_tasks,
                data=torch.zeros(1, 1, embed_dim),
            )
            if class_token
            else None
        )
        embed_len = (
            num_patches
            if no_embed_class
            else num_patches + self.num_prefix_tokens
        )
        self.pos_embed = TaskIndexedParameter(
            num_tasks=hat_config.num_tasks,
            data=torch.randn(1, embed_len, embed_dim) * 0.02,
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.norm_pre = (
            norm_layer_partial(hat_config.num_tasks, embed_dim)
            if pre_norm
            else nn.Identity()
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        # noinspection PyArgumentList
        self.blocks = nn.Sequential(
            *[
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    hat_config=hat_config,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    init_values=init_values,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer_partial,
                    act_layer=act_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = (
            norm_layer_partial(hat_config.num_tasks, embed_dim)
            if not use_fc_norm
            else nn.Identity()
        )

        # Classifier head
        self.fc_norm = (
            norm_layer_partial(hat_config.num_tasks, embed_dim)
            if use_fc_norm
            else nn.Identity()
        )
        self.head = (
            HATLinear(
                in_features=self.embed_dim,
                out_features=num_classes,
                hat_config=hat_config,
            )
            if num_classes > 0
            else nn.Identity()
        )

        if weight_init != "skip":
            self.init_weights(weight_init)

    def init_weights(self, mode: str = ""):
        """Initialize the weights."""
        assert mode in ("jax", "jax_nlhb", "moco", "")
        for __pos_embed in self.pos_embed:
            trunc_normal_(__pos_embed, std=0.02)
        if self.cls_token is not None:
            for __cls_token in self.cls_token:
                nn.init.normal_(__cls_token, std=1e-6)
        head_bias = -math.log(self.num_classes) if "nlhb" in mode else 0.0
        named_apply(get_init_weights_hat_vit(mode, head_bias), self)

    # noinspection PyMethodMayBeStatic
    def _init_weights(self, module: nn.Module):
        """Initialize the weights for the given module. This function is kept
        for backward compatibility."""
        init_weights_hat_vit_timm = init_weights_vit_timm
        init_weights_hat_vit_timm(module)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        """Load pretrained weights from the given checkpoint path."""
        print(f"Loading pretrained weights from {checkpoint_path}")
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        """Return the parameter names that should be excluded from weight
        decay."""
        return {"pos_embed", "cls_token", "dist_token"}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):  # noqa: F841
        """Return the group matcher for compatibility with timm."""
        return dict(
            stem=r"^cls_token|pos_embed|patch_embed",  # stem and embed
            blocks=[(r"^blocks\.(\d+)", None), (r"^norm", (99999,))],
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        """Enable grad checkpointing for training."""
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        """Return the classifier head."""
        return self.head

    def reset_classifier(self, num_classes: int, global_pool=None):
        """Reset the classifier head."""
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ("", "avg", "token")
            self.global_pool = global_pool
        self.head = (
            HATLinear(
                in_features=self.embed_dim,
                out_features=num_classes,
                hat_config=self.hat_config,
            )
            if num_classes > 0
            else nn.Identity()
        )

    def _pos_embed(self, pld: HATPayload) -> HATPayload:
        """Add the positional embedding to the input payload."""
        _data = pld.data
        _task_id = pld.task_id
        assert _task_id is not None
        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token
            # add then concat
            _data = _data + self.pos_embed[_task_id]
            if self.cls_token is not None:
                _data = torch.cat(
                    (
                        self.cls_token[_task_id].expand(
                            _data.shape[0], -1, -1
                        ),
                        _data,
                    ),
                    dim=1,
                )
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token
            # concat then add
            if self.cls_token is not None:
                _data = torch.cat(
                    (
                        self.cls_token[_task_id].expand(
                            _data.shape[0], -1, -1
                        ),
                        _data,
                    ),
                    dim=1,
                )
            _data = _data + self.pos_embed[_task_id]
        _data = self.pos_drop(_data)
        return HATPayload(
            data=_data,
            masker=pld.masker,
            task_id=pld.task_id,
            mask_scale=pld.mask_scale,
            locked_task_ids=pld.locked_task_ids,
            prev_maskers=pld.prev_maskers,
            mask_applied=False,  # Reapply the mask for embedding
        )

    def forward_features(self, pld: HATPayload) -> HATPayload:
        """Forward the input and return the features."""
        pld = self.patch_embed(pld)
        pld = self._pos_embed(pld)
        pld = self.norm_pre(pld)

        if self.grad_checkpointing and not torch.jit.is_scripting():
            pld = checkpoint_seq(self.blocks, pld)
        else:
            pld = self.blocks(pld)
        pld = self.norm(pld)
        return pld

    def forward_head(
        self,
        pld: HATPayload,
        pre_logits: bool = False,
    ) -> HATPayload:
        """Forward the feature and return the logits or the pre-logits."""
        _data = pld.data
        if self.global_pool:
            _data = (
                _data[:, self.num_prefix_tokens :].mean(dim=1)  # noqa: E203
                if self.global_pool == "avg"
                else _data[:, 0]
            )
        pld = HATPayload(
            data=_data,
            masker=pld.masker,
            task_id=pld.task_id,
            mask_scale=pld.mask_scale,
            locked_task_ids=pld.locked_task_ids,
            prev_maskers=pld.prev_maskers,
            mask_applied=True,
        )
        pld = self.fc_norm(pld)
        if not pre_logits:
            pld = self.head(pld)
        return pld

    def forward(self, pld: HATPayload) -> HATPayload:
        """Forward the input and return the logits."""
        pld = self.forward_features(pld)
        pld = self.forward_head(pld)
        return pld


def get_init_weights_hat_vit(mode="jax", head_bias: float = 0.0):
    """Get HAT ViT weight initialization function."""
    # For JAX and timm init, we use the same init functions imported from
    # `timm.models.vision_transformer`.
    init_weights_hat_vit_jax = init_weights_vit_jax
    init_weights_hat_vit_timm = init_weights_vit_timm
    if "jax" in mode:
        return partial(init_weights_hat_vit_jax, head_bias=head_bias)
    elif "moco" in mode:
        return init_weights_hat_vit_moco
    else:
        return init_weights_hat_vit_timm


def init_weights_hat_vit_moco(module: nn.Module, name: str = ""):
    """
    HAT ViT weight initialization, matching moco-v3 impl minus fixed
    PatchEmbed.
    """
    if isinstance(module, nn.Linear):
        if "qkv_" in name:
            # treat the weights of Q, K, V separately
            val = math.sqrt(
                6.0 / float(module.weight.shape[0] + module.weight.shape[1])
            )
            nn.init.uniform_(module.weight, -val, val)
        else:
            nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, "init_weights"):
        module.init_weights()


@torch.no_grad()
def _load_weights(
    model: VisionTransformer,
    checkpoint_path: str,
    prefix: str = "",
):
    """Load weights from .npz checkpoints for official Google Brain Flax
    implementation.
    """
    import numpy as np

    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    def _load_ln(
        ln: TaskIndexedLayerNorm,
        w: torch.Tensor,
        b: torch.Tensor,
    ):
        for __ln in ln:
            __ln.weight.copy_(w)
            __ln.bias.copy_(b)

    w = np.load(checkpoint_path)
    if not prefix and "opt/target/embedding/kernel" in w:
        prefix = "opt/target/"

    if hasattr(model.patch_embed, "backbone"):
        # hybrid0; this part is not tested
        backbone = model.patch_embed.backbone
        stem_only = not hasattr(backbone, "stem")
        stem = backbone if stem_only else backbone.stem
        stem.conv.weight.copy_(
            adapt_input_conv(
                stem.conv.weight.shape[1], _n2p(w[f"{prefix}conv_root/kernel"])
            )
        )
        stem.norm.weight.copy_(_n2p(w[f"{prefix}gn_root/scale"]))
        stem.norm.bias.copy_(_n2p(w[f"{prefix}gn_root/bias"]))
        if not stem_only:
            for i, stage in enumerate(backbone.stages):
                for j, block in enumerate(stage.blocks):
                    bp = f"{prefix}block{i + 1}/unit{j + 1}/"
                    for r in range(3):
                        getattr(block, f"conv{r + 1}").weight.copy_(
                            _n2p(w[f"{bp}conv{r + 1}/kernel"])
                        )
                        getattr(block, f"norm{r + 1}").weight.copy_(
                            _n2p(w[f"{bp}gn{r + 1}/scale"])
                        )
                        getattr(block, f"norm{r + 1}").bias.copy_(
                            _n2p(w[f"{bp}gn{r + 1}/bias"])
                        )
                    if block.downsample is not None:
                        block.downsample.conv.weight.copy_(
                            _n2p(w[f"{bp}conv_proj/kernel"])
                        )
                        block.downsample.norm.weight.copy_(
                            _n2p(w[f"{bp}gn_proj/scale"])
                        )
                        block.downsample.norm.bias.copy_(
                            _n2p(w[f"{bp}gn_proj/bias"])
                        )
        embed_conv_w = _n2p(w[f"{prefix}embedding/kernel"])
    else:
        embed_conv_w = adapt_input_conv(
            model.patch_embed.proj.weight.shape[1],
            _n2p(w[f"{prefix}embedding/kernel"]),
        )
    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(w[f"{prefix}embedding/bias"]))
    for _cls_token in model.cls_token:
        _cls_token.copy_(_n2p(w[f"{prefix}cls"], t=False))
    pos_embed_w = _n2p(
        w[f"{prefix}Transformer/posembed_input/pos_embedding"], t=False
    )
    if pos_embed_w.shape != model.pos_embed[0].shape:
        pos_embed_w = resize_pos_embed(
            # resize pos embedding when different size from pretrained weights
            pos_embed_w,
            model.pos_embed,
            getattr(model, "num_prefix_tokens", 1),
            model.patch_embed.grid_size,
        )
    for _pos_embed in model.pos_embed:
        _pos_embed.copy_(pos_embed_w)

    if isinstance(model.norm, TaskIndexedLayerNorm):
        _load_ln(
            model.norm,
            _n2p(w[f"{prefix}Transformer/encoder_norm/scale"]),
            _n2p(w[f"{prefix}Transformer/encoder_norm/bias"]),
        )
    else:
        model.norm.weight.copy_(
            _n2p(w[f"{prefix}Transformer/encoder_norm/scale"]),
        )
        model.norm.bias.copy_(
            _n2p(w[f"{prefix}Transformer/encoder_norm/bias"]),
        )
    if (
        isinstance(model.head, HATLinear)
        and model.head.bias.shape[0] == w[f"{prefix}head/bias"].shape[-1]
    ):
        model.head.weight.copy_(_n2p(w[f"{prefix}head/kernel"]))
        model.head.bias.copy_(_n2p(w[f"{prefix}head/bias"]))
    # Representation layer has been removed in latest 21k/1k pretrained weights
    for i, block in enumerate(model.blocks.children()):
        block_prefix = f"{prefix}Transformer/encoderblock_{i}/"
        mha_prefix = block_prefix + "MultiHeadDotProductAttention_1/"
        if isinstance(block.norm1, TaskIndexedLayerNorm):
            _load_ln(
                block.norm1,
                _n2p(w[f"{block_prefix}LayerNorm_0/scale"]),
                _n2p(w[f"{block_prefix}LayerNorm_0/bias"]),
            )
        else:
            block.norm1.weight.copy_(
                _n2p(w[f"{block_prefix}LayerNorm_0/scale"]),
            )
            block.norm1.bias.copy_(
                _n2p(w[f"{block_prefix}LayerNorm_0/bias"]),
            )
        block.attn.qkv_q.weight.copy_(
            _n2p(w[f"{mha_prefix}query/kernel"], t=False).flatten(1).T,
        )
        block.attn.qkv_k.weight.copy_(
            _n2p(w[f"{mha_prefix}key/kernel"], t=False).flatten(1).T,
        )
        block.attn.qkv_v.weight.copy_(
            _n2p(w[f"{mha_prefix}value/kernel"], t=False).flatten(1).T,
        )
        block.attn.qkv_q.bias.copy_(
            _n2p(w[f"{mha_prefix}query/bias"], t=False).reshape(-1),
        )
        block.attn.qkv_k.bias.copy_(
            _n2p(w[f"{mha_prefix}key/bias"], t=False).reshape(-1),
        )
        block.attn.qkv_v.bias.copy_(
            _n2p(w[f"{mha_prefix}value/bias"], t=False).reshape(-1),
        )
        block.attn.proj.weight.copy_(
            _n2p(w[f"{mha_prefix}out/kernel"]).flatten(1),
        )
        block.attn.proj.bias.copy_(
            _n2p(w[f"{mha_prefix}out/bias"]),
        )
        for r in range(2):
            getattr(block.mlp, f"fc{r + 1}").weight.copy_(
                _n2p(w[f"{block_prefix}MlpBlock_3/Dense_{r}/kernel"])
            )
            getattr(block.mlp, f"fc{r + 1}").bias.copy_(
                _n2p(w[f"{block_prefix}MlpBlock_3/Dense_{r}/bias"])
            )
        if isinstance(block.norm2, TaskIndexedLayerNorm):
            _load_ln(
                block.norm2,
                _n2p(w[f"{block_prefix}LayerNorm_2/scale"]),
                _n2p(w[f"{block_prefix}LayerNorm_2/bias"]),
            )
        else:
            block.norm2.copy_(
                _n2p(w[f"{block_prefix}LayerNorm_2/scale"]),
            )
            block.norm2.copy_(
                _n2p(w[f"{block_prefix}LayerNorm_2/bias"]),
            )


def _create_hat_vision_transformer(variant, pretrained=False, **kwargs):
    """Create a HAT-Vision-Transformer model."""
    if kwargs.get("features_only", None):
        raise RuntimeError(
            "features_only not implemented for Vision Transformer models."
        )
    pretrained_cfg = resolve_pretrained_cfg(
        variant,
        pretrained_cfg=kwargs.pop(
            "pretrained_cfg",
            default_cfgs.pop(variant, None),
        ),
    )
    model = build_model_with_cfg(
        HATVisionTransformer,
        variant,
        pretrained,
        pretrained_cfg=pretrained_cfg,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load="npz" in pretrained_cfg["url"],
        **kwargs,
    )
    return model


@register_model
def hat_vit_tiny_patch16_224(pretrained=False, **kwargs):
    """HAT-ViT-Tiny (Vit-Ti/16)"""
    model_kwargs = dict(
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        **kwargs,
    )
    model = _create_hat_vision_transformer(
        variant="hat_vit_tiny_patch16_224",
        pretrained=pretrained,
        **model_kwargs,
    )
    return model


@register_model
def hat_vit_tiny_patch16_224_in21k(pretrained=False, **kwargs):
    """HAT-ViT-Tiny (Vit-Ti/16).
    ImageNet-21k weights @ 224x224; this model has valid 21k classifier head
    and no representation (pre-logits) layer
    """
    model_kwargs = dict(
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        **kwargs,
    )
    model = _create_hat_vision_transformer(
        variant="hat_vit_tiny_patch16_224_in21k",
        pretrained=pretrained,
        **model_kwargs,
    )
    return model
