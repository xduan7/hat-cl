import math
from typing import Any, Optional

from hat.constants import (
    DEF_HAT_MAX_TRN_MASK_SCALE,
    DEF_HAT_TRN_MASK_SCALE_STRAT,
)


def get_hat_mask_scale(
    strat: str = DEF_HAT_TRN_MASK_SCALE_STRAT,
    max_trn_mask_scale: float = DEF_HAT_MAX_TRN_MASK_SCALE,
    **kwargs: Any,
) -> float:
    """Get the scale of the HAT attention mask.

    This function is a wrapper for multiple scaling strategies. See the
    following functions for more details:
    - `_get_linear_hat_mask_scale`
    - `_get_exponential_hat_mask_scale`
    - `_get_cosine_hat_mask_scale`

    Args:
        strat: The scaling strategy to use. Can be one of the following:
            "linear", "exponential", "cosine".
        max_trn_mask_scale: The maximum scale of the HAT attention mask.
        **kwargs: Keyword arguments to be passed to the scaling strategy.

    Returns:
        The scale of the HAT attention mask.

    """
    if strat == "linear":
        return _get_linear_hat_mask_scale(
            max_trn_mask_scale=max_trn_mask_scale,
            **kwargs,
        )
    elif strat == "exponential":
        return _get_exponential_hat_mask_scale(
            max_trn_mask_scale=max_trn_mask_scale,
            **kwargs,
        )
    elif strat == "cosine":
        return _get_cosine_hat_mask_scale(
            max_trn_mask_scale=max_trn_mask_scale,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown mask scaling strategy: {strat}.")


def _get_linear_hat_mask_scale(
    progress: float,
    max_trn_mask_scale: float,
    min_trn_mask_scale: Optional[float] = None,
) -> float:
    """Get the scale of the HAT attention mask using a linear scaling.

    The scale is linearly scaled from `min_trn_mask_scale` to
    `max_trn_mask_scale` as the training progresses. This is the scaling
    strategy proposed in the original HAT paper.

    Args:
        progress: The training progress. Must be in the range [0, 1].
        max_trn_mask_scale: The maximum scale of the HAT attention mask.
        min_trn_mask_scale: The optional minimum scale of the HAT attention
            mask. If not provided, it will be set to 1 / `max_trn_mask_scale`.

    Returns:
        The scale of the HAT attention mask.

    """
    # The minimum mask scale prevents the scale from being 0, which is not
    # gradient-friendly during training.
    if min_trn_mask_scale is None:
        min_trn_mask_scale = 1 / max_trn_mask_scale
    return (
        max_trn_mask_scale - min_trn_mask_scale
    ) * progress + min_trn_mask_scale


def _get_exponential_hat_mask_scale(
    progress: float,
    max_trn_mask_scale: float,
    ratio_greater_than_one: float = 0.8,
) -> float:
    """Get the scale of the HAT attention mask using an exponential scaling.

    The scale is exponentially scaled to `max_trn_mask_scale` as the training
    progresses. The purpose of this scaling strategy is to keep the scale
    lower than 1 for a longer period of time, which gives the masks more
    time to learn.

    Args:
        progress: The training progress. Must be in the range [0, 1].
        max_trn_mask_scale: The maximum scale of the HAT attention mask.
        ratio_greater_than_one: The ratio of the scale bigger than 1 in the
            training process, e.g. 0.8 means the scale is bigger than 1 for
            80% of the training process. Must be in the range (0, 1).
            Default to 0.8.

    Returns:
        The scale of the HAT attention mask.

    """
    _exp_base = max_trn_mask_scale ** (1 / ratio_greater_than_one)
    return _exp_base ** (progress - 1 + ratio_greater_than_one)  # type: ignore


def _get_cosine_hat_mask_scale(
    progress: float,
    max_trn_mask_scale: float,
    min_trn_mask_scale: float = 1,
) -> float:
    """Get the scale of the HAT attention mask using a cosine scaling.

    The scale is scaled from `max_trn_mask_scale` to `0` and back to
    `max_trn_mask_scale` using a cosine function as the training
    progresses. If the cosine function goes below `min_trn_mask_scale`,
    `min_trn_mask_scale` will be used instead. This is the scaling strategy
    trains the model weights at the beginning to make sure that the weights
    are in a good direction, and then trains the masks in the middle,
    and finally fine-tunes the model weights at the end with binary masks.

    Args:
        progress: The training progress. Must be in the range [0, 1].
        max_trn_mask_scale: The maximum scale of the HAT attention mask.
        min_trn_mask_scale: The minimum scale of the HAT attention mask.
            If the cosine function goes below this value, this value will be
            used instead. Default to 1.

    Returns:
        The scale of the HAT attention mask.

    """
    half_scale = max_trn_mask_scale / 2
    _scale = half_scale * math.cos(progress * 2 * math.pi) + half_scale
    return _scale if _scale >= min_trn_mask_scale else min_trn_mask_scale
