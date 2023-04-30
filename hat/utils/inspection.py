from typing import TYPE_CHECKING, Optional, Sequence, Union

import pandas as pd
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from hat.modules import HATMasker
else:
    HATMasker = nn.Module


@torch.no_grad()
def _get_hat_masker_util(
    hat_masker: HATMasker,
    task_ids: Optional[Union[int, Sequence[int]]] = None,
) -> dict[int, tuple[int, int]]:
    """Get the HAT utility of a HAT masker."""
    if task_ids is None:
        task_ids = hat_masker.trained_task_ids
    elif isinstance(task_ids, int):
        task_ids = [task_ids]

    _hat_masker_util = {}
    for __task_id in task_ids:
        if __task_id not in hat_masker.trained_task_ids:
            raise ValueError(f"Task ID {__task_id} is not trained.")
        _hat_masker_util[__task_id] = (
            hat_masker.get_binary_mask(
                task_id=__task_id,
                from_cache=False,
            )
            .sum()
            .item(),
            hat_masker.num_features,
        )
    return _hat_masker_util


@torch.no_grad()
def get_hat_util(
    module: nn.Module,
    task_ids: Optional[Union[int, Sequence[int]]] = None,
) -> pd.DataFrame:
    """Get the HAT utility of a module as a Pandas DataFrame.

    The returned DataFrame has rows as the names of the HAT maskers
    from the given module, and columns as the task IDs. The values
    of each cell are a tuple of the number of activations (unmasked
    features) and the total number of features.

    Args:
        module: The module to be inspected.
        task_ids: The task IDs to be inspected. If `None`, all the
            trained tasks will be inspected. Defaults to `None`.

    Returns:
        A Pandas DataFrame containing the HAT utility.

    """
    # noinspection PyProtectedMember
    from hat.modules._base import HATMaskedModuleABC

    _hat_util = {}
    for __n, __m in module.named_modules():
        if isinstance(__m, HATMaskedModuleABC):
            __masker = __m.masker
            _hat_util[f"{__n}.masker"] = _get_hat_masker_util(
                hat_masker=__masker,
                task_ids=task_ids,
            )
    _hat_util_df = pd.DataFrame.from_dict(
        data=_hat_util,
        orient="index",
    )
    # TODO: add column and index names
    # TODO: add summary
    return _hat_util_df
