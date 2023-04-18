from typing import Any, Literal

import torch.nn as nn


def get_hat_reg_term(
    module: nn.Module,
    reg_strat: Literal["uniform"],
    **kwargs: Any,
) -> float:
    """Get the regularization term of a HAT maskers in the given module.

    Args:
        module: The module to get the regularization term from.
        reg_strat: The regularization strategy to use. See
            `hat.modules.HATMasker.get_reg_term` for more details.
        **kwargs: Keyword arguments to be passed to the regularization
            strategy. See `hat.modules.HATMasker.get_reg_term` for more
            details.

    Returns:
        The regularization term.

    """
    from hat.modules import HATMasker

    _reg, _cnt = 0.0, 0
    for __m in module.modules():
        if isinstance(__m, HATMasker):
            _reg += __m.get_reg_term(
                reg_strat=reg_strat,
                **kwargs,
            )
            _cnt += 1
    return _reg / _cnt if _cnt > 0 else 0.0
