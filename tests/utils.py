import random

import numpy as np
import torch
import torch.nn as nn

# noinspection PyProtectedMember
from torch.nn.modules.batchnorm import _BatchNorm

from hat.exceptions import LearningSuppressedWarning, ModuleConversionWarning

from .constants import RANDOM_SEED


def set_random_seed(random_seed: int = RANDOM_SEED):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)


def supress_warnings():
    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=LearningSuppressedWarning)
    warnings.filterwarnings("ignore", category=ModuleConversionWarning)


def set_up():
    set_random_seed()
    supress_warnings()


def deactivate_dropout(module: nn.Module):
    """Deactivate dropout in the module by setting the dropout layer to
    evaluation mode and setting the p value to zero.
    """
    if isinstance(module, nn.Dropout):
        module.p = 0
        module.eval()
    for __m in module.children():
        deactivate_dropout(__m)


def deactivate_tracking_stats(module: nn.Module):
    """Deactivate tracking of running statistics in the module by setting the
    batch normalization layer to evaluation mode and setting the
    `track_running_stats` flag to False.
    """
    if isinstance(module, _BatchNorm):
        module.track_running_stats = False
        module.eval()
    for __m in module.children():
        deactivate_tracking_stats(__m)
