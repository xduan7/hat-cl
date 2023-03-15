import random

import numpy as np
import torch

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
