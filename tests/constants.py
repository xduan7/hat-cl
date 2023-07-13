import functools

import torch

from hat.constants import DEF_HAT_MAX_TRN_MASK_SCALE
from hat.types_ import HATConfig

DEBUG = False
RANDOM_SEED = 0
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
    # As of PyTorch 1.13.1, MPS might cause wrong results.
    # else ("mps" if torch.backends.mps.is_available() else "cpu")
)

BATCH_SIZE = 2

NUM_TASKS = 2 if DEBUG else 10
MAX_TRN_MASK_SCALE = DEF_HAT_MAX_TRN_MASK_SCALE
TRN_MASK_SCALE = 3.0
HAT_CONFIG = HATConfig(num_tasks=NUM_TASKS)

DROPOUT_RATE = 0.2

# Small learning rate to preserve random distribution of the params.
LEARNING_RATE = 1e-5
MOMENTUM = 0.9
# Weight decay must be zero to prevent forgetting over L2 regularization.
WEIGHT_DECAY = 0
OPTIMIZERS = {
    "SGD": functools.partial(
        torch.optim.SGD,
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    ),
    "Adam": functools.partial(
        torch.optim.Adam,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    ),
    "AdamW": functools.partial(
        torch.optim.AdamW,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    ),
    "RMSprop": functools.partial(
        torch.optim.RMSprop,
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    ),
}
