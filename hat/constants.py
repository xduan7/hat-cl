from typing import Final

# Default maximum training scale value for attention masks.
DEF_HAT_MAX_TRN_MASK_SCALE: Final[float] = 100.0

# Default initialization strategy for the HAT attention parameters.
DEF_HAT_INIT_STRAT: Final[str] = "normal"

# Default gradient compensation clamp value for attention masks.
DEF_HAT_GRAD_COMP_CLAMP: Final[float] = 50.0

# Default gradient compensation factor for attention masks.
DEF_HAT_GRAD_COMP_FACTOR: Final[float] = 100.0

# Default clipping value for the attention values.
DEF_HAT_ATTN_CLAMP: Final[float] = 6.0

# Default scaling strategy for HAT attention masks training.
DEF_HAT_TRN_MASK_SCALE_STRAT: Final[str] = "linear"
