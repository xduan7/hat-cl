"""Contains models implemented with HAT modules.

Examples:
    To create any model implemented in this module:
    ```python
    import timm
    import hat.timm_models
    from hat import HATConfig

    model_name = "hat_resnet18"
    hat_config = HATConfig(num_tasks=10)
    model = timm.create_model(model_name, hat_config=hat_config)
    ```
"""
from .resnet import *  # noqa: F401, F403
from .vision_transformer import *  # noqa: F401, F403
