import timm

# noinspection PyUnresolvedReferences
import hat.timm_models
from hat import HATConfig, HATPayload

hat_config = HATConfig(num_tasks=10)
model = timm.create_model("hat_vit_tiny_patch16_224", hat_config=hat_config)
model = model.to("cuda")

print(model.cls_token[0].device)
