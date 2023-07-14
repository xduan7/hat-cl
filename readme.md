# HAT-CL

Redesigned Hard-Attention-to-the-Task for Continual Learning

HAT-CL is a comprehensive reimagining of the Hard-Attention-to-the-Task (HAT) mechanism, designed specifically to combat catastrophic forgetting during Continual Learning (CL). 
Originally proposed in the paper [Overcoming catastrophic forgetting with hard attention to the task](https://arxiv.org/abs/1612.00796), HAT has been instrumental in enabling neural networks to learn successive tasks without erasure of prior knowledge. 
However, the original implementation had its drawbacks, notably incompatibility with PyTorch's optimizers and the requirement for manual gradient manipulation.
HAT-CL aims to rectify these issues with a user-friendly design and a host of new features:

- Seamless compatibility with all PyTorch operations and optimizers.
- Automated gradient manipulation through PyTorch hooks.
- Simple transformation of PyTorch modules to HAT modules with a single line of code.
- Out-of-the-box HAT networks integrated with [timm](https://github.com/huggingface/pytorch-image-models).

---

## Quick Start

### Installation

To install via pip:

```bash
pip install hat-cl
```

Or, if you are using poetry:

```bash
poetry add hat-cl
```

### Basic Usage

To use HAT modules, simply swap the generic PyTorch modules for their HAT counterparts. 
For example, replace `torch.nn.Linear` with `hat.modules.HATLinear`, and `torch.nn.Conv2d` with `hat.modules.HATConv2d`. 
The HAT modules take instances of `hat.HATPayload` as input and output, wrapping the tensor, task ID, and other essential variables for the HAT mechanism.

Below is a basic example of a 2-layer MLP:

```python3
import torch
import torch.nn as nn
from hat import HATPayload, HATConfig
from hat.modules import HATLinear


hat_config = HATConfig(num_tasks=5)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear1 = HATLinear(input_dim, hidden_dim, hat_config)
        self.relu = nn.ReLU()
        self.linear2 = HATLinear(hidden_dim, output_dim, hat_config)
        
    def forward(self, x: HATPayload):
        x = self.linear1(x)
        # You can still pass the payload to the non-HAT modules like this
        x = x.forward_by(self.relu)
        x = self.linear2(x)
        return x
    
    
mlp = MLP(input_dim=128, hidden_dim=32, output_dim=2)

input_payload = HATPayload(torch.rand(10, 128), task_id=0, mask_scale=10.0)
output_payload = mlp(input_payload)
output_data = output_payload.data
```

With these steps, you've created a 2-layer MLP with the HAT mechanism and successfully conducted a forward pass through the model. 
Just like any other PyTorch modules, it's ready to be trained, evaluated, and more—all under-the-hood operations are handled by the HAT modules.

Additionally, HAT-CL provides ready-to-use HAT networks with timm integration. Creating a HAT model is as simple as creating any other [timm](www.github.com/rwightman/pytorch-image-models) model:

```python3
import timm
import hat.timm_models  # This line is necessary to register the HAT models to timm
from hat import HATConfig

hat_config = HATConfig(num_tasks=5)
hat_resnet18 = timm.create_model('hat_resnet18', hat_config=hat_config)
```

---

## Modules

Here's a handy table of PyTorch modules and their HAT counterparts:

| PyTorch module         | HAT module                           |
|------------------------|--------------------------------------|
| `torch.nn.Linear`      | `hat.modules.HATLinear`              |
| `torch.nn.Conv1d`      | `hat.modules.HATConv1d`              |
| `torch.nn.Conv2d`      | `hat.modules.HATConv2d`              |
| `torch.nn.Conv3d`      | `hat.modules.HATConv3d`              |
| `torch.nn.BatchNorm1d` | `hat.modules.TaskIndexedBatchNorm1d` |
| `torch.nn.BatchNorm2d` | `hat.modules.TaskIndexedBatchNorm2d` |
| `torch.nn.BatchNorm3d` | `hat.modules.TaskIndexedBatchNorm3d` |
| `torch.nn.LayerNorm`   | `hat.modules.TaskIndexedLayerNorm`   |


---


## Networks

Here are the currently available timm-compatible HAT networks:

| HAT Network Name           | Has pretrained weights | Description                      |
|----------------------------|------------------------|----------------------------------|
| `hat_resnet18`             | No                     | HAT ResNet-18                    |
| `hat_resnet18s`            | No                     | HAT ResNet-18 for smaller images |
| `hat_resnet34`             | No                     | HAT ResNet-34                    |
| `hat_resnet34s`            | No                     | HAT ResNet-34 for smaller images |
| `hat_vit_tiny_patch16_224` | Yes                    | HAT ViT-Tiny (16, 224)           |


---

## Examples

- [Continual Learning](examples%2Fcontinual_learning.ipynb)
- [Feature Importance](examples%2Ffeature_importance.ipynb)


---


## Limitations

While HAT-CL is designed to be compatible with all PyTorch features, there are some inherent limitations due to the nature of the HAT mechanism:

- **Optimizer Re-initialization**: To prevent carryover of momentum from previous tasks, it is recommended to refresh the optimizer state after each task. This can be achieved by simply re-initializing the optimizer. 
- **Weight Decay (L2 Regularization)**: Weight decay is not compatible with the HAT mechanism due to its gradient modification process. As this occurs after the HAT mechanism modifies the gradient, it may interfere with parameters that should be locked out by the HAT mechanism, leading to potential forgetting. This includes the `weight_decay` parameter in the optimizer, or any optimizer that utilizes weight decay (e.g. AdamW).


---


## TODO

- [ ] Add example notebook for pruning
- [ ] Package paper for implementation details


---


## Authors

Xiaotian Duan (xduan7 at gmail dot com)
