# TODO



## limitations so far:


### Refreshing optimizer state

It is highly advised to refresh the optimizer state after every task, to avoid carrying over the momentum from the previous task.
This can be done by simply re-initializing the optimizer.


### Weight decay (L2 regularization)

Cannot use weight decay (L2 regularization) due to how weight decay works.
The way weight decay works is that, it adds the weight parameter values to the gradients:

```python3

for i, param in enumerate(params):
    
    # Maximize the params based on the objective if necessary
    # ...

    # noinspection PyUnresolvedReferences
    d_p = d_p.add(param, alpha=weight_decay)
    
    # Add momentum if necessary
    # ...
    
    # Update the params
    param.add_(d_p, alpha=-lr)

```

The code segment above is what happens in the step function. 
At this point, the gradient is already modified (compensated or conditioned) by the HAT mechanism.
However, the weight decay changes the gradient on top of the mechanism.
This might change the parameters that are supposed to be locked out by the HAT mechanism, and therefore causing forgetting.


