import torch
from torch import nn
from pytorchcv.model_provider import get_model


class EnsembleNet(nn.Module):
    """
    * ## Example of Ensemble Attack
    * Ensemble multiple models as your proxy model to increase the black-box transferability ([paper](https://arxiv.org/abs/1611.02770))
    """
    def __init__(self, model_names):
        super().__init__()
        self.models = nn.ModuleList([
            get_model(name, pretrained=True) for name in model_names
        ])

    # TODO: Check the implementation again
    def forward(self, x):
        logits = []

        for i, m in enumerate(self.models):
            logits.append(m(x))

        return torch.stack(logits, dim=0).mean(dim=0)

def fgsm(model: nn.Module, x, y, loss_fn, epsilon=8, **kwargs):
    """ Fast gradient sign method, use gradient ascent on x_adv to maximize loss. """
    device = next(model.parameters()).device

    x_adv = x.detach().clone() # initialize x_adv as original benign image x
    x_adv.requires_grad = True # need to obtain gradient of x_adv to find best perturbation
    epsilon = epsilon.to(device)

    loss = loss_fn(model(x_adv), y)
    loss.backward()

    grad = x_adv.grad.detach()
    x_adv = x_adv + epsilon * grad.sign()

    return x_adv

def ifgsm(model, x, y, loss_fn, epsilon=8, alpha=0.8, num_iter=20, **kwargs):
    """
    Iterated fast gradient sign method.

    # alpha and num_iter can be decided by yourself
    """
    x_adv = x
    for _ in range(num_iter):
        # x_adv = fgsm(model, x_adv, y, loss_fn, alpha) # call fgsm with (epsilon = alpha) to obtain new x_adv
        x_adv = x_adv.detach().clone()
        x_adv.requires_grad = True # need to obtain gradient of x_adv, thus set required grad

        loss = loss_fn(model(x_adv), y) # calculate loss
        loss.backward() # calculate gradient
        # fgsm: use gradient ascent on x_adv to maximize loss
        grad = x_adv.grad.detach()
        x_adv = x_adv + alpha * grad.sign()

        x_adv = torch.max(torch.min(x_adv, x + epsilon), x - epsilon) # clip new x_adv back to [x-epsilon, x+epsilon]

    return x_adv

def mifgsm(model, x, y, loss_fn, epsilon=8, alpha=0.8, num_iter=20, decay=1.0, **kwargs):
    """
    Multistep Iterated fast gradient sign method.
    """
    device = next(model.parameters()).device
    x_adv = x

    # initialize momentum tensor
    momentum = torch.zeros_like(x).detach().to(device)
    # write a loop of num_iter to represent the iterative times
    for i in range(num_iter):
        x_adv = x_adv.detach().clone()
        x_adv.requires_grad = True # need to obtain gradient of x_adv, thus set required grad

        loss = loss_fn(model(x_adv), y) # calculate loss
        loss.backward() # calculate gradient
        # TODO: Momentum calculation
        grad = None
        x_adv = x_adv + alpha * grad.sign()
        x_adv = torch.max(torch.min(x_adv, x+epsilon), x-epsilon) # clip new x_adv back to [x-epsilon, x+epsilon]

    return x_adv
