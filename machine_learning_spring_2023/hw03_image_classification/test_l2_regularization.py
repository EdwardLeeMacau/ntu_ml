import torch
from torch import nn
from torch.optim import SGD
from torchvision import models
from model import Regularization


def test_optimizer_weight_decay():
    """ Verify what have done on batch normalization layer when weight decay is nonzero. """
    model = nn.BatchNorm1d(5)
    criteria = nn.MSELoss()
    optimizer = SGD(model.parameters(), lr=1, weight_decay=1e-1)
    print(f'{dict(model.named_parameters())=}')

    def update_param(x):
        # Make a fake label such that MSELoss equals 0.
        pred = model(x)
        label = pred.detach()
        # print(f'{x=}')
        # print(f'{pred=}')
        # print(f'{label=}')

        # Back-propagate once
        loss = criteria(pred, label)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        return loss.item()

    for _ in range(10):
        # Prediction always equals to label.
        # Therefore, MSELoss should produce 0 gradient to bottom parameters.
        x = torch.tensor([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=torch.float)
        loss = update_param(x)
        print(f'{loss=}')
        print(f'{dict(model.named_parameters())=}')

def test_explicit_weight_decay():
    """ Test module Regularization. """
    model = models.resnet18(num_classes=1)
    criteria = nn.MSELoss()
    optimizer = SGD(model.parameters(), lr=0.5, weight_decay=0)
    regularization = Regularization(l2norm=1)

    # for m in model.modules():
    #     print(f'{type(m)=}, {isinstance(m, torch.nn.modules.batchnorm._BatchNorm)=}')
    # for i, (name, p) in enumerate(model.named_parameters()):
    #     print(f'{i:02d} - {name=}, {p=}')

    def update_param(x):
        # Make a fake label such that MSELoss equals 0.
        pred = model(x)
        label = pred.detach()
        # print(f'{x=}')
        # print(f'{pred=}')
        # print(f'{label=}')

        # Back-propagate once
        loss = criteria(pred, label) + regularization(model)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        return loss.item()

    layer4 = model.layer4
    conv = layer4[0].conv1
    bn = layer4[0].bn1

    print(f"{list(conv.parameters())}")
    print(f"{list(bn.parameters())}")

    for _ in range(100):
        x = torch.randn(size=(2, 3, 32, 32))
        loss = update_param(x)
        print(f'{loss=}')

    print(f"{list(conv.parameters())}")
    print(f"{list(bn.parameters())}")

if __name__ == "__main__":
    # test_optimizer_weight_decay()
    test_explicit_weight_decay()