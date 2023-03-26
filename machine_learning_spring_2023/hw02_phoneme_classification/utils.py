import random
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam, RAdam, SGD
from typing import Dict


""" Mapping table, to convert string as scheduler constructor.

Reference: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
"""
_scheduler = {
    'StepLR': StepLR,
}

""" Mapping table, to convert string as optimizer constructor.

Reference: https://pytorch.org/docs/stable/optim.html
"""
_optimizer = {
    'Adam': Adam,
    'RAdam': RAdam,
    'SGD': SGD,
}

def same_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def flatten(param: Dict, prefix: str = ""):
    retval = {}

    for k, v in param.items():
        p = k if prefix == '' else f'{prefix}.{k}'
        retval.update(flatten(v, p)) if isinstance(v, dict) else retval.update({p: v})

    return retval

def count_parameters(model: torch.nn.Module):
    """ Return number of trainable parameters in a torch model.

    Reference: https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
