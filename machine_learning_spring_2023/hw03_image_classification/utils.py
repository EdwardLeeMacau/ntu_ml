import random
import numpy as np
import torch
import os
import io
from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam, RAdam, SGD
from typing import Dict, Tuple
from matplotlib import pyplot as plt
import itertools

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

# TODO: Logger which adapts Google Colab, Kaggle, and server
class Logger:
    pass

def argmin(d: Dict):
    val = min(d.values())
    for k, v in d.items():
        if v == val:
            return (k, v)

def argmax(d: Dict):
    val = max(d.values())
    for k, v in d.items():
        if v == val:
            return (k, v)

def sizeof_fmt(num, suffix="B") -> str:
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"

# TODO: support mixup augmentation
# Reference:
#
# https://arxiv.org/pdf/1710.09412.pdf
# https://github.com/moskomule/mixup.pytorch/blob/master/main.py
# https://www.kaggle.com/code/kaushal2896/data-augmentation-tutorial-basic-cutout-mixup
def mixup(batch, alpha=0.2, dtype=torch.float) -> Tuple[torch.Tensor, torch.Tensor]:
    x, y = batch
    size = x.shape[0]

    shuffle = torch.randperm(size)
    lamda_y = torch.tensor(np.random.beta(alpha, alpha, size), dtype=dtype).unsqueeze(dim=-1)
    lamda_x = lamda_y.unsqueeze(dim=-1).unsqueeze(dim=-1)

    x = lamda_x * x + (1 - lamda_x) * x[shuffle]
    y = lamda_y * y + (1 - lamda_y) * y[shuffle]

    return (x, y)

class ModelCheckpointPreserver:
    """ k-max model preserving """
    def __init__(self, key: str, k=1, dirname="./"):
        self.k = k
        self.key = key
        self.dirname = dirname
        self.records = { }

    def is_full(self):
        return len(self.records) == self.k

    def update(self, model: torch.nn.Module, metric: float, iteration: int) -> bool:
        def fname(iteration: int):
            return os.path.join(f"{self.dirname}", f"model-{iteration:08d}.pt")

        # If metric is lower than all elements in queue (don't be kept)
        if self.is_full() and metric < min(self.records.values()):
            return False

        # Drop previous model from disk
        if self.is_full():
            min_key, _ = argmin(self.records)

            self.records.pop(min_key)
            os.remove(fname(min_key))

        # Store current model to disk
        self.records[iteration] = metric

        device = next(model.parameters()).device

        model = model.cpu()
        torch.save(model.state_dict(), fname(iteration))
        model = model.to(device)

        return True

    def get_best(self) -> Tuple[int, float]:
        return argmax(self.records)

def same_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# TODO: Handle keyword 'kwarg'
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

def plot_confusion_matrix(cm, class_names, **kwargs) -> io.BytesIO:
    """ Convert confusion matrix as an image in .png format stores in BytesIO buffer.

    Reference: https://towardsdatascience.com/exploring-confusion-matrix-evolution-on-tensorboard-e66b39f4ac12
    """
    figure = plt.figure(figsize=(8, 8))

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)

    return buf
