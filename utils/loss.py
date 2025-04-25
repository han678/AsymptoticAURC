import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import digamma
from torch import Tensor
from torch.nn.modules.loss import _Loss


def compute_harmonic_alphas(n, use_diagamma=True):
    if use_diagamma:
        return [digamma(n + 1) - digamma(n - rank + 1) for rank in range(1, n + 1)]
    else:
        return np.cumsum(1 / (n - np.arange(n)))


def compute_ln_alphas(n):
    # alphas in ln formular.(also used when compute population AURC_a) 
    alphas = [0] * n
    for rank in range(1, n + 1):
        alpha = -np.log(1 - rank / (n + 1))
        alphas[rank - 1] = alpha
    return alphas


def sele_alphas(n):
    alphas = [0] * n
    for rank in range(1, n + 1):
        alpha = (rank / n) 
        alphas[rank - 1] = alpha
    return alphas


def entropy(x):
    eps = 1e-9 
    x = torch.clamp(x, min=eps, max=1 - eps).to(x.device)  
    return -torch.sum(torch.log(x) * x, dim=1)


def top12_margin(x):
    """top-1 - top-2"""
    values, _ = torch.topk(x, k=2, dim=-1)
    if x.ndim == 1:
        return values[0] - values[1]
    return values[:, 0] - values[:, 1]


def gini_score(x):
    score = 1 - torch.norm(x, dim=1, p=2)**2
    return score


def get_score_function(name):
    if name == "MSP":
        return lambda x: torch.max(F.softmax(x, dim=1),dim=1).values
    elif name == "NegEntropy":
        return lambda x: -entropy(x)
    elif name == "SoftmaxMargin":
        return lambda x: top12_margin(F.softmax(x, dim=1))
    elif name == "MaxLogit":
        return lambda x: torch.max(x, dim=1).values
    elif name == "l2_norm":
        return lambda x: torch.norm(F.softmax(x, dim=1), dim=1, p=2)
    elif name == "NegGiniScore":
        return lambda x: -gini_score(F.softmax(x, dim=1))
    else:
        raise ValueError(f"Unknown select function: {name}")


class BaseAURCLoss(_Loss):
    def __init__(self, criterion=torch.nn.CrossEntropyLoss(), score_function="MSP", batch_size=128, reduction='mean', alpha_fn=None):
        super().__init__(reduction=reduction)
        self.criterion = criterion
        self.criterion.reduction = 'none'
        self.reduction = reduction
        self.score_func = get_score_function(score_function)
        self.alphas = alpha_fn(batch_size) if alpha_fn is not None else None
        self.batch_size = batch_size

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        with torch.no_grad():
            confidence = self.score_func(input)
            indices_sorted = torch.argsort(confidence, descending=False)
            reverse_indices = torch.argsort(indices_sorted)
            reordered_alphas = torch.tensor(self.alphas, dtype=input.dtype, device=input.device)[reverse_indices]
        losses = self.criterion(input, target) * reordered_alphas
        if self.reduction == 'mean':
            return torch.mean(losses)
        elif self.reduction == 'sum':
            return torch.sum(losses)
        else:
            return losses


class EmAURCLoss(BaseAURCLoss):
    def __init__(self, criterion=torch.nn.CrossEntropyLoss(), batch_size=128, score_function="MSP", reduction='mean'):
        super().__init__(criterion, score_function, batch_size, reduction, compute_harmonic_alphas)

class LnAURCLoss(BaseAURCLoss):
    def __init__(self, criterion=torch.nn.CrossEntropyLoss(), batch_size=128, score_function="MSP", reduction='mean'):
        super().__init__(criterion, score_function, batch_size, reduction, compute_ln_alphas)

class SeleLoss(BaseAURCLoss):
    def __init__(self, criterion=torch.nn.CrossEntropyLoss(), batch_size=128, score_function="MSP", reduction='mean'):
        super().__init__(criterion, score_function, batch_size, reduction, sele_alphas)

