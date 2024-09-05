import numpy as np
import torch
from scipy.special import digamma
from torch import Tensor
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
from collections import OrderedDict
import torch.nn as nn

def exact_alphas(n, use_diagamma=False):
    alphas = [0] * n
    if use_diagamma:
        for rank in range(1, n + 1):
            # Using the digamma function for approximations
            alphas[rank - 1] = (digamma(n + 1) - digamma(n - rank + 1)) / n
    else:
        # Compute alphas using the exact method
        cumulative_sum = 0
        for rank in range(1, n + 1):
            cumulative_sum += 1 / (n - (rank - 1))
            alpha = cumulative_sum / n
            alphas[rank - 1] = alpha
    return alphas


def approx_alphas(n):
    alphas = [0] * n
    for rank in range(1, n + 1):
        if rank == n:
            alpha = (np.log(n + 1) + 0.577) / n  # 0.577 is the Euler–Mascheroni constant
            # alpha = -np.log(1 - rank / n + 1e-10) / n
        else:
            alpha = -np.log(1 - rank / n) / n
        alphas[rank - 1] = alpha
    return alphas

def entropy(x):
    logits = F.softmax(x, dim=1)
    return torch.sum(torch.log(logits) * logits, dim=1) 


def get_score_function(name):
    if name == "softmax":
        return lambda x: torch.max(F.softmax(x, dim=1),dim=1).values
    elif name == "neg_entropy":
        return lambda x: -entropy(x)
    elif name == "l2_norm":
        return lambda x: -torch.norm(F.softmax(x, dim=1), dim=1, p=2)
    else:
        raise ValueError(f"Unknown select function: {name}")

class AURCLoss(nn.Module):
    def __init__(self, criterion=torch.nn.CrossEntropyLoss(), score_func="softmax", approx=True) -> None:
        super(AURCLoss, self).__init__()
        self.approx = approx
        self.criterion = criterion
        self.criterion.reduction = 'none'
        self.score_func = get_score_function(score_func)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        n = len(target)
        alphas = approx_alphas(n) if self.approx else exact_alphas(n)
        with torch.no_grad():
            confidence = self.score_func(input)
            indices_sorted = torch.argsort(confidence, descending=False)
            reverse_indices = torch.argsort(indices_sorted)
            reordered_alphas = torch.tensor(alphas, dtype=input.dtype, device=input.device)[reverse_indices]
        # sorted_loss = losses[indices_sorted]
        # loss2 = sorted_loss * torch.tensor(alphas, dtype=input.dtype, device=input.device)
        # diff = torch.sort(loss)[0]-torch.sort(loss2)[0]
        losses = self.criterion(input, target)
        loss = losses * reordered_alphas
        return torch.mean(loss)

class SeleLoss(nn.Module):
    # SELE loss from paper Optimal strategies for reject option classifiers https://arxiv.org/pdf/2101.12523
    def __init__(self, criterion=torch.nn.CrossEntropyLoss(), score_func="softmax") -> None:
        super(SeleLoss, self).__init__()
        self.criterion = criterion
        self.criterion.reduction = 'none'
        self.score_func = get_score_function(score_func)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n = len(target)
        alphas = [(n- rank + 1)/(n**2) for rank in range(1, n + 1)]
        with torch.no_grad():
            confidence = self.score_func(input)
            indices_sorted = torch.argsort(confidence, descending=False)
            reverse_indices = torch.argsort(indices_sorted)
            reordered_alphas = torch.tensor(alphas, dtype=input.dtype, device=input.device)[reverse_indices]
        losses = self.criterion(input, target)
        loss = losses * reordered_alphas
        return torch.mean(loss)

# from: https://github.com/ondrejbohdal/meta-calibration/blob/4dc752c41251b18983a4de22a3961ddcc26b122a/Metrics
class ECELoss(nn.Module):
    """
    Compute ECE (Expected Calibration Error)
    """
    def __init__(self, p=1, n_bins=15):
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.p = p
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += prop_in_bin * (torch.abs(avg_confidence_in_bin - accuracy_in_bin))**self.p
        return ece