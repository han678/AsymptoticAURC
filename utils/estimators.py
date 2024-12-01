from __future__ import print_function, absolute_import

import collections
import math
import torch
import torch.nn.functional as F
import numpy as np

from utils.loss import compute_asy_alphas, compute_mc_alphas, sele_alphas

__all__ = ["get_EAURC", "get_mc_AURC", "get_sele_score", "get_asy_AURC"]

def get_EAURC(residuals, confidence):
    '''
    EAURC proposed by Geifman (only valid for 0/1 loss)

    Args:
        residuals (list): The residuals of the model predictions.

    Returns:
        float: The EAURC.      
    '''  
        curve = []
    m = len(residuals)
    idx_sorted = np.argsort(confidence)
    temp1 = residuals[idx_sorted]
    cov = len(temp1)
    acc = sum(temp1)
    curve.append((cov/ m, acc / len(temp1)))
    for i in range(0, len(idx_sorted)-1):
        cov = cov-1
        acc = acc-residuals[idx_sorted[i]]
        curve.append((cov / m, acc /(m-i)))
    AUC = sum([a[1] for a in curve])/len(curve)
    err = np.mean(residuals)
    kappa_star_aurc = err + (1 - err) * (np.log(1 - err))
    EAURC = AUC-kappa_star_aurc
    return EAURC

def get_mc_AURC(residuals, confidence):
    m = len(residuals)
    idx_sorted = np.argsort(confidence)
    temp1 = residuals[idx_sorted]
    alphas = compute_mc_alphas(n=m)
    asy_AURC = sum(np.array(temp1) *alphas)
    return asy_AURC

def get_asy_AURC(residuals, confidence):
    '''

    Compute the asymptotic AURC for infinite samples

    Args:
        residuals (list): The residuals of the model predictions.
        confidence (list): The confidence of the model predictions.

    Returns:
        float: The asymptotic AURC.         

    '''
    m = len(residuals)
    idx_sorted = np.argsort(confidence)
    temp1 = residuals[idx_sorted]
    alphas = compute_asy_alphas(n=m)
    asy_AURC = sum(np.array(temp1) *alphas)
    return asy_AURC

def get_sele_score(residuals, confidence):
    '''
    Compute the SELE score

    Args:
        residuals (list): The residuals of the model predictions.
        confidence (list): The confidence of the model predictions.

    Returns:
        float: The SELE score.      
    '''

    m = len(residuals)
    idx_sorted = np.argsort(confidence)
    temp1 = residuals[idx_sorted]
    alphas = sele_alphas(n=m)
    score = sum(np.array(temp1) *alphas)
    return score
