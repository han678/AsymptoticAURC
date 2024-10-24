from __future__ import print_function, absolute_import

import collections
import math
import torch
import torch.nn.functional as F
import numpy as np

from utils.loss import compute_asy_alphas, compute_alphas, sele_alphas

__all__ = ["get_geifman_AURC", "get_mc_AURC", "get_sele_score", "get_asy_AURC"]

def get_geifman_AURC(residuals):
    '''
    An approximation for AURC proposed by Geifman (only valid for 0/1 loss)

    Args:
        residuals (list): The residuals of the model predictions.

    Returns:
        float: The AURC approximation.      
    '''     
    err = np.mean(residuals) 
    kappa_star_aurc = err + (1 - err) * (np.log(1 - err))
    return kappa_star_aurc

def get_mc_AURC(residuals, confidence):
    m = len(residuals)
    idx_sorted = np.argsort(confidence)
    temp1 = residuals[idx_sorted]
    alphas = compute_alphas(n=m)
    asy_AURC = sum(np.array(temp1) *alphas)
    return asy_AURC

def get_asy_AURC(residuals, confidence):
    '''

    Compute the asymptotic AURC

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