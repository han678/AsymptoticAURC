from __future__ import print_function, absolute_import

import numpy as np

from utils.loss import compute_ln_alphas, compute_harmonic_alphas, sele_alphas

__all__ = ["get_AURC", "get_em_AURC", "get_sele_score", "get_ln_AURC"]

def get_AURC(residuals, confidence):
    '''
    AURC proposed by Geifman

    Args:
        residuals (list): The residuals of the model predictions.

    Returns:
        float: The AURC.      
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
    return AUC

def get_em_AURC(residuals, confidence):
    m = len(residuals)
    idx_sorted = np.argsort(confidence)
    temp1 = residuals[idx_sorted]
    alphas = compute_harmonic_alphas(n=m)
    mc_AURC = sum(np.array(temp1) * alphas / m)
    return mc_AURC

def get_ln_AURC(residuals, confidence):
    '''
    Compute the AURC in ln formular given infinite samples.

    Args:
        residuals (list): The residuals of the model predictions.
        confidence (list): The confidence of the model predictions.

    Returns:
        float: AURC estimator with ln weights.         

    '''
    m = len(residuals)
    idx_sorted = np.argsort(confidence)
    temp1 = residuals[idx_sorted]
    alphas = compute_ln_alphas(n=m)
    ln_AURC = sum(np.array(temp1) * alphas / m)
    return ln_AURC

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
    score = sum(np.array(temp1) * alphas / m)
    return score
