from __future__ import print_function, absolute_import

import collections
import math
import torch
import torch.nn.functional as F
import numpy as np

from utils.loss import approx_alphas, exact_alphas

__all__ = ["geifman_AURC", "alpha_AURC"]

def geifman_AURC(residuals, confidence):
    # only valid for 0/1 loss
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
    coverage = [point[0] for point in curve]
    risk = [point[1] for point in curve]
    result={"auc": AUC, "EAURC": EAURC, "geifman_AURC": kappa_star_aurc}
    return result

def alpha_AURC(residuals, confidence, approx=False, return_dict=True):
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
    if approx:
        alphas = approx_alphas(n=m)
    else:
        alphas = exact_alphas(n=m, use_diagamma=False)
    alpha_AURC = sum(np.array(temp1) *alphas)
    coverage = [point[0] for point in curve]
    risk = [point[1] for point in curve]
    if return_dict:
        result={"risk": risk, "coverage": coverage, "auc": AUC}
        if approx:
            result["alpha_approx_AURC"] = alpha_AURC
        else:
            result["alpha_exact_AURC"] = alpha_AURC
        return result
    else:
        return alpha_AURC

def get_brier_score(confidences, targets):
    """
    Compute the Brier score for probabilistic predictions using NumPy.
    
    Args:
        confidences (numpy.ndarray): The probabilities of each class. Shape (N, C) where C is number of classes.
        targets (numpy.ndarray): The one-hot encoded true labels. Shape (N, C) where C is number of classes.
        
    Returns:
        float: The Brier score for the predictions.
    """
    differences = confidences - targets
    squared_differences = differences ** 2
    score = np.mean(squared_differences)
    return score


def get_ece_score(confidences, targets, n_bins=15):
    """
    Calculate the Expected Calibration Error (ECE).
    
    Args:
        confidences (np.ndarray): Predicted probabilities or confidence scores, shape (N, C), where C is number of classes.
        targets (np.ndarray): True labels or one-hot encoded labels, shape (N,) or (N, C).
        n_bins (int): Number of bins to use for ECE calculation.

    Returns:
        float: The ECE score.
    """
    # Convert to NumPy arrays
    confidences = np.asarray(confidences)
    targets = np.asarray(targets)
    
    # If targets are one-hot encoded, convert them to class indices
    if targets.ndim > 1:
        targets = np.argmax(targets, axis=1)
    
    # Get the predicted class indices and their probabilities
    predicted_classes = np.argmax(confidences, axis=1)
    predicted_probs = np.max(confidences, axis=1)  # Using maximum confidence as predicted probability
    
    # Initialize bins
    accuracy_bins = np.zeros(n_bins)
    confidence_bins = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    
    # Calculate the binning step size
    for bin_index in range(n_bins):
        lower_bound, upper_bound = bin_index / n_bins, (bin_index + 1) / n_bins
        
        for i in range(confidences.shape[0]):
            if lower_bound < predicted_probs[i] <= upper_bound:
                bin_counts[bin_index] += 1
                if predicted_classes[i] == targets[i]:
                    accuracy_bins[bin_index] += 1
                confidence_bins[bin_index] += predicted_probs[i]
        
        # Calculate mean accuracy and confidence for non-empty bins
        if bin_counts[bin_index] != 0:
            accuracy_bins[bin_index] /= bin_counts[bin_index]
            confidence_bins[bin_index] /= bin_counts[bin_index]
    
    # Compute the ECE score
    ece = np.sum(bin_counts * np.abs(accuracy_bins - confidence_bins)) / np.sum(bin_counts)
    
    return ece