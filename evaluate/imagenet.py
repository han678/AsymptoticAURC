import timm 

import argparse
import os
import random
import numpy as np
import torch
from cifar import calculate_mean_variance, cross_entropy_loss, zero_one_loss
from utils.loaders import prepare_dataset
from fastai.vision.all import *
from visualize import plot_csf
from visualize.plot_statistic_metrics import plot_aurc_metrics, plot_bias, plot_mse
import torch.nn.functional as F
from utils.loss import get_score_function
from utils.estimators import get_asy_AURC, get_EAURC, get_mc_AURC, get_sele_score

def get_batch_sample_results(model, test_loader, device, score_func_name="softmax", return_all=False):
    """Load and evaluate the batch sample results for ImageNet dataset."""
    all_mc_aurc, all_sele, all_geifman_aurc, all_01_mc_aurc, all_01_sele = [], [], [], [], []
    EPS = 1e-7
    results= {}
    score_func = get_score_function(score_func_name)
    with torch.no_grad():
        for _, (images, targets) in enumerate(test_loader):
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            scores = torch.clamp(F.softmax(logits, dim=1), min=EPS, max=1 - EPS).cpu().numpy()
            confidences = score_func(logits).cpu().numpy()
            targets = F.one_hot(targets, num_classes=logits.shape[1]).to(device).cpu().numpy()
            loss1 = zero_one_loss(scores, targets)
            loss2 = cross_entropy_loss(scores, targets)
            geifman_aurc = get_geifman_AURC(residuals=loss1)
            mc_aurc_01 = get_mc_AURC(residuals=loss1, confidence=confidences)
            sele_01 = get_sele_score(residuals=loss1, confidence=confidences)
            mc_aurc = get_mc_AURC(residuals=loss2, confidence=confidences)
            sele = get_sele_score(residuals=loss2, confidence=confidences)
            all_mc_aurc.append(mc_aurc)
            all_sele.append(sele)
            all_geifman_aurc.append(geifman_aurc)
            all_01_mc_aurc.append(mc_aurc_01)
            all_01_sele.append(sele_01)
    if not return_all:
        results["mc_aurc"] = calculate_mean_variance(all_mc_aurc)
        results["sele"] = calculate_mean_variance(all_sele)
        results["2sele"] = calculate_mean_variance([x * 2 for x in all_sele])
        results["geifman_aurc"] = calculate_mean_variance(all_geifman_aurc)
        results["01_mc_aurc"] = calculate_mean_variance(all_01_mc_aurc)
        results["01_sele"] = calculate_mean_variance(all_01_sele)
        results["01_2sele"] = calculate_mean_variance([x * 2 for x in all_01_sele])
    else:
        results["mc_aurc"] = all_mc_aurc
        results["sele"] = all_sele
        results["2sele"] = [x * 2 for x in all_sele]
        results["geifman_aurc"] = all_geifman_aurc
        results["01_mc_aurc"] = all_01_mc_aurc
        results["01_sele"] = all_01_sele
        results["01_2sele"] = [x * 2 for x in all_01_sele]
    return results

def get_full_data_results(model, test_loader, device, score_func_name="softmax"):
    """Evaluate the model on the test set for ImageNet dataset."""
    all_scores, all_targets, all_confidences = [], [], []
    total_correct_1 = 0
    total_correct_5 = 0
    score_func = get_score_function(score_func_name)
    EPS = 1e-7
    with torch.no_grad():
        for _, (images, targets) in enumerate(test_loader):
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            # loss = criterion(logits, targets)
            scores = F.softmax(logits, dim=1)
            scores = torch.clamp(scores, min=EPS, max=1 - EPS)
            confidence = score_func(logits)
            _, pred = scores.topk(5, 1, largest=True, sorted=True)
            correct = pred.eq(targets.view(targets.size(0), -1).expand_as(pred)).float()
            total_correct_1 += correct[:, :1].sum().item()
            total_correct_5 += correct[:, :5].sum().item()
            all_scores.append(scores.cpu().numpy())
            all_targets.append(F.one_hot(targets, num_classes=scores.shape[1]).to(device).cpu().numpy())
            all_confidences.append(confidence.cpu().numpy())
    all_scores = np.vstack(all_scores)
    all_targets = np.vstack(all_targets)
    all_confidences = np.hstack(all_confidences)
    loss1 = zero_one_loss(all_scores, all_targets)
    loss2 = cross_entropy_loss(all_scores, all_targets)
    top1_acc = 100. * total_correct_1 / len(test_loader.dataset)
    top5_acc = 100. * total_correct_5 / len(test_loader.dataset)
    result = {"test_acc_1": top1_acc, "test_acc_5": top5_acc}
    result["01_true_aurc"] = get_mc_AURC(residuals=loss1, confidence=all_confidences)
    result["asy_aurc"] = get_asy_AURC(residuals=loss2, confidence=all_confidences)
    result["01_asy_aurc"] = get_asy_AURC(residuals=loss1, confidence=all_confidences)
    result["true_aurc"] = get_mc_AURC(residuals=loss2, confidence=all_confidences)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--data_dir', type=str, default='./data/ILSVRC2012')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20)
    parser.add_argument('--dataset', type=str, default='imagenet')
    parser.add_argument('--output_path', type=str, default='outputs')

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args_dict = vars(args)
    model_names = ['vit_small_patch16_224','vit_large_patch16_224', 'swin_tiny_patch4_window7_224', 'swin_base_patch4_window7_224']
    device = args_dict['device']
    num_workers = args_dict['num_workers']
    output_path = args_dict['output_path']

    criterion = torch.nn.CrossEntropyLoss()
    score_func_name="MSP"
    if not os.path.isdir(output_path):
        os.makedirs(output_path, exist_ok=True)

    subdirs = ['estimator', 'mse', 'bias', 'csf']
    for subdir in subdirs:
        os.makedirs(os.path.join(output_path, subdir), exist_ok=True)
    metrics_name = ['mc_aurc', 'sele', '2sele',  'true_aurc', '01_mc_aurc', '01_sele', '01_2sele', 'geifman_aurc', '01_true_aurc']
    results = {}
    batch_size_list = [16, 32, 64, 128, 256, 512, 1024]
    
    for model_name in model_names:
        results[model_name] = {}
        model = timm.create_model(model_name, pretrained=True).to(device)
        model.eval()
        for batch_size in batch_size_list:
            print(f'batch size: {batch_size}')
            results[model_name][str(batch_size)] = {}
            loader = prepare_dataset(args.dataset, batch_size=batch_size, load_train=False, num_workers=4, data_dir=args.data_dir) 
            batch_sample_results = get_batch_sample_results(model, loader, device, score_func_name)
            results[model_name][str(batch_size)].update(batch_sample_results)
        full_data_results = get_full_data_results(model, loader, device, score_func_name)
        results[model_name].update(full_data_results)
        figs_path = os.path.join(output_path, f'estimator/imagenet_{model_name}')
        plot_aurc_metrics(results[model_name], batch_size_list=batch_size_list, figs_path=figs_path)
        figs_path2 = os.path.join(output_path, f'mse/imagenet_{model_name}')
        plot_mse(results[model_name], batch_size_list=batch_size_list, figs_path=figs_path2)
        figs_path3 = os.path.join(output_path, f'bias/imagenet_{model_name}')
        plot_bias(results[model_name], batch_size_list=batch_size_list, figs_path=figs_path3)
        print(results)

        score_function_names = ["MSP", "NegEntropy", "SoftmaxMargin", "MaxLogit", "l2_norm", "NegGiniScore"]
        results = {}
        batch_size = 128
        for score_function_name in score_function_names:
            results[score_function_name] = {}
            loader = prepare_dataset(args.dataset, batch_size=batch_size, load_train=False, num_workers=4, data_dir=args.data_dir) 
            batch_sample_results = get_batch_sample_results(model, loader, device, score_func_name=score_function_name, return_all=True)
            results[score_function_name].update(batch_sample_results)
            full_data_results = get_full_data_results(model, loader, device, score_func_name=score_function_name)
            results[score_function_name].update(full_data_results)
        figs_path = os.path.join(output_path, f'csf/imagenet_{model_name}')
        plot_csf(results, score_function_names, figs_path=figs_path)
        
