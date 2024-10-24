import argparse
import glob
import os
import random
import statistics
import numpy as np
import torch
from utils.loaders import CustomTensorDataset
from visualize.plot_statistic_metrics import plot_aurc_metrics, plot_bias, plot_mae, plot_mse
import statistics

import numpy as np
import torch
import torch.nn.functional as F
from utils.loss import get_score_function
from utils.estimators import get_asy_AURC, get_geifman_AURC, get_mc_AURC, get_sele_score

def calculate_mean_variance(data):
    mean = statistics.mean(data)
    std = statistics.stdev(data)
    return {'mean': mean, 'std': std}

def zero_one_loss(predictions, targets):
    return np.argmax(predictions, 1) != np.argmax(targets, 1)

def cross_entropy_loss(logits, targets):
    """Compute cross-entropy loss between prediction probabilities and targets."""
    ce_loss = -np.sum(targets * np.log(logits), axis=1) 
    return ce_loss

def get_logits_and_labels(preds_dict, logits_key, labels_key):
    logits = preds_dict[logits_key]
    labels = preds_dict[labels_key]
    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits)
        labels = torch.tensor(labels)
    return logits, labels


def get_batch_sample_results_from_logits(test_loader, device, score_func_name="MSP", return_all=False):
    """Load and evaluate the batch sample results for CIFAR/Amazon datasets."""
    all_mc_aurc, all_sele, all_geifman_aurc, all_01_mc_aurc, all_01_sele = [], [], [], [], []
    EPS = 1e-7
    results= {}
    score_func = get_score_function(score_func_name)
    for _, (logits, targets) in enumerate(test_loader):
        logits, targets = logits.to(device), F.one_hot(targets, num_classes=logits.shape[1]).to(device).cpu().numpy()
        scores = F.softmax(logits, dim=1)
        scores = torch.clamp(scores, min=EPS, max=1 - EPS).cpu().numpy()
        confidences = score_func(logits).cpu().numpy()
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

def get_full_data_results_from_logits(test_loader, criterion, device, score_func_name="softmax"):
    """Evaluate the model on the test set for CIFAR/Amazon dataset."""
    all_scores, all_targets, all_confidences = [], [], []
    total_correct_1 = 0
    total_correct_5 = 0
    score_func = get_score_function(score_func_name)
    EPS = 1e-7
    with torch.no_grad():
        for _, (logits, targets) in enumerate(test_loader):
            logits, targets = logits.to(device), targets.to(device)
            loss = criterion(logits, targets)
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
    result = {"test_acc_1": top1_acc, "test_acc_5": top5_acc, "test_loss": loss}
    result["01_asy_aurc"] = get_asy_AURC(residuals=loss1, confidence=all_confidences)
    result["asy_aurc"] = get_asy_AURC(residuals=loss2, confidence=all_confidences)
    result["01_true_aurc"] = get_mc_AURC(residuals=loss1, confidence=all_confidences)
    result["true_aurc"] = get_mc_AURC(residuals=loss2, confidence=all_confidences)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_trained_models', type=str, default='results/cifar_outputs')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=24) 
    parser.add_argument('--output_path', type=str, default='outputs')

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args_dict = vars(args)
    path = args_dict['path_to_trained_models']
    device = args_dict['device']
    num_workers = args_dict['num_workers']
    output_path = args_dict['output_path']
    criterion = torch.nn.CrossEntropyLoss()
    if not os.path.isdir(output_path):
        os.makedirs(output_path, exist_ok = True)
    subdirs = ['estimator', 'mse', 'bias', 'mae']
    for subdir in subdirs:
        os.makedirs(os.path.join(output_path, subdir), exist_ok=True)

    metrics_name = ['mc_aurc', 'sele', 'true_aurc', 'asy_aurc', '01_mc_aurc', '01_sele', 'geifman_aurc', '01_true_aurc', '01_asy_aurc']
    model_names = ['PreResNet20','PreResNet56', 'PreResNet110', 'PreResNet164', 'WideResNet28x10', 'VGG16BN']
    datasets = ['cifar10', 'cifar100']
    seeds = [5, 10, 21, 42, 84]
    for dataset in datasets: 
        results = {}
        for model_name in model_names: 
            results[model_name] = {}
            dist_mc_aurc, dist_sele, dist_2sele, dist_geifman_aurc, dist_01_mc_aurc, dist_01_sele, dist_01_2sele = ({} for _ in range(7))
            for seed in seeds:
                results[model_name][str(seed)] = {}
                folder_name = f'{dataset}_{model_name}_250_{seed}_output_{seed}'
                path_to_root_folder = os.path.join(path, folder_name)
                print("Root folder path: ", path_to_root_folder)
                cache_path = os.path.join(path_to_root_folder, '*', '*', '*cache*')
                matches = glob.glob(cache_path, recursive=True)
                if len(matches) == 0:
                    print('Cache folder not found')
                    continue
                path_to_cache_folder = matches[0]
                print("Cache folder path:", path_to_cache_folder)
                # Find the epoch of best model
                path_to_best_model = glob.glob(os.path.join(path_to_cache_folder, 'epoch_*'))[0]
                best_model_filename = os.path.basename(path_to_best_model)
                epoch = best_model_filename.split("_")[1]
                print(f"Best model saved at epoch: {epoch}")

                # Collect results
                preds_dict = torch.load(os.path.join(path_to_cache_folder, f'preds_{epoch}.pth'))
                logits_test, labels_test = get_logits_and_labels(preds_dict, 'preds_test', 'gt_test')
                batch_size_list = [8, 16, 32, 64, 128, 256, 512, 1024]
                test_set = CustomTensorDataset(tensors=(logits_test, labels_test), transform=None)
                for batch_size in batch_size_list:
                    results[model_name][str(seed)][str(batch_size)] = {}
                    loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
                    batch_sample_results = get_batch_sample_results_from_logits(loader, device)
                    results[model_name][str(seed)][str(batch_size)].update(batch_sample_results)
                full_data_results = get_full_data_results_from_logits(loader, criterion, device, score_func_name="MSP")
                results[model_name][str(seed)].update(full_data_results)
                figs_path1 = os.path.join(output_path, f'estimator/{dataset}_{model_name}_{seed}')
                figs_path2 = os.path.join(output_path, f'mse/{dataset}_{model_name}_{seed}')
                figs_path3 = os.path.join(output_path, f'bias/{dataset}_{model_name}_{seed}')
                plot_aurc_metrics(results[model_name][str(seed)], batch_size_list=batch_size_list, figs_path=figs_path1)
                plot_mse(results[model_name][str(seed)], batch_size_list=batch_size_list, figs_path=figs_path2)
                plot_bias(results[model_name][str(seed)], batch_size_list=batch_size_list, figs_path=figs_path3)

            # Calculate mean and variance of the abs(bias) terms across different models
            for batch_size in batch_size_list:
                dist_mc_aurc[str(batch_size)], dist_sele[str(batch_size)], dist_2sele[str(batch_size)], dist_geifman_aurc[str(batch_size)], dist_01_mc_aurc[str(batch_size)], dist_01_sele[str(batch_size)], dist_01_2sele[str(batch_size)] = ([] for _ in range(7))
                for seed in seeds:
                    dist_mc_aurc[str(batch_size)].append(abs(results[model_name][str(seed)][str(batch_size)]["mc_aurc"]["mean"]-results[model_name][str(seed)]["true_aurc"]))
                    dist_sele[str(batch_size)].append(abs(results[model_name][str(seed)][str(batch_size)]["sele"]["mean"]-results[model_name][str(seed)]["true_aurc"]))
                    dist_2sele[str(batch_size)].append(abs(2 * results[model_name][str(seed)][str(batch_size)]["sele"]["mean"]-results[model_name][str(seed)]["true_aurc"]))
                    dist_geifman_aurc[str(batch_size)].append(abs(results[model_name][str(seed)][str(batch_size)]["geifman_aurc"]["mean"]-results[model_name][str(seed)]["01_true_aurc"]))
                    dist_01_mc_aurc[str(batch_size)].append(abs(results[model_name][str(seed)][str(batch_size)]["01_mc_aurc"]["mean"]-results[model_name][str(seed)]["01_true_aurc"]))
                    dist_01_sele[str(batch_size)].append(abs(results[model_name][str(seed)][str(batch_size)]["01_sele"]["mean"]-results[model_name][str(seed)]["01_true_aurc"]))
                    dist_01_2sele[str(batch_size)].append(abs(2 * results[model_name][str(seed)][str(batch_size)]["01_sele"]["mean"]-results[model_name][str(seed)]["01_true_aurc"]))
            all_seed_results = {}
            dict_list = [dist_mc_aurc, dist_sele, dist_2sele, dist_geifman_aurc, dist_01_mc_aurc, dist_01_sele, dist_01_2sele]
            for dictionary in dict_list:
                mean, std = [], []
                dict_name = [k for k, v in locals().items() if v is dictionary and k.startswith('dist')][0].replace('dist_', '')
                if dict_name not in all_seed_results:
                    all_seed_results[dict_name] = {}
                for key, values in dictionary.items():
                    mean.append(round(statistics.mean(values),4)) # mean of the bias term
                    std.append(round(statistics.stdev(values),4)) 
                all_seed_results[dict_name] = {'mean': mean, 'std': std}
            print(f"({dataset}) {model_name}: \n ")
            print(f"mc_aurc: {all_seed_results['mc_aurc']['mean']} ({all_seed_results['mc_aurc']['std']})")
            print(f"sele: {all_seed_results['sele']['mean']} ({all_seed_results['sele']['std']})")
            print(f"2*sele: {all_seed_results['2sele']['mean']} ({all_seed_results['2sele']['std']})")
            print(f"geifman_aurc: {all_seed_results['geifman_aurc']['mean']} ({all_seed_results['geifman_aurc']['std']})")    
            print(f"01_mc_aurc: {all_seed_results['01_mc_aurc']['mean']} ({all_seed_results['01_mc_aurc']['std']})")
            print(f"01_sele: {all_seed_results['01_sele']['mean']} ({all_seed_results['01_sele']['std']})")
            print(f"01_2sele: {all_seed_results['01_2sele']['mean']} ({all_seed_results['01_2sele']['std']})")
            print("\n ")
            figs_path3 = os.path.join(output_path, f'mae/{dataset}_{model_name}')
            plot_mae(all_seed_results, batch_size_list=batch_size_list, figs_path=figs_path3)
        # print(results)