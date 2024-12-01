import argparse
import json
import os
import random

import numpy as np
import torch
from cifar import get_batch_sample_results_from_logits, get_full_data_results_from_logits, get_logits_and_labels
from utils.loaders import CustomTensorDataset
from visualize.plot_csf import plot_csf
from visualize.plot_statistic_metrics import plot_aurc_metrics, plot_bias, plot_mse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_trained_models_outputs', type=str, default='results/Amazon')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20) # 21
    parser.add_argument('--output_path', type=str, default='outputs')

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args_dict = vars(args)
    path = args_dict['path_to_trained_models_outputs']
    model_names = ['bert','distill_bert', 'distill_roberta', 'roberta']
    device = args_dict['device']
    num_workers = args_dict['num_workers']
    output_path = args_dict['output_path']
    subdirs = ['estimator', 'mse', 'bias', 'csf']
    for subdir in subdirs:
        os.makedirs(os.path.join(output_path, subdir), exist_ok=True)

    criterion = torch.nn.CrossEntropyLoss()
    if not os.path.isdir(output_path):
        os.makedirs(output_path, exist_ok = True)
    metrics_name = ['mc_aurc', 'sele', '2sele',  'true_aurc', '01_mc_aurc', '01_sele', '01_2sele', 'e_aurc', '01_true_aurc']
    results = {}
    for model_name in model_names:
        results[model_name] = {}
        dist_mc_aurc, dist_sele, dist_2sele, dist_geifman_aurc, dist_01_mc_aurc, dist_01_sele, dist_01_2sele = ({} for _ in range(7))
        folder_name = f'amazon_{model_name}'
        path_to_root_folder = os.path.join(path, folder_name)
        print("Root folder path: ", path_to_root_folder)

        # Collect results
        with open(os.path.join(path_to_root_folder, f'target_agg.json'), 'r') as file:
            preds_dict  = json.load(file)
        logits_test, labels_test = get_logits_and_labels(preds_dict, 'y_logits', 'y_true')
        batch_size_list = [8, 16, 32, 64, 128, 256, 512, 1024]
        test_set = CustomTensorDataset(tensors=(logits_test, labels_test), transform=None)
        for batch_size in batch_size_list:
            results[model_name][str(batch_size)] = {}
            loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            batch_sample_results = get_batch_sample_results_from_logits(loader, device)
            results[model_name][str(batch_size)].update(batch_sample_results)
        full_data_results = get_full_data_results_from_logits(loader, criterion, device, score_func_name="MSP")
        results[model_name].update(full_data_results)
        figs_path1 = os.path.join(output_path, f'estimator/amazon_{model_name}')
        figs_path2 = os.path.join(output_path, f'mse/amazon_{model_name}')
        figs_path3 = os.path.join(output_path, f'bias/amazon_{model_name}')

        # make plots for aurc metrics
        plot_aurc_metrics(results[model_name], batch_size_list=batch_size_list, figs_path=figs_path1)
        plot_mse(results[model_name], batch_size_list=batch_size_list, figs_path=figs_path2)
        plot_bias(results[model_name], batch_size_list=batch_size_list, figs_path=figs_path3)   

        # make plots for csf metrics
        score_function_names = ["MSP", "NegEntropy", "SoftmaxMargin", "MaxLogit", "l2_norm", "NegGiniScore"]
        results = {}
        batch_size = 128
        for score_function_name in score_function_names:
            results[score_function_name] = {}
            loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            batch_sample_results = get_batch_sample_results_from_logits(loader, device, score_func_name=score_function_name, return_all=True)
            results[score_function_name].update(batch_sample_results)
            full_data_results = get_full_data_results_from_logits(loader, criterion, device, score_func_name=score_function_name)
            results[score_function_name].update(full_data_results)
        figs_path = os.path.join(output_path, f'csf/amazon_{model_name}')
        plot_csf(results, score_function_names, figs_path=figs_path)
