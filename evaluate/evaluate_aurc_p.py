import argparse
import glob
import json
import os
import random
import timm 

from matplotlib import pyplot as plt
import numpy as np
import torch
from cifar import get_full_data_results_from_logits, get_logits_and_labels
from evaluate_imagenet import get_full_data_results
from utils.loaders import CustomTensorDataset, prepare_dataset
from scipy import stats

def plot_population_aurc(results, data_names, figs_path):
    """Plot the population AURC."""
    all_values = []
    loss_types = ["01", "ce"]
    for loss_type in loss_types:
        list1 = []
        list2 = []
        fig, ax = plt.subplots(figsize=(9, 9))
        for dataset in data_names:
            x = []
            y = []
            for model_name in results[dataset].keys():
                x_value = results[dataset][model_name]['01_true_aurc'] if loss_type == '01' else results[dataset][model_name]['true_aurc']
                y_value = results[dataset][model_name]['01_asy_aurc'] if loss_type == '01' else results[dataset][model_name]['asy_aurc']
                x.append(x_value)
                y.append(y_value)
                list1.append(x_value)
                list2.append(y_value)
                all_values.extend([x_value, y_value]) 
            ax.scatter(x, y, label=dataset)
        ax.set_xlabel(r'$AURC_p$', fontsize=22)
        ax.set_ylabel(r'$AURC_{a}$', fontsize=22)
        if all_values:
            min_val = min(all_values)
            max_val = max(all_values)
            ax.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='darkgray', label='$AURC_p = AURC_{a}$')  
        ax.legend(fontsize=22)
        fig.savefig(os.path.join(figs_path, f'aurc_{loss_type}.png'))
        plt.close(fig)
        t_statistic, p_value = stats.ttest_ind(list1, list2)
        with open(f't_test_{loss_type}.txt', 'w') as file:
            file.write(f"loss type: {loss_type}\n")
            file.write(f"T-statistic: {t_statistic}\n")
            file.write(f"P-value: {p_value}\n")
  
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_trained_models', type=str, default='results')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20)
    parser.add_argument('--output_path', type=str, default='outputs')

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args_dict = vars(args)
    device = args_dict['device']
    num_workers = args_dict['num_workers']
    output_path = args_dict['output_path']
    criterion = torch.nn.CrossEntropyLoss()
    if not os.path.isdir(output_path):
        os.makedirs(output_path, exist_ok = True)
    cifar_models = ['PreResNet20', 'PreResNet56', 'PreResNet110', 'PreResNet164', 'WideResNet28x10', 'VGG16BN']
    imagenet_models = ['vit_small_patch16_224','vit_large_patch16_224', 'swin_tiny_patch4_window7_224', 'swin_base_patch4_window7_224']
    amazon_models = ['bert','distill_bert', 'distill_roberta', 'roberta']
    data_names = ['cifar10', 'cifar100', 'amazon', 'imagenet'] 
    results = {}
    batch_size = 128
    score_func_name="MSP"
    for dataset in data_names:
        results[dataset] = {}
        if dataset in ['cifar10', 'cifar100']:
            model_names = cifar_models
        elif dataset == 'imagenet':
            model_names = imagenet_models
        elif dataset == 'amazon':
            model_names = amazon_models
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        if dataset in ['cifar10', 'cifar100']:
            for model_name in model_names:
                for seed in [5, 10, 21, 42, 84]:
                    results[dataset][f'{model_name}_{seed}'] = {}
                    folder_name = f'{dataset}_{model_name}_250_{seed}_output_{seed}'

                    path_to_root_folder = os.path.join(f'{output_path}/outputs', folder_name)
                    print("Root folder path: ", path_to_root_folder)
                    cache_path = os.path.join(path_to_root_folder, '*', '*', '*cache*')
                    matches = glob.glob(cache_path, recursive=True)
                    if len(matches) == 0:
                        print('Cache folder not found')
                        continue
                    path_to_cache_folder = matches[0]
                    print("Cache folder path:", path_to_cache_folder)
                    path_to_best_model = glob.glob(os.path.join(path_to_cache_folder, 'epoch_*'))[0]
                    best_model_filename = os.path.basename(path_to_best_model)
                    epoch = best_model_filename.split("_")[1]
                    print(f"Best model saved at epoch: {epoch}")
                    preds_dict = torch.load(os.path.join(path_to_cache_folder, f'preds_{epoch}.pth'))
                    logits_test, labels_test = get_logits_and_labels(preds_dict, 'preds_test', 'gt_test')
                    test_set = CustomTensorDataset(tensors=(logits_test, labels_test), transform=None)
                    loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
                    full_data_results = get_full_data_results_from_logits(loader, criterion, device, score_func_name)
                    results[dataset][f'{model_name}_{seed}'].update(full_data_results)
        elif dataset == "amazon":
            for model_name in model_names:
                results[dataset][model_name] = {}
                folder_name = f'amazon_{model_name}'
                path_to_root_folder = os.path.join(f'{output_path}/Amazon', folder_name)
                print("Root folder path: ", path_to_root_folder)
                with open(os.path.join(path_to_root_folder, f'target_agg.json'), 'r') as file:
                    preds_dict  = json.load(file)
                logits_test, labels_test = get_logits_and_labels(preds_dict, 'y_logits', 'y_true')
                test_set = CustomTensorDataset(tensors=(logits_test, labels_test), transform=None)
                loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
                full_data_results = get_full_data_results_from_logits(loader, criterion, device, score_func_name)
                results[dataset][model_name].update(full_data_results)

        elif dataset == "imagenet":
            loader = prepare_dataset('imagenet', batch_size=batch_size, load_train=False, num_workers=4) 
            for model_name in model_names:
                results[dataset][model_name] = {}
                model = timm.create_model(model_name, pretrained=True).to(device)
                model.eval()
                full_data_results = get_full_data_results(model, loader, device, score_func_name)
                results[dataset][model_name].update(full_data_results)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
    plot_population_aurc(results, data_names, figs_path=output_path)
        
