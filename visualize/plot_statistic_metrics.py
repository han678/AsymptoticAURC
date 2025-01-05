from matplotlib import pyplot as plt
import numpy as np


def get_label(metric_name):
    if "mc" in metric_name:
        return "Ours"
    elif "sele" in metric_name and "2sele" not in metric_name:
        return "SELE"
    elif "true" in metric_name:
        return r"$AURC_p$"
    elif "aurc_a" in metric_name:
        return r"$AURC_a$"
    elif "2sele" in metric_name:
        return r"$2\times$SELE"
    else:
        return metric_name

def plot_aurc_metrics(data_dict, batch_size_list, figs_path):
    metrics_name_1 = ['mc_aurc', 'sele', '2sele', 'true_aurc']
    metrics_name_2 = ['01_mc_aurc', '01_sele', '01_2sele', '01_true_aurc']
    metrics = [metrics_name_1, metrics_name_2]
    descrip = ["_ce", ""]
    colors = ['C0', 'C2', 'C1',  'C4', 'magenta', 'yellow', 'black']  # List of colors for the plots

    for i in range(len(metrics)):
        metrics_name = metrics[i]
        plt.figure(figsize=(10, 8))
        for metric, color in zip(metrics_name[0:-1], colors):  # Use zip to iterate over metrics and colors
            y, yerr = [], []
            for batch_size in batch_size_list:
                y.append(data_dict[str(batch_size)][metric]['mean'])
                yerr.append(data_dict[str(batch_size)][metric]['std'])
            plt.errorbar(np.log2(batch_size_list), y, yerr=yerr, fmt='o-', color=color, label=get_label(metric), capsize=3)
            lower_bound = [mean - err for mean, err in zip(y, yerr)]
            upper_bound = [mean + err for mean, err in zip(y, yerr)]
            plt.fill_between(np.log2(batch_size_list), lower_bound, upper_bound, color=color, alpha=0.15)
        # Ensure to plot the last metric with a specific style if it's constant across batch sizes
        if metrics_name[-1] in data_dict:
            plt.plot(np.log2(batch_size_list), [data_dict[metrics_name[-1]]] * len(batch_size_list), 
                     label=get_label(metrics_name[-1]), color='black', linestyle=':')
        
        plt.xlabel(r'$\log_2(n)$', fontsize=24)
        plt.ylabel('Finite sample estimator', fontsize=24)
        plt.legend(fontsize=23)
        #plt.grid(True)
        plt.savefig(f'{figs_path}{descrip[i]}.png')
        plt.show()
        plt.close()

def plot_bias(data_dict, batch_size_list, figs_path):
    metrics_name_1 = ['mc_aurc', 'sele', '2sele', 'true_aurc']
    metrics_name_2 = ['01_mc_aurc', '01_sele', '01_2sele', '01_true_aurc']
    metrics = [metrics_name_1, metrics_name_2]
    descrip = ["_ce", ""]
    colors = ['C0', 'C2', 'C1',  'C4', 'magenta', 'yellow', 'black']  # List of colors for the plots

    for i in range(len(metrics)):
        metrics_name = metrics[i]
        plt.figure(figsize=(10, 8))
        for metric, color in zip(metrics_name[0:-1], colors):  # Use zip to iterate over metrics and colors
            y, yerr = [], []
            for batch_size in batch_size_list:
                y.append(data_dict[str(batch_size)][metric]['mean']- data_dict[metrics_name[-1]])
                yerr.append(data_dict[str(batch_size)][metric]['std'])
            plt.errorbar(np.log2(batch_size_list), y, yerr=yerr, fmt='o-', color=color, label=get_label(metric), capsize=3)
            lower_bound = [mean - err for mean, err in zip(y, yerr)]
            upper_bound = [mean + err for mean, err in zip(y, yerr)]
            plt.fill_between(np.log2(batch_size_list), lower_bound, upper_bound, color=color, alpha=0.15)
        
        plt.xlabel(r'$\log_2(n)$', fontsize=24)
        plt.ylabel('Bias', fontsize=24)
        plt.legend(fontsize=24)
        #plt.grid(True)
        plt.savefig(f'{figs_path}{descrip[i]}.png')
        plt.show()
        plt.close()

def plot_mse(data_dict, batch_size_list, figs_path):
    metrics_name_1 = ['mc_aurc', 'sele', '2sele', 'true_aurc']
    metrics_name_2 = ['01_mc_aurc', '01_sele', '01_2sele', '01_true_aurc']
    metrics = [metrics_name_1, metrics_name_2]
    descrip = ["_ce", ""]
    colors = ['C0', 'C2', 'C1',  'C4', 'magenta', 'yellow', 'black']  # List of colors for the plots

    for i in range(len(metrics)):
        metrics_name = metrics[i]
        plt.figure(figsize=(10, 8))
        for metric, color in zip(metrics_name[0:-1], colors):  # Use zip to iterate over metrics and colors
            y = []
            for batch_size in batch_size_list:
                y.append((data_dict[str(batch_size)][metric]['mean']-data_dict[metrics_name[-1]])**2)
            plt.plot(np.log2(batch_size_list), y, 'o-', color=color, label=get_label(metric))
        plt.xlabel(r'$\log_2(n)$', fontsize=24)
        plt.ylabel('MSE', fontsize=24)
        plt.legend(fontsize=24)
        plt.savefig(f'{figs_path}{descrip[i]}.png')
        plt.show()
        plt.close()

def plot_var(data_dict, batch_size_list, figs_path):
    metrics_name_1 = ['mc_aurc', 'sele', '2sele', 'true_aurc']
    metrics_name_2 = ['01_mc_aurc', '01_sele', '01_2sele', '01_true_aurc']
    metrics = [metrics_name_1, metrics_name_2]
    descrip = ["_ce", ""]
    colors = ['C0', 'C2', 'C1',  'C4', 'magenta', 'yellow', 'black']  # List of colors for the plots

    for i in range(len(metrics)):
        metrics_name = metrics[i]
        plt.figure(figsize=(10, 8))
        for metric, color in zip(metrics_name[0:-1], colors):  # Use zip to iterate over metrics and colors
            y = []
            for batch_size in batch_size_list:
                y.append((data_dict[str(batch_size)][metric]['mean']-data_dict[metrics_name[-1]])**2)
            plt.plot(np.log2(batch_size_list), y, 'o-', color=color, label=get_label(metric))
        plt.xlabel(r'$\log_2(n)$', fontsize=20)
        plt.ylabel('Variance', fontsize=20)
        plt.legend(fontsize=19)
        plt.savefig(f'{figs_path}{descrip[i]}.png')
        plt.show()
        plt.close()

def plot_mae(all_seed_results, batch_size_list, figs_path):
    metrics_name_1 = ['mc_aurc', 'sele'] 
    metrics_name_2 = ['01_mc_aurc', '01_sele'] 
    metrics = [metrics_name_1, metrics_name_2]
    descrip = ["_ce", ""]
    colors = ['C0', 'C2',  'C4', 'C1', 'magenta', 'yellow', 'black'] 

    for i in range(len(metrics)):
        metrics_name = metrics[i]
        plt.figure(figsize=(10, 8))
        for metric, color in zip(metrics_name, colors):
            y, yerr = all_seed_results[metric]['mean'], all_seed_results[metric]['std']
            plt.errorbar(np.log2(batch_size_list), y, yerr, fmt='o', label=get_label(metric), capsize=4, color=color)
            lower_bound = [mean - err for mean, err in zip(y, yerr)]
            upper_bound = [mean + err for mean, err in zip(y, yerr)]
            plt.fill_between(np.log2(batch_size_list), lower_bound, upper_bound, color=color, alpha=0.15)
        
        plt.xlabel(r'$\log_2(n)$', fontsize=24)
        plt.ylabel('MAE', fontsize=24)
        plt.legend(fontsize=20, ncol=len(metrics_name), loc='upper left')
        #plt.grid(True)
        plt.savefig(f'{figs_path}{descrip[i]}.png')
        plt.show()
        plt.close() 
