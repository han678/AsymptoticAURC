import matplotlib.pyplot as plt
import numpy as np
# pip install python-ternary
import matplotlib.patches as mpatches 
from matplotlib.lines import Line2D
from visualize.plot_statistic_metrics import get_label


def get_score_name(score_func_name):
    if score_func_name == "MSP":
        return "MSP"
    elif score_func_name == "NegEntropy":
        return "NegEntropy"
    elif score_func_name == "MaxLogit":
        return "MaxLogit"
    elif score_func_name == "l2_norm":
        return r"$\ell_2$ norm"
    elif score_func_name == "SoftmaxMargin":
        return "SoftmaxMargin"
    elif score_func_name == "NegGiniScore":
        return "NegGiniScore"
    else:
        raise ValueError(f"Unknown select function: {score_func_name}")


def plot_csf(data_dict, score_function_names, figs_path):
    metrics_name_1 = ['asy_aurc', 'sele', '2sele', 'true_aurc']
    metrics_name_2 = ['01_asy_aurc', '01_sele', '01_2sele', 'geifman_aurc', '01_true_aurc']
    metrics = [metrics_name_1, metrics_name_2]
    descrip = ["_ce", ""]
    colors = ['C0', 'C2', 'C1', 'C4', 'magenta', 'yellow', 'black']  # List of colors for the plots

    for i in range(len(metrics)):
        metrics_name = metrics[i]
        plt.figure(figsize=(10, 8))

        num_metrics = len(metrics_name)
        num_score_functions = len(score_function_names)
        group_spacing = 1.5  # Space between groups of boxplots
        metric_spacing = 0.2  # Space between individual metrics within the same score function
        positions = []
        for j in range(num_score_functions):
            base_position = j * group_spacing  # Base position for the score function
            for k in range(num_metrics - 1):  # Exclude 'true_aurc'
                positions.append(base_position + k * metric_spacing)
        group_midpoints = []
        for j, score_function in enumerate(score_function_names): 
            data_to_plot = [data_dict[score_function][metric] for metric in metrics_name if 'true_aurc' not in metric]
            pos_range = positions[j*(num_metrics - 1):(j+1)*(num_metrics - 1)]
            box = plt.boxplot(data_to_plot, patch_artist=True, positions=pos_range, widths=0.2)
            for patch, color in zip(box['boxes'], colors[:num_metrics - 1]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7) 
            true_aurc_value = np.mean(data_dict[score_function][metrics_name[-1]])  # Use the mean as the value for the horizontal line
            plt.hlines(y=true_aurc_value, xmin=pos_range[0] - 0.3, xmax=pos_range[-1] + 0.3, color=colors[-1], linestyle='--', label=f"{metrics_name[-1]} for {score_function}")
            group_midpoints.append(np.mean(pos_range))
        plt.xticks(group_midpoints, [get_score_name(score_func) for score_func in score_function_names], rotation=0, ha="center", fontsize="large")
        plt.ylabel('Metric Value', fontsize="large")
        legend_handles = []
        for k in range(num_metrics - 1):  # Legend for metrics except 'true_aurc'
            patch = mpatches.Patch(color=colors[k], label=get_label(metrics_name[k]))
            patch.set_alpha(0.7)  # Set legend color transparency
            legend_handles.append(patch)
        true_aurc_line = Line2D([0], [0], color=colors[-1], linestyle='--', label=get_label(metrics_name[-1]))
        legend_handles.append(true_aurc_line) 
        plt.legend(handles=legend_handles, loc='upper left', fontsize="large", ncol=len(legend_handles))
        plt.tight_layout()
        plt.savefig(f'{figs_path}{descrip[i]}.png')
        plt.show()
        plt.close()