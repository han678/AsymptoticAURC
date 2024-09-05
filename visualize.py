import os
import pandas as pd
import matplotlib.pyplot as plt

def  plot_train():
    dataset = 'cifar10'
    archs = ["vgg13_bn", "vgg16_bn", "resnet18", "resnet34","wrn"]  #["vgg13","vgg16", "vgg19", "vgg13_bn", "vgg16_bn", "vgg19_bn", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "wrn"]
    add_names = ["Train Top1 Acc", "Train Top5 Acc", "Test Top1 Acc", "Test Top5 Acc"] if dataset in ['cifar100', 'imagenet'] else ['Train Acc', "Test Acc"]
    column_names = ['Epoch', 'Train Loss'] + add_names + ['AUC', 'EAURC', 'AURC(g)', "ECE", 'Brier Score', '0_1 AURC(e)', '0_1 AURC(a)', 'AURC(e)', 'AURC(a)']
    score_func = "l2_norm" # "neg_entropy" "softmax" "l2_norm"
    log_path = f"outputs/{dataset}/{score_func}/exact"
    benchmark_path = f"outputs/{dataset}/ce"
    lrs = [0.001, 0.01 ,0.05]
    plot_benchmark=True
    for i in range(1, len(column_names) - 1):
        y_name = column_names[i]
        fig_path = f"figs/{dataset}/{score_func}/exact/{y_name}"
        if not os.path.isdir(fig_path):
            os.makedirs(fig_path, exist_ok = True)
        for arch in archs:
            for lr in lrs:
                df_bs64, df_bs128, df_bs256 = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
                if plot_benchmark:
                    benchmark_bs64, benchmark_bs128, benchmark_bs256 = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
                batch_list = [64, 128, 256]
                seed_list = [25, 40, 41, 42, 51]
                for batch in batch_list:
                    temp_df = pd.DataFrame()
                    for seed in seed_list:
                        root_dir = os.path.join(log_path, f'seed{seed}lr{lr}')
                        logger_dir = os.path.join(root_dir, f'{arch}_bs{batch}_fintune.txt')
                        skiprows = 2 if y_name in ["Train Top1 Acc", "Train Top5 Acc", 'Train Loss'] else 1
                        data = pd.read_csv(logger_dir, sep='\t', skiprows=skiprows, names=column_names)  
                        data.replace('None', .0, inplace=True)
                        data = data.apply(pd.to_numeric, errors='coerce')
                        length = len(data[y_name])
                        if length <= 35:
                            temp_df[seed] = data[y_name]
                        else:
                            temp_df[seed] = data[y_name][0:length-10]
                    if batch == 64:
                        df_bs64 = pd.concat([df_bs64, temp_df], axis=1)
                    elif batch == 128:
                        df_bs128 = pd.concat([df_bs128, temp_df], axis=1)
                    elif batch == 256:
                        df_bs256 = pd.concat([df_bs256, temp_df], axis=1)

                    if plot_benchmark:
                        temp_df = pd.DataFrame()
                        for seed in seed_list:
                            root_dir = os.path.join(benchmark_path, f'seed{seed}lr{lr}')
                            logger_dir = os.path.join(root_dir, f'{arch}_bs{batch}_fintune.txt')
                            skiprows = 2 if y_name in ["Train Top1 Acc", "Train Top5 Acc", 'Train Loss'] else 1
                            data = pd.read_csv(logger_dir, sep='\t', skiprows=skiprows, names=column_names)  
                            data.replace('None', .0, inplace=True)
                            data = data.apply(pd.to_numeric, errors='coerce')
                            length = len(data[y_name])
                            temp_df[seed] = data[y_name]
                        if batch == 64:
                            benchmark_bs64 = pd.concat([benchmark_bs64, temp_df], axis=1)
                        elif batch == 128:
                            benchmark_bs128 = pd.concat([benchmark_bs128, temp_df], axis=1)
                        elif batch == 256:
                            benchmark_bs256 = pd.concat([benchmark_bs256, temp_df], axis=1)
                    
                if plot_benchmark:
                    benchmark_mean_bs64, benchmark_std_bs64 = benchmark_bs64.mean(axis=1), benchmark_bs64.std(axis=1)
                    benchmark_mean_bs128, benchmark_std_bs128 = benchmark_bs128.mean(axis=1), benchmark_bs128.std(axis=1)
                    benchmark_mean_bs256, benchmark_std_bs256 = benchmark_bs256.mean(axis=1), benchmark_bs256.std(axis=1)

                mean_bs64, std_bs64 = df_bs64.mean(axis=1), df_bs64.std(axis=1)
                mean_bs128, std_bs128 = df_bs128.mean(axis=1), df_bs128.std(axis=1)
                mean_bs256, std_bs256 = df_bs256.mean(axis=1), df_bs256.std(axis=1)
                epochs = range(0, len(mean_bs64))
                plt.figure(figsize=(10, 5)) 
                plt.plot(epochs, mean_bs64, label='BS 64', linewidth=1, c="forestgreen")
                plt.fill_between(epochs, mean_bs64 - std_bs64, mean_bs64 + std_bs64, alpha=0.2)
                plt.plot(epochs, mean_bs128, label='BS 128', linewidth=1, color='turquoise')
                plt.fill_between(epochs, mean_bs128 - std_bs128, mean_bs128 + std_bs128, alpha=0.2)
                plt.plot(epochs, mean_bs256, label='BS 256', linewidth=1, color='coral')
                plt.fill_between(epochs, mean_bs256 - std_bs256, mean_bs256 + std_bs256, alpha=0.2)
                if plot_benchmark:
                    plt.plot(epochs, benchmark_mean_bs64, label='Benchmark BS64', linewidth=1, c="darkviolet")
                    plt.fill_between(epochs, benchmark_mean_bs64 - benchmark_std_bs64, benchmark_mean_bs64 + benchmark_std_bs64, alpha=0.2)
                    plt.plot(epochs, benchmark_mean_bs128, label='Benchmark BS128', linewidth=1, color='cadetblue')
                    plt.fill_between(epochs, benchmark_mean_bs128 - benchmark_std_bs128, benchmark_mean_bs128 + benchmark_std_bs128, alpha=0.2)
                    plt.plot(epochs, benchmark_mean_bs256, label='Benchmark BS256', linewidth=1, color='pink')
                    plt.fill_between(epochs, benchmark_mean_bs256 - benchmark_std_bs256, benchmark_mean_bs256 + benchmark_std_bs256, alpha=0.2)

                plt.title(f'{y_name} over Epochs for Different Batch Sizes')
                plt.xlabel('Epoch')
                plt.ylabel(f'{y_name}')
                plt.legend()
                plt.grid(True)
                plt.savefig(fig_path + f'/{arch}_lr{lr}_{y_name}.png')
                plt.show()
                plt.close()

if __name__ == '__main__':
    plot_train()

                              


