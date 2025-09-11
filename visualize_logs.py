import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from idatasets.RMNIST import TASK_ORDER

results_path = "./results/rmnist/"
column_map = {f'task_{index}_acc':f'task_{order}_acc' for index, order in enumerate(TASK_ORDER)}

def plot_heat_map(df, title, out_path):
    df = df.rename(columns = column_map)
    
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(df, vmin = 0, vmax = 100, cmap='viridis', fmt=".4f", cbar_kws={'shrink': 0.8, 'pad': 0.15})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha = 'right')
    
    task_averages = df.mean(axis=1)
    for i, avg in enumerate(task_averages):
        ax.text(len(df.columns) + 0.5, i + 0.5, f'Avg: {avg:.2f}', 
                va='center', ha='left', fontsize=9, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
     

if __name__ == '__main__':
    pre_in_path = results_path + 'pre_meta_training_acc.csv'
    pre_out_path = results_path + 'Pre_meta_training_task_accuracies.png'
    pre_title = 'Heatmap of Pre-meta-training Task Accuracies'
    pre_df = pd.read_csv(pre_in_path, index_col = 0)
    plot_heat_map(pre_df, pre_title, pre_out_path)

    
    post_in_path = results_path + 'post_meta_training_acc.csv' 
    post_out_path = results_path + 'Post_meta_training_task_accuracies.png'
    post_title = 'Heatmap of Post-meta-training Task Accuracies'
    post_df = pd.read_csv(post_in_path, index_col = 0)
    plot_heat_map(post_df, post_title, post_out_path)
   
    print(f'Pre-meta-training acc > Post-meta-training acc count', np.sum((pre_df.values > post_df.values) & (post_df.values != 0))) 
    post_zeros = post_df.values == 0
    combined_np = post_df.values.copy()
    combined_np[post_zeros] = pre_df.values[post_zeros]
    combined_df = pd.DataFrame(combined_np, index = pre_df.index, columns = pre_df.columns) 
    combined_out_path = results_path + 'Combined_task_accuracies.png'
    combined_title = 'Task Accuracies'
    plot_heat_map(combined_df, combined_title, combined_out_path)
