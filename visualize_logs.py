import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    pre_meta_training_acc_path = './results/rmnist/pre_meta_training_acc.csv'
    pre_meta_acc = pd.read_csv(pre_meta_training_acc_path, index_col = 0)
    plt.figure(figsize=(6, 4))
    ax = sns.heatmap(pre_meta_acc, vmin = 0, vmax = 100, cmap='viridis', fmt=".4f")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha = 'right')
    
    plt.title('Heatmap of Pre-meta-training Task Accuracies')
    plt.tight_layout()
    plt.savefig('Pre_meta_training_task_accuracies.png')
    
    post_meta_training_acc_path = './results/rmnist/post_meta_training_acc.csv' 
    post_meta_acc = pd.read_csv(post_meta_training_acc_path, index_col = 0)
    plt.figure(figsize=(6, 4))
    ax = sns.heatmap(post_meta_acc, vmin = 0, vmax = 100, cmap='viridis', fmt=".4f")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha = 'right')
    
    plt.title('Heatmap of Post-meta-training Task Accuracies')
    plt.tight_layout()
    plt.savefig('Post_meta_training_task_accuracies.png')
   
    combined_task_acc = pd.DataFrame(np.maximum(pre_meta_acc.values, post_meta_acc.values), index = pre_meta_acc.index, columns = pre_meta_acc.columns) 
    plt.figure(figsize=(6, 4))
    ax = sns.heatmap(combined_task_acc, vmin = 0, vmax = 100, cmap='viridis', fmt=".4f")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha = 'right')
    
    plt.title('Task Accuracies')
    plt.tight_layout()
    plt.savefig('Combined_task_accuracies.png')
