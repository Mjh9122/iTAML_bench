import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

results_path = "./results/rec_seed20_1/"

def plot_heat_map(df, title, out_path):
    
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
    
def get_stats(result):
    nt = result.size(0)
    baseline = result[0]
    
    # Standard metrics
    acc = result.diag()
    fin = result[nt - 1]
    bwt = result[nt - 1] - acc

    fwt = torch.zeros(nt)
    for t in range(1, nt):
        fwt[t] = result[t - 1, t] - baseline[t]

    # Additional metrics
    retained_acc = []
    learned_acc = []
    future_acc = []

    for t in range(nt):
        learned_acc.append(result[t, t].item())

        if t > 0:
            retained = result[t, :t].mean().item()
        else:
            retained = 0.0
        retained_acc.append(retained)

        if t < nt - 1:
            future = result[t, t + 1:].mean().item()
        else:
            future = 0.0
        future_acc.append(future)
        

    print(' '.join(['%.4f' % r for r in baseline]))
    print('|')
    for row in range(result.size(0)):
        print(' '.join(['%.4f' % r for r in result[row]]))
    print('')
    print('Diagonal Accuracy: %.4f' % (acc.mean()/100))
    print('Final Accuracy: %.4f' % (fin.mean()/100))
    print('Backward: %.4f' % (bwt.mean()/100))
    print('Forward:  %.4f' % (fwt.mean()/100))

    print('\nRetained Accuracy per Task:')
    print(' '.join(['%.4f' % r for r in retained_acc]))

    print('\nLearned Accuracy per Task:')
    print(' '.join(['%.4f' % r for r in learned_acc]))

    print('\nFuture Accuracy per Task:')
    print(' '.join(['%.4f' % r for r in future_acc]))
        
    # Return summary stats
    stats = [
        acc.mean(),
        fin.mean(),
        bwt.mean(),
        fwt.mean(),
        np.mean(retained_acc),
        np.mean(learned_acc),
        np.mean(future_acc)
    ]

    return stats
     

if __name__ == '__main__':
    post_in_path = results_path + 'post_meta_training_acc.csv' 
    post_out_path = results_path + 'Post_meta_training_task_accuracies.png'
    post_title = 'Heatmap of Post-meta-training Task Accuracies'
    post_df = pd.read_csv(post_in_path, index_col = 0)
    
    results = torch.from_numpy(post_df.values).float()
    
    get_stats(results)
    
    plot_heat_map(post_df, post_title, post_out_path)
   