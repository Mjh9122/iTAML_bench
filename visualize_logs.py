import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os


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
    
def get_stats(result, file=None):
    import sys
     
    output_file = file if file is not None else sys.stdout

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
        
    print(' '.join(['%.4f' % r for r in baseline]), file=output_file)
    print('|', file=output_file)
    for row in range(result.size(0)):
        print(' '.join(['%.4f' % r for r in result[row]]), file=output_file)
    print('', file=output_file)
    print('Diagonal Accuracy: %.4f' % (acc.mean()/100), file=output_file)
    print('Final Accuracy: %.4f' % (fin.mean()/100), file=output_file)
    print('Backward: %.4f' % (bwt.mean()/100), file=output_file)
    print('Forward:  %.4f' % (fwt.mean()/100), file=output_file)
    print('\nRetained Accuracy per Task:', file=output_file)
    print(' '.join(['%.4f' % r for r in retained_acc]), file=output_file)
    print('\nLearned Accuracy per Task:', file=output_file)
    print(' '.join(['%.4f' % r for r in learned_acc]), file=output_file)
    print('\nFuture Accuracy per Task:', file=output_file)
    print(' '.join(['%.4f' % r for r in future_acc]), file=output_file)
        
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

def plot_context_loss(df, figsize=(12, 8), save_path=None):
    plt.figure(figsize=figsize)
    unique_sessions = sorted(df['session'].unique())
    colors = plt.cm.tab20(range(len(unique_sessions)))
    
    for i, session in enumerate(unique_sessions):
        session_data = df[df['session'] == session]
        plt.plot(session_data['global_batch'], 
                   session_data['context_loss'],
                   color = colors[i], 
                   label=f'Session {session}',
                   alpha=0.8,
                    linewidth=1.5,
                    marker='o',
                    markersize=3)
    
    plt.xlabel('Global Batch', fontsize=12)
    plt.ylabel('Context Loss', fontsize=12)
    plt.title('Context Loss vs Global Batch (by Session)', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
     
def combine_log_files(results_path, num_sessions=20, rows_per_session=233):
    all_data = []
    
    for session_i in range(num_sessions):
        filename = os.path.join(results_path, f"session_{session_i}_log.txt")
        
        df_session = pd.read_csv(filename, sep='\t')
        df_session.columns = df_session.columns.str.strip()
        df_session['session'] = session_i
        df_session['batch'] = range(len(df_session))
        df_session['global_batch'] = session_i * rows_per_session + df_session['batch']
        
        df_session = df_session.rename(columns={
            'Support Loss': 'support_loss',
            'Context Loss': 'context_loss',
            'Support Acc.': 'support_acc',
            'Context Acc.': 'context_acc'
        })
        
        df_session = df_session[['global_batch', 'session', 'batch', 
                                'support_loss', 'context_loss', 'support_acc', 'context_acc']]
        
        all_data.append(df_session)

    return pd.concat(all_data, ignore_index=True)     

def combine_test_files(results_path, num_sessions=20):
    all_data = []
    
    for session_i in range(num_sessions):
        filename = os.path.join(results_path, f"session_{session_i}_meta_log.txt")
        
        df_session = pd.read_csv(filename, sep='\t')
        df_session.columns = df_session.columns.str.strip()
        df_session['session'] = session_i
        df_session['task'] = range(num_sessions)
        
        df_session = df_session.rename(columns={
            'Context acc. pre': 'context_acc_pre',
            'Context acc. post': 'context_acc_post',
            'Test acc. pre': 'test_acc_pre',
            'Test acc. post': 'test_acc_post'
        })
        
        df_session = df_session[['session', 'task', 'context_acc_pre', 'context_acc_post', 'test_acc_pre', 'test_acc_post']]
        
        all_data.append(df_session)

    return pd.concat(all_data, ignore_index=True)

def plot_improvement(df, figsize=(12, 8), save_path=None):
    df['context_improvement'] = df['context_acc_post'] - df['context_acc_pre']
    df['test_improvement'] = df['test_acc_post'] - df['test_acc_pre']

    session_improvements = df.groupby('session')[['context_improvement', 'test_improvement']].mean()
    
    plt.figure(figsize=figsize)
    plt.plot(session_improvements.index, session_improvements['context_improvement'], 
             marker='o', label='Context Set Improvement', linewidth=2)
    plt.plot(session_improvements.index, session_improvements['test_improvement'], 
             marker='s', label='Test Set Improvement', linewidth=2)
    
    plt.xlabel('Session Number')
    plt.ylabel('Accuracy Improvement (Post - Pre)')
    plt.title('Meta-Training Effectiveness Across Tasks')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    plt.show()


def process_results(results_path):
    sessions = 20
    post_in_path = results_path + 'post_meta_training_acc.csv' 
    post_out_path = results_path + 'Post_meta_training_task_accuracies.png'
    post_title = 'Heatmap of Post-meta-training Task Accuracies'
    post_df = pd.read_csv(post_in_path, index_col=0)
    
    log_df = combine_log_files(results_path) 
    plot_context_loss(log_df, save_path=results_path + "context_loss.jpg")

    meta_df = combine_test_files(results_path)
    plot_improvement(meta_df)
    results = torch.from_numpy(post_df.values).float()
    
    metrics_path = results_path + 'metrics.txt'
    with open(metrics_path, 'w') as f:
        stats = get_stats(results, file=f)
        print(stats[:3])
        
    plot_heat_map(post_df, post_title, post_out_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('results_path', help='Path to results directory')
    args = parser.parse_args()
    process_results(args.results_path)