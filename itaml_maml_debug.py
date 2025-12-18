'''
TaICML incremental learning
Copyright (c) Jathushan Rajasegaran, 2019
'''
from __future__ import print_function

import random
import torch
from nonstationary_feature_extractor import NonStationaryClassifier 
import matplotlib.pyplot as plt

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import random
from tqdm import tqdm

import mlflow
import mlflow.pytorch
from pathlib import Path


from non_stationary_datasets import get_dataset_registry
from torch.utils.data import IterableDataset


class EpisodicBatcher(IterableDataset):
    def __init__(self, dataset, n_way: int, k_shot: int, q_query: int, num_tasks: int):
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.num_tasks = num_tasks

    def __iter__(self):
        for _ in range(self.num_tasks):
            yield self.dataset.sample_n_way_k_shot(self.n_way, self.k_shot, self.q_query)

    def __len__(self):
        return self.num_tasks

def episodes_to_batch(episodes, args):
        """
        Stack multiple episodes into a single batch.

        episodes: list of tuples (Sx, Sy, Qx, Qy, Sg, Qg) length E=num_task_per_iter
        Returns:
          Qx_b:  [E, Bq, ...]
          Qy_b:  [E, Bq]
          dom_x: [E, Kc, ...]
          dom_y: [E, Kc]
          Sg_b:  [E, Kc] (global labels, optional downstream)
          Qg_b:  [E, Bq] (global labels, optional downstream)
        """
        device = args.device
        Kc_target = args.n_way * args.k_shot
        Kc_target = max(1, Kc_target)

        dom_x_list, dom_y_list = [], []
        Qx_list, Qy_list = [], []
        Sg_list, Qg_list = [], []

        for (Sx, Sy, Qx, Qy, Sg, Qg) in episodes:
            # to device
            Sx = Sx.to(device); Sy = Sy.to(device)
            Qx = Qx.to(device); Qy = Qy.to(device)
            Sg = Sg.to(device); Qg = Qg.to(device)

            Ns = Sx.size(0)
            if Ns == Kc_target:
                Sx_ctx = Sx
                Sy_ctx = Sy
                Sg_ctx = Sg
            elif Ns > Kc_target:
                idx = torch.randperm(Ns, device=device)[:Kc_target]
                Sx_ctx = Sx.index_select(0, idx)
                Sy_ctx = Sy.index_select(0, idx)
                Sg_ctx = Sg.index_select(0, idx)
            else:
                idx = torch.randint(low=0, high=Ns, size=(Kc_target,), device=device)
                Sx_ctx = Sx.index_select(0, idx)
                Sy_ctx = Sy.index_select(0, idx)
                Sg_ctx = Sg.index_select(0, idx)

            # Append per-episode tensors with episode dim
            dom_x_list.append(Sx_ctx.unsqueeze(0))   # [1, Kc, ...]
            dom_y_list.append(Sy_ctx.unsqueeze(0))   # [1, Kc]
            Sg_list.append(Sg_ctx.unsqueeze(0))      # [1, Kc]

            # Queries already per-episode
            Qx_list.append(Qx.unsqueeze(0))          # [1, Bq, ...]
            Qy_list.append(Qy.unsqueeze(0))          # [1, Bq]
            Qg_list.append(Qg.unsqueeze(0))          # [1, Bq]

        dom_x = torch.cat(dom_x_list, dim=0)  # [E, Kc, ...]
        dom_y = torch.cat(dom_y_list, dim=0)  # [E, Kc]
        Sg_b  = torch.cat(Sg_list,  dim=0)    # [E, Kc]

        Qx_b = torch.cat(Qx_list, dim=0)      # [E, Bq, ...]
        Qy_b = torch.cat(Qy_list, dim=0)      # [E, Bq]
        Qg_b = torch.cat(Qg_list, dim=0)      # [E, Bq]

        return Qx_b, Qy_b, dom_x, dom_y, Sg_b, Qg_b

def observe(model, inner_opt, outer_opt, Qx, Qy, dom_x, dom_y, args, update_model=True, verbose=True):
    model.train()
    
    num_tasks = dom_x.shape[0]
    samples_per_task = dom_x.shape[1]
    num_classes = args.n_way
    
    Sy = dom_y
    Sx = dom_x
    
    Sx = Sx.to(args.device)
    Sy = Sy.to(args.device)
    Qx = Qx.to(args.device)
    Qy = Qy.to(args.device)
    
    # Create one-hot targets
    Sy_one_hot = torch.FloatTensor(num_tasks, samples_per_task, num_classes).to(args.device)
    Sy_one_hot.zero_()
    Sy_one_hot.scatter_(2, Sy[:, :, None], 1)
    
    Qy_one_hot = torch.FloatTensor(num_tasks, Qy.shape[1], num_classes).to(args.device)
    Qy_one_hot.zero_()
    Qy_one_hot.scatter_(2, Qy[:, :, None], 1)
    
    task_metrics = []
    train_loss = 0.0
    
    # Save initial model state
    initial_state = {k: v.clone() for k, v in model.state_dict().items()}
    outer_opt.zero_grad()
    
    for task_idx in range(num_tasks):
        # Reset model to initial state for each task
        model.load_state_dict(initial_state)
        
        task_Sx = Sx[task_idx]
        task_Sy_one_hot = Sy_one_hot[task_idx]
        task_Sy = Sy[task_idx]
        
        task_Qx = Qx[task_idx]
        task_Qy_one_hot = Qy_one_hot[task_idx]
        task_Qy = Qy[task_idx]
        
        task_loss_history = []
        
        # Inner loop adaptation
        for _ in range(args.r):
            inner_opt.zero_grad()
            
            _, task_logits = model(task_Sx)
            inner_loss = F.binary_cross_entropy_with_logits(task_logits, task_Sy_one_hot)
            inner_loss.backward()
            inner_opt.step()
            
            task_loss_history.append(inner_loss.item())
            train_loss += inner_loss.item()
        
        # Evaluate on support set
        model.eval()
        with torch.no_grad(): 
            logits_S, _ = model(task_Sx)
            preds_train = logits_S.argmax(dim=-1)
            train_acc = (preds_train == task_Sy).float().mean().item()
        
        # Outer loop update
        model.train()
        outer_opt.zero_grad() 
        logits_Q, _ = model(task_Qx)
        
        if update_model:
            outer_loss = F.binary_cross_entropy_with_logits(logits_Q, task_Qy_one_hot) 
            outer_loss.backward()
        
        # Evaluate on query set
        model.eval()
        with torch.no_grad():
            logits_Q_eval, _ = model(task_Qx)
            preds_Q_eval = logits_Q_eval.argmax(dim=-1)
            test_acc = (preds_Q_eval == task_Qy).float().mean().item()
            
        
        task_metric = {
            'task_idx': task_idx,
            'initial_loss': task_loss_history[0],
            'final_loss': task_loss_history[-1],
            'loss_reduction': task_loss_history[0] - task_loss_history[-1],
            'train_acc': train_acc,
            'test_acc': test_acc,
        }
        task_metrics.append(task_metric)
        
        if verbose:
            print(f"\nTask {task_idx}:")
            print(f"  Loss: {task_loss_history[0]:.4f} -> {task_loss_history[-1]:.4f} "
                  f"({task_metric['loss_reduction']:.4f})")
            print(f"  Train Acc: {train_acc*100:.2f}%")
            print(f"  Test Acc: {test_acc*100:.2f}%")
    
    avg_loss = np.mean([m['final_loss'] for m in task_metrics])
    avg_task_train_acc = np.mean([m['train_acc'] for m in task_metrics])
    avg_task_test_acc = np.mean([m['test_acc'] for m in task_metrics])
    
    results = {
        'avg_loss': avg_loss,
        'avg_task_train_acc': avg_task_train_acc,
        'avg_task_test_acc': avg_task_test_acc,
        'task_metrics': task_metrics,
    }
    
    if verbose:
        print(f"\nAggregated Results:")
        print(f"  Avg Inner Loss: {avg_loss:.4f}")
        print(f"  Avg Task Train Acc (during adaptation): {avg_task_train_acc*100:.2f}%")
        print(f"  Avg Task Test Acc (during adaptation): {avg_task_test_acc*100:.2f}%")
    
    return avg_loss, avg_task_train_acc, avg_task_test_acc, results

class args:
    # Place to put all logs and intermediate results
    checkpoint = "results/non_stat/"
    savepoint = "models/" + "/".join(checkpoint.split("/")[1:])
    data_path = "Datasets"
    
    n_way = 5
    k_shot = 5
    q_query = 15
    n_tasks = 4
    seed = 0
    dataset = 'vggflowers'
    inner_lr = 0.01
    outer_lr = 0.001
    device = 'cuda'
    epochs = 1000
    dropout_p = .2
   
    r = 5
    beta = .1
    
    mlflow_uri = "mlruns"
    mlflow_experiment_name = "iTAML baseline"

    

if __name__ == "__main__":
    # CUDA + seeding
    use_cuda = torch.cuda.is_available()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)

    # Args
    state = {key:value for key, value in args.__dict__.items() if not key.startswith('__') and not callable(key)}
    print(state)
    
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.mlflow_experiment_name)
    
    with mlflow.start_run():
        mlflow.log_params({
           'n_way': args.n_way,
           'k_shot': args.k_shot,
           'q_query': args.q_query,
          'dataset': args.dataset,
           'inner_lr': args.inner_lr,
           'outer_lr': args.outer_lr,
           'device': args.device,
           'r': args.r,
           'beta': args.beta,
           'epochs':args.epochs,
        })
        # Dataset
        dataset_reg, names = get_dataset_registry(args.data_path)
        test_dataset_reg, _ = get_dataset_registry(args.data_path, split = 'test')

        ds = dataset_reg['vggflowers']
        test_ds = test_dataset_reg['vggflowers']

        # Model + optim
        model = NonStationaryClassifier(dropout_p = args.dropout_p, classes = args.n_way).to(args.device)
        inner_opt = torch.optim.SGD(model.parameters(), args.inner_lr)
        outer_opt = torch.optim.SGD(model.parameters(), args.outer_lr)

        best_test_acc = 0
        best_model_state = None
        patience = 100
        no_improve = 0
    
        for epoch in tqdm(range(args.epochs)):
            # Get episodes, combine into batch
            episodes = [ds.sample_n_way_k_shot(args.n_way, args.k_shot, args.q_query) for _ in range(args.n_tasks)]
            Qx_b, Qy_b, dom_x, dom_y, Sg_b, Qg_b = episodes_to_batch(episodes, args)
            
            avg_loss, train_acc, test_acc, results = observe(model, inner_opt, outer_opt, Qx_b, Qy_b, dom_x, dom_y, args, verbose=False)
            
            mlflow.log_metrics({
                'train_loss': avg_loss,
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'accuracy_gap': train_acc - test_acc
            }, step=epoch)

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                print(f"Early stop at epoch {epoch}, best was {best_test_acc:.4f}")
                model.load_state_dict(best_model_state)
                break
    
        train_ds_testing_episodes = [ds.sample_n_way_k_shot(args.n_way, args.k_shot, args.q_query) for _ in range(100)]
        Qx_b, Qy_b, dom_x, dom_y, Sg_b, Qg_b = episodes_to_batch(train_ds_testing_episodes, args) 
        _, _, final_train_acc, results = observe(model, inner_opt, outer_opt, Qx_b, Qy_b, dom_x, dom_y, args, update_model=False, verbose=False)
        mlflow.log_metrics({
            'final_train_accuracy': final_train_acc
        })
        
        test_ds_testing_episodes = [test_ds.sample_n_way_k_shot(args.n_way, args.k_shot, args.q_query) for _ in range(100)]
        Qx_b, Qy_b, dom_x, dom_y, Sg_b, Qg_b = episodes_to_batch(test_ds_testing_episodes, args) 
        _, _, final_test_acc, results = observe(model, inner_opt, outer_opt, Qx_b, Qy_b, dom_x, dom_y, args, update_model=False, verbose=False)
        mlflow.log_metrics({
            'final_test_accuracy': final_test_acc
        }) 
        
        