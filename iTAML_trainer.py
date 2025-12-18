import pytorch_lightning as pl

class iTAML_Classifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.mlp1 = nn.Linear(784, 100)
        self.mlp2 = nn.Linear(100, 100)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(100, 10, bias=False)

    def forward(self, x):
        x = x.view(-1, 784)
        
        x = self.mlp1(x)
        x = F.relu(x)
        
        x = self.mlp2(x)
        x = F.relu(x)

        x1 = self.fc(x)
        x2 = self.fc(x)
        
        return x2, x1
    
    def training_step(self, batch, batch_index):
        inputs, target, task, idx = batch
    
        # Create one-hot targets
        targets_one_hot = torch.FloatTensor(len(inputs), num_classes)
        targets_one_hot.zero_()
        targets_one_hot.scatter_(2, targets[:, None], 1)
        targets_one_hot = targets_one_hot.to(args.device)
    
        # Save base model parameters
        base_params = [p.clone().detach() for p in model.parameters()]
    
        # task-specific adapted parameters
        task_params = []
    
        # Per-task metrics
        task_metrics = []
        train_loss = 0.0
 
        for task_idx in range(num_tasks):
            # Reset params to base for this task
            for p, base_p in zip(model.parameters(), base_params):
                p.data.copy_(base_p)
            
            # Task specific inputs/targets
            task_inputs = inputs[task_idx]
            task_targets_one_hot = targets_one_hot[task_idx]
            task_targets = targets[task_idx]
            
            # Task test data
            task_test_x = Qx[task_idx]
            task_test_y = Qy[task_idx]
            
            task_loss_history = []
            
            for _ in range(args.r):
                optimizer.zero_grad()
                
                _, task_outputs = model(task_inputs)
                loss = F.binary_cross_entropy_with_logits(task_outputs, task_targets_one_hot)
                loss.backward()
                optimizer.step()
                
                task_loss_history.append(loss.item())
                train_loss += loss.item()
            
            task_params.append([p.data.clone() for p in model.parameters()])
            
            with torch.no_grad():
                model.eval()
                
                logits_train, _ = model(task_inputs)
                preds_train = logits_train.argmax(dim=-1)
                train_acc = (preds_train == task_targets).float().mean().item()
                
                logits_test, _ = model(task_test_x)
                preds_test = logits_test.argmax(dim=-1)
                test_acc = (preds_test == task_test_y).float().mean().item()
                
                model.train()
            
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
    
    alpha = np.exp(-args.beta / num_tasks)
    if update_model:
        if verbose:
            print(f"Alpha = exp(-{args.beta}/{num_tasks}) = {alpha:.4f}")
    
        for i, (p, base_p) in enumerate(zip(model.parameters(), base_params)):
            avg_task_param = torch.mean(
                torch.stack([task_p[i] for task_p in task_params]), 
                dim=0
            )
            p.data.copy_(alpha * avg_task_param + (1 - alpha) * base_p)
            
    else:
        for i, (p, base_p) in enumerate(zip(model.parameters(), base_params)):
            p.data.copy_(base_p) 

    avg_loss = np.mean([m['final_loss'] for m in task_metrics])
    avg_task_train_acc = np.mean([m['train_acc'] for m in task_metrics])
    avg_task_test_acc = np.mean([m['test_acc'] for m in task_metrics])
    
    results = {
        'avg_loss': avg_loss,
        'avg_task_train_acc': avg_task_train_acc,
        'avg_task_test_acc': avg_task_test_acc,
        'task_metrics': task_metrics,
        'alpha': alpha,
    }
    
    if verbose:
        print(f"\nAggregated Results:")
        print(f"  Avg Inner Loss: {avg_loss:.4f}")
        print(f"  Avg Task Train Acc (during adaptation): {avg_task_train_acc*100:.2f}%")
        print(f"  Avg Task Test Acc (during adaptation): {avg_task_test_acc*100:.2f}%")
    
    return avg_loss, avg_task_train_acc, avg_task_test_acc, results