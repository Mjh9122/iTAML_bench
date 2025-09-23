'''
TaICML incremental learning
Copyright (c) Jathushan Rajasegaran, 2019
'''
from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import pickle
import torch
import pdb
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig

import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.autograd import gradcheck
import sys
import random
import collections

from basic_net import *
#from learner_task_itaml import Learner
#import incremental_dataloader as data
import rmnist_dataloader as data
from learner_rmnist import Learner

class args:
    # Place to put all logs and intermediate results
    checkpoint = "results/seed20_run1/"
    savepoint = "models/" + "/".join(checkpoint.split("/")[1:])
    
    # Place where dataset is stores
    data_path = "../Datasets/RMNIST/"
    
    # Self explanatory 
    num_class = 10
    class_per_task = 10
    num_task = 20
    test_samples_per_task = 10000
    dataset = "rmnist"
    optimizer = 'sgd'
    epochs = 1
    lr = 0.01
    train_batch = 256
    test_batch = 256
    context_size = 90
    workers = 16
    sess = 0
    gamma = 0.5
    memory = 30000 
    mu = 1
    beta = 0.5
    
    # Shuffle classes?
    random_classes = False
    
    # Use validation split?
    validation = 0
    
    # Task Order
    #task_order = [18, 1, 19, 8, 10, 17, 6, 13, 4, 2, 5, 14, 9, 7, 16, 11, 3, 0, 15, 12]
    #task_order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    task_order = [8, 28, 9, 29, 12, 32, 13, 33, 16, 36, 17, 37, 0, 20, 1, 21, 4, 24, 5, 25]
    
    # Recurring?
    recurring = True
    
state = {key:value for key, value in args.__dict__.items() if not key.startswith('__') and not callable(key)}
print(state)

# Use CUDA
use_cuda = torch.cuda.is_available()
seed = 20 
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if use_cuda:
    torch.cuda.manual_seed_all(seed)

def main():
    model = BasicNet1(args, 0).cuda() 
    print('  Total params: %.2fM ' % (sum(p.numel() for p in model.parameters())/1000000.0))

    # Create dirs to save logs
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)
    if not os.path.isdir(args.savepoint):
        mkdir_p(args.savepoint)
    np.save(args.checkpoint + "/seed.npy", seed)
    
    # Create dataset
    inc_dataset = data.IncrementalDataset(
                        dataset_name=args.dataset,
                        args = args,
                        random_order=args.random_classes,
                        shuffle=True,
                        seed=seed,
                        batch_size=args.train_batch,
                        workers=args.workers,
                        validation_split=args.validation,
                        increment=args.class_per_task,
                    )
    
    # Should be 0 unless for some reason you want to skip in the task order
    start_sess = int(sys.argv[1])
    
    # Initialize memory 
    # data_memory, task_memory, age
    memory = np.array([], dtype = 'uint32'), np.array([], dtype = 'uint32'), 0
    
    for ses in range(start_sess, args.num_task):
        args.sess=ses 
        
        if(ses==0):
            torch.save(model.state_dict(), os.path.join(args.savepoint, 'base_model.pth.tar'))
        if(start_sess==ses and start_sess!=0): 
            inc_dataset._current_task = ses
            with open(args.savepoint + "/sample_per_task_testing_"+str(args.sess-1)+".pickle", 'rb') as handle:
                sample_per_task_testing = pickle.load(handle)
            inc_dataset.sample_per_task_testing = sample_per_task_testing
            args.sample_per_task_testing = sample_per_task_testing
        
        if ses>0: 
            path_model=os.path.join(args.savepoint, 'session_'+str(ses-1) + '_model_best.pth.tar')  
            prev_best=torch.load(path_model)
            model.load_state_dict(prev_best)

            with open(args.savepoint + "/memory_"+str(args.sess-1)+".pickle", 'rb') as handle:
                memory = pickle.load(handle)
            
            
        # Get dataloaders for the new tasks
        task_info, train_loader, val_loader, test_loader = inc_dataset.new_task()
        print(f'Task info {task_info}')
        print(f'Samples per task in testing set {", ".join([str(i) + ": " + str(j) for i, j in inc_dataset.sample_per_task_testing.items()])}')
        args.sample_per_task_testing = inc_dataset.sample_per_task_testing
        
        
        # Main training loop, train on new train loader, test on the test set with no task adaptation
        main_learner=Learner(model=model,args=args,trainloader=train_loader,
                             trainset = inc_dataset.train_dataset, 
                             testloader=test_loader, use_cuda=use_cuda, 
                             memory = memory)
        
        main_learner.learn()
        memory = main_learner.get_memory()
        
        # Test on all tasks after adapting to thier task. 
        acc_task = main_learner.meta_test(main_learner.best_model, memory, inc_dataset)
        
        # Count tasks in the memory buffer for sanity check
        _, tasks, _ = memory
        tsk, counts = np.unique(tasks, return_counts = True)
        print(f'Length of memory: {len(tasks)}')
        print(f'Samples per task stats: {list(zip([int(i) for i in tsk], [int(i) for i in counts]))}')
        
        
        # Save all
        with open(args.savepoint + "/memory_"+str(args.sess)+".pickle", 'wb') as handle:
            pickle.dump(memory, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(args.savepoint + "/acc_task_"+str(args.sess)+".pickle", 'wb') as handle:
            pickle.dump(acc_task, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        with open(args.savepoint + "/sample_per_task_testing_"+str(args.sess)+".pickle", 'wb') as handle:
            pickle.dump(inc_dataset.sample_per_task_testing, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        time.sleep(5)
if __name__ == '__main__':
    main()

