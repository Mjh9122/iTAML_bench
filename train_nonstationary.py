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
#import rmnist_dataloader as data
import non_stationary_datasets as data
from learner_rmnist import Learner

class args:
    # Place to put all logs and intermediate results
    checkpoint = "results/non_stat/"
    savepoint = "models/" + "/".join(checkpoint.split("/")[1:])
    data_path = "../Datasets/nonstationary_root/"
    
    # Self explanatory 
    n_datasets = 4
    n_way = 5
    k_shot = 5
    q_query = 15
    n_tasks = 32
    epochs = 50   
    iterations = 40
    memory = 12800 
    seed = 0
    
    test_samples_per_task = 10000
    optimizer = 'sgd'

    lr = 0.01
    train_batch = 256
    test_batch = 256
    workers = 16
    sess = 0
    gamma = 0.5
   
    mu = 1
    beta = 0.5
    
    
state = {key:value for key, value in args.__dict__.items() if not key.startswith('__') and not callable(key)}
print(state)

# Use CUDA
use_cuda = torch.cuda.is_available()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed_all(args.seed)

def main():
    # TODO
    # Change out model for correct model
    model = BasicNet1(args, 0).cuda() 
    print('  Total params: %.2fM ' % (sum(p.numel() for p in model.parameters())/1000000.0))

    # Create dirs to save logs
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)
    if not os.path.isdir(args.savepoint):
        mkdir_p(args.savepoint)
    np.save(args.checkpoint + "/seed.npy", seed)
    
    # Create dataset registry
    dataset_reg, names = get_dataset_registry()
    
    # Initialize memory 
    # data_memory, task_memory, age
    memory = np.array([], dtype = 'uint32'), np.array([], dtype = 'uint32'), 0
    
    for ses in range(0, n_datasets):
        args.sess=ses 
        current_dataset = dataset_reg[names[ses]]
        
        # If its the first time through, save teh model as base
        if(ses==0):
            torch.save(model.state_dict(), os.path.join(args.savepoint, 'base_model.pth.tar'))
        # Else save model under session 
        if ses>0: 
            path_model=os.path.join(args.savepoint, 'session_'+str(ses-1) + '_model_best.pth.tar')  
            prev_best=torch.load(path_model)
            model.load_state_dict(prev_best)

            with open(args.savepoint + "/memory_"+str(args.sess-1)+".pickle", 'rb') as handle:
                memory = pickle.load(handle)
        
        # TODO
        # Create train dataloader for this dataset -> #epochs {#n_tasks{(n_way, k_shot){# iterations}}} 
        for epoch in range(args.epochs):
            # Create dataloader of n_tasks tasks
            tasks = [current_dataset.sample_n_way_k_shot(args.n_way, args.k_shot, args.q_query) for _ in range(args.n_tasks)]
            
            
            main_learner(model = model, args = args, trainloader = dataloader, use_cuda = use_cuda, memory = memory)
            main_learner.learn()
            memory = main_learner.get_memory()
                    

        # Meta test after training on a dataset
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

