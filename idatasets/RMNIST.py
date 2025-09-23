import torch
import os
import numpy as np

import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader


class RMNIST(Dataset):
    """ 
    Combine MNIST rotation dataset from LAMAML with the expected dataset in iTAML 
    """
    def __init__(self, root, train = True, transform = None, loader=default_loader, download = False, task_order = None, recurring = False):
        # iTAML init
        self.root = root
        self.transform = transform
        self.load = default_loader
        self.train = train 
        self.task_order = task_order
        self.recurring = recurring

        # LAMAML init
        train_dataset, test_dataset = torch.load(os.path.join(root, "mnist_rotations.pt"))
        self.n_tasks = len(train_dataset)

        if self.train:
            self.dataset = train_dataset
        else:
            self.dataset = test_dataset 

        image_stack, label_stack, tid_stack = [], [], []

        for tid in range(self.n_tasks):
            images = self.dataset[tid][1]
            labels = self.dataset[tid][2]
            
            if self.task_order is None or self.recurring:
                tids = torch.full((images.size(0),), tid, dtype=torch.long) 
            else:
                tids = torch.full((images.size(0),), self.task_order[tid], dtype=torch.long)
            
            image_stack.append(images)
            label_stack.append(labels)
            tid_stack.append(tids)

        self.images = torch.cat(image_stack)
        self.targets = torch.cat(label_stack)
        self.tids = torch.cat(tid_stack)  
        
        if self.recurring and self.train:
            self.split_permute()  
            self.relabel()
               

        self.print_dataset_stats()

    def __len__(self):
        return self.images.shape[0] 

    def __getitem__(self, idx):
        img = self.images[idx]
        target = self.targets[idx]  
        task = self.tids[idx]
        if isinstance(idx, int) or np.issubdtype(type(idx), np.integer):
            img = img.reshape(1, 28, 28)
        else:
            img = img.reshape(-1, 1, 28, 28)


        if self.transform is not None:
            img = self.transform(img)

        return img, target, task, idx
    

    def print_dataset_stats(self):
        print(f"Printing RMNIST {'Train' if self.train else 'Test'} Dataset Stats")
        print(f"Total images: {self.images.shape[0]}")
        print(f"Tasks present in dataset: {[int(i) for i in torch.unique(self.tids)]}")
        
        
    def split_permute(self):
        even_task_indices = torch.where(self.tids % 2 == 0)
        new_images = torch.empty_like(self.images[even_task_indices])
        new_targets = torch.empty_like(self.targets[even_task_indices])
        new_tids = torch.empty_like(self.tids[even_task_indices])

        write_idx = 0
        for task in torch.unique(self.tids[even_task_indices]):
            indices = torch.where(self.tids == task)[0]
            first_half = indices[:len(indices)//2]
            second_half = indices[len(indices)//2:] 
            
            first_half = first_half[torch.randperm(first_half.size(0))]
            second_half = second_half[torch.randperm(second_half.size(0))]
            
            first_size = len(first_half)
            new_images[write_idx:write_idx+first_size] = self.images[first_half]
            new_targets[write_idx:write_idx+first_size] = self.targets[first_half]
            new_tids[write_idx:write_idx+first_size] = task * 2
            write_idx += first_size
            
            second_size = len(second_half)
            new_images[write_idx:write_idx+second_size] = self.images[second_half]
            new_targets[write_idx:write_idx+second_size] = self.targets[second_half]
            new_tids[write_idx:write_idx+second_size] = task * 2 + 1
            write_idx += second_size
            
        self.images = new_images
        self.targets = new_targets
        self.tids = new_tids
    
    def relabel(self):
        new_tids = torch.empty_like(self.tids)
        
        for old_tid in torch.unique(self.tids):
            mask = self.tids == old_tid
            new_tids[mask] = self.task_order.index(int(old_tid))
        
        self.tids = new_tids