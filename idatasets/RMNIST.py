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
    def __init__(self, root, train = True, transform = None, loader=default_loader, download = False):
        # iTAML init
        self.root = root
        self.transform = transform
        self.load = default_loader
        self.train = train 

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

            # Sample subset if needed
            # rand_idx = torch.randperm(images.size(0))[:int(images.size(0) * self.args.dataset_percent)]
            # images = images[rand_idx]
            # labels = labels[rand_idx]
            tids = torch.full((images.size(0),), tid, dtype=torch.long)

            image_stack.append(images)
            label_stack.append(labels)
            tid_stack.append(tids)

        self.images = torch.cat(image_stack)
        self.targets = torch.cat(label_stack)
        self.tids = torch.cat(tid_stack)        

        print(f"Total images: {self.images.shape[0]}, Total labels: {self.targets.shape[0]}, Total tids: {self.tids.shape[0]}")
        print(np.unique(self.targets.numpy(), return_counts = True))

    def __len__(self):
        return self.images.shape[0] 

    def __getitem__(self, idx):
        img = self.images[idx]
        target = self.targets[idx]  
        img = img.reshape(1, 28, 28)


        if self.transform is not None:
            img = self.transform(img)

        return img, target