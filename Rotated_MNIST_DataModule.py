from typing import Optional, Tuple, Dict, Any, List
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import numpy as np
from torchvision import transforms
import os

class RotatedMNISTDataset(Dataset):
    def __init__(self, images, targets, tids, transform=None):
        self.images = images
        self.targets = targets
        self.tids = tids
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx]
        target = self.targets[idx]
        task = self.tids[idx]
        
        if img.dim() == 2:
            img = img.unsqueeze(0)
        
        if self.transform is not None:
           img = self.transform(img)
        
        return img, target, task, idx

class RotatedMNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        task_order: List[int], 
        recurring: bool = False,
        batch_size: int = 256,
        test_batch_size: int = 256,
        memory: int = 30000,
        num_workers: int = 1
    ):
        self.data_dir = data_dir
        self.task_order = task_order
        self.recurring = recurring
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.memory = memory
        self.num_workers = num_workers
        
        self.current_task_index = 0
        self.replay_buffer = {}
        self.n_tasks = None
        
    
    def prepare_data(self):
        pass
    
    def setup(self, stage: Optional[str] = None):
        rotated_path = os.path.join(self.data_dir, "mnist_rotations.pt")
        train_dataset, test_dataset = torch.load(rotated_path)
        
        self.n_tasks_base = len(train_dataset)
        
        self._stack_tasks(train_dataset, test_dataset)
        
        if self.recurring:
            self._apply_recurring_split()
            self.n_tasks = len(torch.unique(self.train_tids))
        else:
            self.n_tasks = self.n_tasks_base
            
        self._create_torch_datasets()
            
        self.replay_buffer = {i: [] for i in range(self.n_tasks)}
        
    def _stack_tasks(self, train_dataset, test_dataset):
        train_images, train_labels, train_tids = [], [], []
        test_images, test_labels, test_tids = [], [], []
    
        for tid in range(len(train_dataset)):
            train_images.append(train_dataset[tid][1]) 
            train_labels.append(train_dataset[tid][2])
            train_tids.append(torch.full((train_dataset[tid][1].size(0),), tid, dtype=torch.long))
            
            test_images.append(test_dataset[tid][1])
            test_labels.append(test_dataset[tid][2])
            test_tids.append(torch.full((test_dataset[tid][1].size(0),), tid, dtype=torch.long))
            
        self.train_images = torch.cat(train_images)
        self.train_targets = torch.cat(train_labels)
        self.train_tids = torch.cat(train_tids)
        
        self.test_images = torch.cat(test_images)
        self.test_targets = torch.cat(test_labels)
        self.test_tids = torch.cat(test_tids)
        
    def _apply_recurring_split(self):
        even_task_mask = self.train_tids % 2 == 0
        even_task_indices = torch.where(even_task_mask)[0]
        
        new_images = torch.empty_like(self.train_images[even_task_indices])
        new_targets = torch.empty_like(self.train_targets[even_task_indices])
        new_tids = torch.empty_like(self.train_tids[even_task_indices])
        
        write_idx = 0
        for task in torch.unique(self.train_tids[even_task_indices]):
            task = int(task) 
            task_mask = self.train_tids == task
            indices = torch.where(task_mask)[0]
            
            mid = len(indices) // 2
            first_half = indices[:mid]
            second_half = indices[mid:]
            
            first_half = first_half[torch.randperm(len(first_half))]
            second_half = second_half[torch.randperm(len(second_half))]
            
            first_size = len(first_half)
            new_images[write_idx:write_idx+first_size] = self.train_images[first_half]
            new_targets[write_idx:write_idx+first_size] = self.train_targets[first_half]
            new_tids[write_idx:write_idx+first_size] = task * 2
            write_idx += first_size
            
            second_size = len(second_half)
            new_images[write_idx:write_idx+second_size] = self.train_images[second_half]
            new_targets[write_idx:write_idx+second_size] = self.train_targets[second_half]
            new_tids[write_idx:write_idx+second_size] = task * 2 + 1
            write_idx += second_size
        
        self.train_images = new_images
        self.train_targets = new_targets
        self.train_tids = new_tids
    
    def _create_torch_datasets(self):
        transform = lambda x: (x - 0.1307)/0.3081
        
        self.train_dataset = RotatedMNISTDataset(
            self.train_images, self.train_targets, self.train_tids, transform
        )
        self.test_dataset = RotatedMNISTDataset(
            self.test_images, self.test_targets, self.test_tids, transform
        )
        
    def _get_train_indices(self, task_index_list: List[int]) -> List[int]:
        task_ids_list = [self.task_order[i] for i in task_index_list]
        task_ids_np = np.array(self.train_dataset.tids, dtype=int)
        sample_indices = np.where(np.isin(task_ids_np, task_ids_list))[0]
        return [int(i) for i in sample_indices]
    
    def _get_test_indices(self, task_index_list: List[int]) -> List[int]:
        task_ids_list = [self.task_order[i] for i in task_index_list]
        task_ids_np = np.array(self.test_dataset.tids, dtype="uint32")
        sample_indices = np.where(np.isin(task_ids_np, task_ids_list))[0]
        return list(sample_indices)
    
    def train_dataloader(self) -> DataLoader:
        current_task_indices = self._get_train_indices([self.current_task_index])
        
        sampler = SubsetRandomSampler(current_task_indices)
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        all_tasks = list(range(self.n_tasks))
        test_indices = self._get_test_indices(all_tasks)
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def val_dataloader(self) -> DataLoader:
        return self.test_dataloader()
    
    # def update_replay_buffer(self, task_id: int, samples: torch.Tensor, labels: torch.Tensor):

    # def sample_replay(self, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    def get_current_task_index(self) -> int:
        return self.current_task_index
  
    def set_current_task_index(self, task_index: int):
        self.current_task_index = task_index
        
    def next_task(self):
        if self.current_task_index < self.n_tasks - 1:
            self.current_task_index += 1
        else:
            raise ValueError(f"Already at final task {self.n_tasks - 1}")

