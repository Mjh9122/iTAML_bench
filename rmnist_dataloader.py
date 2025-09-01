'''
TaICML incremental learning
Copyright (c) Jathushan Rajasegaran, 2019
'''
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Sampler
from torchvision import datasets, transforms
# from imagenet import ImageNet
from idatasets.CUB200 import Cub2011
from idatasets.omniglot import Omniglot
from idatasets.celeb_1m import MS1M
from idatasets.RMNIST import RMNIST
import collections
from utils.cutout import Cutout

class SubsetRandomSampler(Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices, shuffle):
        self.indices = indices
        self.shuffle = shuffle

    def __iter__(self):
        if(self.shuffle):
            return (self.indices[i] for i in torch.randperm(len(self.indices)))
        else:
            return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)
    
    

class IncrementalDataset:

    def __init__(
        self,
        dataset_name,
        args,
        random_order=False,
        shuffle=True,
        workers=10,
        batch_size=128,
        seed=1,
        increment=10,
        validation_split=0.
    ):
        self.dataset_name = dataset_name.lower().strip()
        datasets = _get_datasets(dataset_name)
        self.train_transforms = datasets[0].train_transforms 
        self.common_transforms = datasets[0].common_transforms
        try:
            self.meta_transforms = datasets[0].meta_transforms
        except:
            self.meta_transforms = datasets[0].train_transforms
        self.args = args
        
        self._setup_data(
            datasets,
            args.data_path,
            random_order=random_order,
            seed=seed,
            increment=increment,
            validation_split=validation_split
        )
        

        self._current_task = 0

        self._batch_size = batch_size
        self._workers = workers
        self._shuffle = shuffle
        self.sample_per_task_testing = {}
    @property
    def n_tasks(self):
        return len(self.increments)
    
    def get_train_indices(self, task_ids, task_list, memory=None):
        task_ids_np = np.array(task_ids, dtype = int)
        sample_indices = np.where(np.isin(task_ids_np, task_list))[0]
        sample_tasks = task_ids_np[sample_indices]
        
        sample_indices = [int(i) for i in sample_indices] 
        sample_tasks = [torch.tensor(i) for i in sample_tasks]

        for_memory = (sample_indices, sample_tasks)
        
        # Add indices to memory buffer (they are selected later)
        if memory is not None:
            memory_indices, memory_targets = memory
            memory_indices2 = np.tile(memory_indices, (self.args.mu,))
            all_indices = np.concatenate([memory_indices2, sample_indices])
        else:
            all_indices = sample_indices
            
        return all_indices, for_memory
    
    def get_test_indices(self, task_ids, task_list, memory=None):
        task_ids_np = np.array(task_ids, dtype = "uint32")
        sample_indices = np.where(np.isin(task_ids_np, task_list))[0]
        sample_tasks = list(task_ids_np[sample_indices])
        tasks, counts = np.unique(sample_tasks, return_counts = True)
        self.sample_per_task_testing = dict(zip(tasks, counts))

        return list(sample_indices)
    

    def new_task(self, memory=None):
        """ Creates a new task

        Args:
            memory (_type_, optional): indices of samples in memory. Defaults to None.
        """
        print(f'Current task {self._current_task}')
        
        train_indices, for_memory = self.get_train_indices(self.train_dataset.tids, [self._current_task], memory=memory)
        test_indices = self.get_test_indices(self.test_dataset.tids, list(range(self._current_task + 1)), memory=memory)

        self.train_data_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self._batch_size,shuffle=False,num_workers=16, sampler=SubsetRandomSampler(train_indices, True))
        self.test_data_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.args.test_batch,shuffle=False,num_workers=16, sampler=SubsetRandomSampler(test_indices, False))

        
        task_info = {
            "task": self._current_task,
            "max_task": len(self.increments),
            "n_train_data": len(train_indices),
            "n_test_data": len(test_indices)
        }

        self._current_task += 1

        return task_info, self.train_data_loader, self.test_data_loader, self.test_data_loader, for_memory
    
     
        
    # for verification   
    def get_galary(self, task, batch_size=10):
        indexes = []
        dict_ind = {}
        seen_classes = []
        for i, t in enumerate(self.train_dataset.targets):
            if not(t in seen_classes) and (t< (task+1)*self.args.class_per_task and (t>= (task)*self.args.class_per_task)):
                seen_classes.append(t)
                dict_ind[t] = i
                
        od = collections.OrderedDict(sorted(dict_ind.items()))
        for k, v in od.items(): 
            indexes.append(v)
            
        data_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, sampler=SubsetRandomSampler(indexes, False))
    
        return data_loader
    
    
    def get_custom_loader_idx(self, indexes, mode="train", batch_size=10, shuffle=True):
     
        if(mode=="train"):
            data_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, sampler=SubsetRandomSampler(indexes, True))
        else: 
            data_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, sampler=SubsetRandomSampler(indexes, False))
    
        return data_loader
    
    
    def get_custom_loader_class(self, task_id, mode="train", batch_size=10, shuffle=False):
        
        if(mode=="train"):
            train_indices, for_memory = self.get_test_indices(self.train_dataset.task_id, task_id, mode="train", memory=None)
            data_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, sampler=SubsetRandomSampler(train_indices, True))
        else: 
            test_indices, _ = self.get_same_index(self.test_dataset.targets, class_id, mode="test")
            data_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, sampler=SubsetRandomSampler(test_indices, False))
            
        return data_loader

    def _setup_data(self, datasets, path, random_order=False, seed=1, increment=10, validation_split=0.):
        self.increments = []
        self.class_order = []
        
        trsf_train = transforms.Compose(self.train_transforms)
        try:
            trsf_mata = transforms.Compose(self.meta_transforms)
        except:
            trsf_mata = transforms.Compose(self.train_transforms)
            
        trsf_test = transforms.Compose(self.common_transforms)
        
        current_class_idx = 0  # When using multiple datasets
        for dataset in datasets:
            if(self.dataset_name=="imagenet"):
                train_dataset = dataset.base_dataset(root=path, split='train', download=False, transform=trsf_train)
                test_dataset = dataset.base_dataset(root=path, split='val', download=False, transform=trsf_test)
                
            elif(self.dataset_name=="cub200" or self.dataset_name=="cifar100" or self.dataset_name=="mnist"  or self.dataset_name=="caltech101"  or self.dataset_name=="omniglot"  or self.dataset_name=="celeb"):
                train_dataset = dataset.base_dataset(root=path, train=True, download=True, transform=trsf_train)
                test_dataset = dataset.base_dataset(root=path, train=False, download=True, transform=trsf_test)

            elif(self.dataset_name=="svhn"):
                train_dataset = dataset.base_dataset(root=path, split='train', download=True, transform=trsf_train)
                test_dataset = dataset.base_dataset(root=path, split='test', download=True, transform=trsf_test)
                train_dataset.targets = train_dataset.labels
                test_dataset.targets = test_dataset.labels

            elif(self.dataset_name=="rmnist"):
                train_dataset = dataset.base_dataset(root=path, train = True, transform = trsf_train)
                test_dataset = dataset.base_dataset(root=path, train = False, transform = trsf_test)
                
                
            order = [i for i in range(self.args.num_class)]
            if random_order:
                random.seed(seed)  
                random.shuffle(order)
            elif dataset.class_order is not None:
                order = dataset.class_order
                
            for i,t in enumerate(train_dataset.targets):
                train_dataset.targets[i] = order[t]
            for i,t in enumerate(test_dataset.targets):
                test_dataset.targets[i] = order[t]
            self.class_order.append(order)

            self.increments = [increment for _ in range(len(order) // increment)]

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    @staticmethod
    def _map_new_class_index(y, order):
        """Transforms targets for new class order."""
        return np.array(list(map(lambda x: order.index(x), y)))
    
    
    def get_memory(self, memory, for_memory, seed=1):
        random.seed(seed)
        memory_per_task = self.args.memory // (self.args.sess+1)
        self._data_memory, self._tasks_memory = np.array([]), np.array([])
        mu = 1
        
        #update old memory
        if(memory is not None):
            data_memory, tasks_memory = memory
            data_memory = np.array(data_memory, dtype="int32")
            tasks_memory = np.array(tasks_memory, dtype="int32")
            for task_idx in range(self.args.sess + 1):
                idx = np.where(tasks_memory==task_idx)[0][:memory_per_task]
                self._data_memory = np.concatenate([self._data_memory, np.tile(data_memory[idx], (mu,))   ])
                self._tasks_memory = np.concatenate([self._tasks_memory, np.tile(tasks_memory[idx], (mu,))    ])
                
                
        #add new classes to the memory
        new_indices, new_tasks = for_memory

        new_indices = np.array(new_indices, dtype="int32")
        new_tasks = np.array(new_tasks, dtype="int32")
        idx = np.where(new_tasks==self.args.sess)[0][:memory_per_task]
        self._data_memory = np.concatenate([self._data_memory, np.tile(new_indices[idx],(mu,))   ])
        self._tasks_memory = np.concatenate([self._tasks_memory, np.tile(new_tasks[idx],(mu,))    ])
            
        print(f'Length of memory: {len(self._data_memory)}')
        tasks, counts = np.unique(self._tasks_memory, return_counts = True) 
        task_count_dict = {int(i):int(j) for i, j in zip(tasks, counts)}
        print(f'Samples per task in memory: {task_count_dict}')

        return list(self._data_memory.astype("int32")), list(self._tasks_memory.astype("int32"))
    
def _get_datasets(dataset_names):
    return [_get_dataset(dataset_name) for dataset_name in dataset_names.split("-")]


def _get_dataset(dataset_name):
    dataset_name = dataset_name.lower().strip()

    if dataset_name == "rmnist":
        return iRMNIST 
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))

class DataHandler:
    base_dataset = None
    train_transforms = []
    mata_transforms = [transforms.ToTensor()]
    common_transforms = [transforms.ToTensor()]
    class_order = None

class iRMNIST(DataHandler):
    base_dataset = RMNIST
    train_transforms = [transforms.Normalize((0.1307,), (0.3081,)) ]
    common_transforms = [transforms.Normalize((0.1307,), (0.3081,))]
