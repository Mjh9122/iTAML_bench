import torch as t
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F
from tqdm import tqdm
from idatasets.RMNIST import RMNIST
import matplotlib.pyplot as plt



train_dataset = RMNIST(root = "../Datasets/RMNIST/", train = True, transform = transforms.Normalize((0.1307,), (0.3081,)))
test_dataset = RMNIST(root = "../Datasets/RMNIST/", train = False, transform = transforms.Normalize((0.1307,), (0.3081,)))

train_loader = DataLoader(train_dataset, batch_size = 256, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = 256, shuffle = False)

model = nn.Sequential(
    nn.Flatten(start_dim = 1),
    nn.Linear(784, 100), 
    nn.ReLU(),             
    nn.Linear(100, 100),  
    nn.ReLU(),         
    nn.Linear(100, 10, bias=False) 
)

model.to('cuda')

optim = Adam(model.parameters(), .01) 

losses = []
for batch_idx, (inputs, targets, _ )in tqdm(enumerate(train_loader), total = len(train_loader)):
    model.train()
    optim.zero_grad()
    
    inputs = inputs.to('cuda')
    targets = targets.to('cuda')
    outputs = model(inputs)
    loss = F.cross_entropy(outputs, targets)
    loss.backward()
    optim.step()
    
    losses.append(loss.item())
    
correct = 0
total = 0

for inputs, targets, _ in tqdm(test_loader):  
    model.eval()
    inputs = inputs.to('cuda')   
    targets = targets.to('cuda')  
    outputs = model(inputs)
    _, predicted = outputs.max(1)
    total += targets.size(0)
    correct += predicted.eq(targets).sum().item()

accuracy = 100. * correct / total
print(f'Test Accuracy after batch {batch_idx}: {accuracy:.6f}%')