import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations, WindowOperations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import os
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

# transform = transforms.Compose(
#     [transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))])

# # Create datasets for training & validation, download if necessary
# training_set = torchvision.datasets.FashionMNIST('./data', train=True, transform=transform, download=True)
# validation_set = torchvision.datasets.FashionMNIST('./data', train=False, transform=transform, download=True)

# # Create data loaders for our datasets; shuffle for training, not for validation
# training_loader = torch.utils.data.DataLoader(training_set, batch_size=4, shuffle=True)
# validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=4, shuffle=False)

# # Class labels
# classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
#         'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

# # Report split sizes
# print('Training set has {} instances'.format(len(training_set)))
# print('Validation set has {} instances'.format(len(validation_set)))

# dataiter = iter(training_loader)
# images, labels = next(dataiter)

class ToTensor(object):
    def __call__(self, sample):
        window, labels = sample['window'], sample['labels']
        return {'window': torch.tensor(window.values), 'labels': torch.tensor(labels.values)}

class EEGDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.csv_file = pd.read_csv(csv_file, delimiter='\t', header=None)
        self.transform = transform
    
    def __len__(self):
        return len(self.csv_file)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        window = self.csv_file.iloc[idx:idx+1250, 7]
        labels = self.csv_file.iloc[idx:idx+1250, 32]
        sample = {'window': window, 'labels': labels}

        if self.transform:
            sample = self.transform(sample)
        
        return sample

eeg_dataset = EEGDataset(csv_file='./test.csv', transform=transforms.Compose([ToTensor()]))

for i, sample in enumerate(eeg_dataset):
    if i == 58751: #58751
        break
    
    # print(type(sample), type(sample['window']), type(sample['labels']))
    # print(i, sample['window'].shape, sample['labels'].shape)

    # if i == 0: #58750
    #     print(sample['window'], sample['labels'])

dataloader = DataLoader(eeg_dataset, batch_size=4, drop_last=True, shuffle=True, num_workers=0)

for i_batch, sample_batched, in enumerate(dataloader):
    if i_batch == 10:
        break

    print(i_batch, type(sample_batched), type(sample_batched['window']), sample_batched['window'].size())