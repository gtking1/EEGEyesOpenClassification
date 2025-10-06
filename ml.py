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
import torch.utils.data as data_utils
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

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
        return {'window': torch.tensor(window.values).unsqueeze(0).to(torch.float32), 
                'labels': torch.tensor(labels).unsqueeze(0).to(torch.float32)}#.values)}

class Normalize(object):
    def __call__(self, sample):
        window, labels = sample['window'], sample['labels']
        return {'window': torch.abs(window), 
                'labels': labels}

class EEGDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.csv_file = pd.read_csv(csv_file, delimiter='\t', header=None)
        self.transform = transform
    
    def __len__(self):
        return len(self.csv_file)
    
    def __getitem__(self, idx):
        # if idx >= len(self.csv_file) / 1250:
        #     return

        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        window = self.csv_file.iloc[idx*1250:idx*1250+1250, 7]
        labels = [self.csv_file.iloc[idx*1250, 32]]#:idx*1250+1250, 32] # remove brackets for timestep prediction
        sample = {'window': window, 'labels': labels}

        if self.transform:
            sample = self.transform(sample)
        
        return sample

eeg_dataset = EEGDataset(csv_file='./test.csv', transform=transforms.Compose([ToTensor(),
                                                                              Normalize()]))
eeg_dataset = data_utils.Subset(eeg_dataset, range(0, 48))

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(eeg_dataset, [0.6, 0.2, 0.2])

#print(train_dataset[0]['window'])
#print(train_dataset[0]['labels'].shape)
print(len(train_dataset), len(val_dataset), len(test_dataset))

# for i, sample in enumerate(eeg_dataset):
#     # if i == 48: #58751
#     #     break
    
#     # print(type(sample), type(sample['window']), type(sample['labels']))
#     print(i, sample['window'].shape, sample['labels'].shape)

#     #print(sample['window'], sample['labels'])

#     # if i == 0: #58750
#     #     print(sample['window'], sample['labels'])

train_dataloader = DataLoader(train_dataset, batch_size=4, drop_last=True, shuffle=True, num_workers=0)
val_dataloader = DataLoader(val_dataset, batch_size=4, drop_last=True, shuffle=False, num_workers=0)
test_dataloader = DataLoader(test_dataset, batch_size=4, drop_last=True, shuffle=False, num_workers=0)
#test_dataloader = DataLoader(eeg_dataset, batch_size=4, shuffle=False, num_workers=0)
print(len(train_dataloader), len(val_dataloader), len(test_dataloader))

# for i_batch, sample_batched, in enumerate(val_dataloader):
#     # if i_batch == 13:
#     #     break

#     print(i_batch, type(sample_batched), type(sample_batched['window']), sample_batched['window'].size())
#     print(sample_batched['labels'])

import torch.nn as nn
import torch.nn.functional as F

# PyTorch models inherit from torch.nn.Module
class EEGEyesOpenCloseClassifier(nn.Module):
    def __init__(self):
        super(EEGEyesOpenCloseClassifier, self).__init__()
        # self.conv1 = nn.Conv1d(4, 8, 5, stride=2)
        # self.pool = nn.MaxPool1d(2, 2)
        # self.conv2 = nn.Conv1d(8, 16, 5)
        # self.fc1 = nn.Linear(612, 306)
        # self.fc2 = nn.Linear(306, 153)
        # self.fc3 = nn.Linear(153, 2)

        self.conv1 = nn.Conv1d(1, 1, 5, stride=5)
        self.pool1 = nn.MaxPool1d(5)
        self.fc1 = nn.Linear(50, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = x.view(4, 612)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        print(x)
        return x


model = EEGEyesOpenCloseClassifier()

loss_fn = torch.nn.BCELoss()

# Optimizers specified in the torch.optim package
optimizer = torch.optim.Adam(model.parameters())

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(train_dataloader):
        # Every data instance is an input + label pair
        inputs = data['window']
        labels = data['labels']

        #print(inputs, labels)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 2 == 1:
            last_loss = running_loss / 1000 # loss per batch
            #print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(train_dataloader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/trainer_{}'.format(timestamp))
epoch_number = 0

EPOCHS = 5

best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    #print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)

    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(val_dataloader):
            vinputs = vdata['window']
            vlabels = vdata['labels']
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    #print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    epoch_number += 1

writer.close()

model.eval()
with torch.no_grad():
    for i, vdata in enumerate(test_dataset):
        vinputs = vdata['window']
        vlabels = vdata['labels']
        voutputs = model(vinputs)
        print(vlabels)
        print(voutputs)

# Normalize (abs value?) data, correct outputs