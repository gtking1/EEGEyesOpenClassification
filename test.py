import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms

# PyTorch TensorBoard support
from datetime import datetime

import os
import torch.utils.data as data_utils
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
import torch.nn as nn
import torch.nn.functional as F

# m = nn.Conv1d(4, 4, 50, stride=2)
# input = torch.randn(4, 1250)
# output = m(input)
# print(output.shape)

# Example of target with class indices
loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
print(input)
target = torch.empty(3, dtype=torch.long).random_(5)
print(target)
# output = loss(input, target)
# output.backward()
# Example of target with class probabilities
# input = torch.randn(3, 5, requires_grad=True)
# target = torch.randn(3, 5).softmax(dim=1)
# output = loss(input, target)
# output.backward()

test = torch.tensor([1, 2, 3, 4, 5]).to(torch.float32)
test = test.unsqueeze(0)
test = F.normalize(test)
print(test)