import argparse
import logging
import time
from collections import deque

import brainflow
import pyqtgraph as pg
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations, WindowOperations
from pyqtgraph.Qt import QtWidgets, QtCore
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = DataFilter.read_file('0926_480_seconds.csv') # rows of features, columns of data
df = pd.DataFrame(np.transpose(data)) # rows of data, columns of features
# print('Data From the File')
# print(data[:250])

exg_channels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
sampling_rate = 125
curves = list()
plots = list()

# for i in range(len(exg_channels)):
#     p = win.addPlot(row=i, col=0)
#     p.showAxis('left', False)
#     p.setMenuEnabled('left', False)
#     p.showAxis('bottom', False)
#     p.setMenuEnabled('bottom', False)
#     if i == 0:
#         p.setTitle('TimeSeries Plot')
#     plots.append(p)
#     curve = p.plot()
#     curves.append(curve)

timestampChannel = -3
channel = 1
print(df.head(10))
# for i in range(16):
#     DataFilter.detrend(data[i + 1], DetrendOperations.LINEAR.value)
#     DataFilter.perform_bandpass(data[i + 1], sampling_rate, 3.0, 45.0, 2,
#                                 FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
#     DataFilter.perform_bandstop(data[i + 1], sampling_rate, 48.0, 52.0, 2,
#                                 FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
#     DataFilter.perform_bandstop(data[i + 1], sampling_rate, 58.0, 62.0, 2,
#                                 FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)

import matplotlib.pylab as pylab
params = {'axes.labelsize': 'xx-small',
          'xtick.labelsize':'xx-small',
          'ytick.labelsize':'xx-small'}
pylab.rcParams.update(params)

fig, axes = plt.subplots(4, 4, figsize=(12, 10))
firstTimestamp = data[timestampChannel][0]
for i in range(4):
    for j in range(4):
        axes[i, j].set_xlabel(f"Channel { i * 4 + j + 1 }", )
        axes[i, j].plot((data[timestampChannel] - firstTimestamp).tolist(), data[i * 4 + j + 1].tolist())

# x = [1758908771.210296, 1758908774.499588, 1758908774.503593, 1758908777.500031]
# point = x[1]
# ymin = min(data[channel])
# index = df.where(df[30] == point).first_valid_index()
# ymax = df[channel][index]
# plt.vlines(x, ymin, ymax, colors='red', linestyles='dashed')
plt.show()
print(df.shape)
#print("Expected number of samples:", self.update_speed_ms / 1000 * self.sampling_rate * 2 * self.num_each_seg)

timestamps = data[timestampChannel].tolist()
labels = data[-1].tolist()

print(timestamps[-1] - timestamps[0])
truthcount = 0
falsecount = 0

for i in range(len(labels)):
    truthcount += labels[i]
    if not labels[i]:
        falsecount += 1

print(truthcount, falsecount)

block = 0

for i in range(1, len(labels)):
    if not labels[i] == labels[i - 1]:
        print(block + 1)
        block = 0
    else:
        block += 1

print(block + 1)
    