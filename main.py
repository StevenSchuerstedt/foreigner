print("hello world")

import os
import glob
from copy import deepcopy
import json
import sklearn
import torch
import datasets

from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
import math
import note_seq
import torch.nn.functional as F

dirname = os.path.dirname(__file__)

import gpt2_composer

composers = ['bach', 'beethoven', 'chopin', 'grieg', 'haydn', 'liszt', 'mendelssohn', 'rachmaninov']

data_files = {"input": "DATA/data.json"}
dataset = datasets.load_dataset("json", data_files=data_files)

print(dataset['input']['bach'][1])


DATASET_PATH = os.path.join(dirname, 'DATA\\chopin')
TOKENS_PATH = os.path.join(dirname, 'DATA\\TOKENS')
#midi_files = glob.glob(os.path.join(DATASET_PATH, "*.mid"))

# S tensor([0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.5000, 0.0000, 0.0000,
#         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.5000, 0.0000],
#        dtype=torch.float64)
# P tensor([0.0729, 0.0538, 0.0430, 0.0805, 0.0662, 0.0673, 0.0588, 0.0621, 0.0726,
#         0.0558, 0.0646, 0.0628, 0.0580, 0.0609, 0.0609, 0.0597],
#        dtype=torch.float64, grad_fn=<CopySlices>)
# kl: tensor(-0.1765, dtype=torch.float64, grad_fn=<KlDivBackward0>)

f = ['hu', 'ha']
d = {}

for i in f:
    d[i] = 'toll'

print(d)