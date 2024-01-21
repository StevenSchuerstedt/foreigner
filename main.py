print("hello world")

import os
import glob
from copy import deepcopy
import json
import sklearn
import torch

from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
import math
import note_seq

dirname = os.path.dirname(__file__)

import gpt2_composer

composers = ['bach', 'beethoven', 'chopin', 'grieg', 'haydn', 'liszt', 'mendelssohn', 'rachmaninov']

DATASET_PATH = os.path.join(dirname, 'DATA\\chopin')
TOKENS_PATH = os.path.join(dirname, 'DATA\\TOKENS')
#midi_files = glob.glob(os.path.join(DATASET_PATH, "*.mid"))


# t = torch.tensor(np.ones([8, 512]))
# s = torch.tensor(np.ones([8, 512]))

# i = 3

# A = torch.dot(t[i], s[i])

# B = torch.matmul(t[i], s.transpose(0,1))

# print(A)
# C = torch.logsumexp(A, 0)
# print(C)

# compute NTXENT Loss
      #     A = torch.exp(torch.dot(t[i], s[i]) / v)
      #     B = torch.mul(t[i], s) / v
      #     C = torch.mul(t, s[i]) / v

      #     L_cont += -( (torch.log(A) - torch.logsumexp(B)) + (torch.log(A) - torch.logsumexp(C)))


# s = torch.tensor(np.ones([10]))

# print(s)

# t = torch.sum(torch.relu(torch.exp(s - 0.3)))

# print(t)

print(math.floor(499 / 100))