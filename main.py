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
import torch.nn.functional as F

dirname = os.path.dirname(__file__)

import gpt2_composer

composers = ['bach', 'beethoven', 'chopin', 'grieg', 'haydn', 'liszt', 'mendelssohn', 'rachmaninov']

DATASET_PATH = os.path.join(dirname, 'DATA\\chopin')
TOKENS_PATH = os.path.join(dirname, 'DATA\\TOKENS')
#midi_files = glob.glob(os.path.join(DATASET_PATH, "*.mid"))

from AttributionHead import AttributionHead


model = AttributionHead("checkpoints/checkpoint-20000")

model.load('test/head')













# similarity_scores:  [659232.6, 491974.1, 425067.06, 272199.44, 544362.1, 323409.12, 472035.25, 303624.7, 557990.1, 491712.9, 227400.45, 351568.06, 477384.62, 323115.4, 555855.44, 337630.12]
# similarity_scores sorted:  [659232.6  557990.1  555855.44 544362.1  491974.1  491712.9  477384.62
#  472035.25 425067.06 351568.06 337630.12 323409.12 323115.4  303624.7
#  272199.44 227400.45]
# P: tensor([0.1121, 0.0592, 0.0592, 0.0592, 0.0592, 0.0592, 0.0592, 0.0592, 0.0592,
#         0.0592, 0.0592, 0.0592, 0.0592, 0.0592, 0.0592, 0.0592],
#        dtype=torch.float64)

# sorted = np.array([659232.6,  557990.1,  555855.44, 544362.1,  491974.1,  491712.9, 477384.62 ,472035.25, 425067.06, 351568.06, 337630.12, 323409.12, 323115.4,  303624.7, 272199.44, 227400.45])

# n = torch.relu(torch.exp(torch.tensor(557990.1 - sorted[0])/1) - 0)
# print("n", n)        
# A = torch.exp( torch.tensor((sorted - sorted[0]))/1 )
# print("A", A) 
# B =  A - 0
# d = torch.sum(B)
# print("d", d) 
        
# P = n / d

# print("P", P) 


# print(F.softplus(torch.tensor(0.)))

# print(F.softplus(torch.tensor(1.)))