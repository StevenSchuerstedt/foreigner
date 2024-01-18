from typing import Optional
import numpy as np
import note_seq
import transformers
import torch
import datasets
import gpt2_composer
from torch import nn 
import torch.nn.functional as F
from transformers import Trainer


#load data 


#start with only input data, generated data, x and x_tilde?? calculate s in loss function? or beforehand

#list of similarit scores, => s = dot(F(x), F_tilde(x_tilde))
s = []

def klLoss(input, target):
    #TODO: add expectation value over all x_tilde??
    return F.kl_div(input, target)


tau = torch.nn.Parameter(torch.Tensor([1.0]))
lámbda = torch.nn.Parameter(torch.Tensor([0.0]))

#TODO: calculate Pr with tau und lambda


similarity_count = 8


#calulate P distribution
P = torch.tensor(np.ones[similarity_count])

for i in range(similarity_count):
    n = torch.relu(torch.exp((s[i] - s[0])/tau) - lámbda)

    d = torch.sum(torch.relu(torch.exp( (s - s[0])/tau ) - lámbda))

    P[i] = n / d

#construct ground truth distribution
S = torch.tensor(np.ones[similarity_count])

#TODO: 
#S = ...


#TODO: does this work??
optimizer = torch.optim.SGD([tau, lámbda], lr=0.5)


#training loop
loss = klLoss(S, P)

optimizer.zero_grad()

loss.backward()

optimizer.step()

