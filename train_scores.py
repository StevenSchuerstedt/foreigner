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

#list of similarit scores, => s = dot(F(x), F_tilde(x_tilde))
s = []

def klLoss(input, target):
    #TODO: add expectation value of x_tilde??
    return F.kl_div(input, target)


tau = torch.nn.Parameter(torch.Tensor([1.0]))
lámbda = torch.nn.Parameter(torch.Tensor([0.0]))

#TODO: calculate Pr with tau und lambda