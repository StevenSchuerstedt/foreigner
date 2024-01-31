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

DATASET_PATH = os.path.join(dirname, 'DATA\\chopin')
TOKENS_PATH = os.path.join(dirname, 'DATA\\TOKENS')
#midi_files = glob.glob(os.path.join(DATASET_PATH, "*.mid"))

from AttributionHead import AttributionHead


f = AttributionHead("checkpoints/checkpoint-22500_new_basemodel")
f_tilde = AttributionHead("checkpoints/checkpoint-22500_new_basemodel")

f.load("checkpoint_attribute/f", "checkpoint_attribute/transformer_f")
f_tilde.load("checkpoint_attribute/f_tilde", "checkpoint_attribute/transformer_f_tilde")

tokenizer = gpt2_composer.load_tokenizer("")

# load dataset
data_files = {"generated": "DATA/attribution_generated_old.txt", "input": "DATA/attribution_input_old.txt", "test_generated": "DATA/attribution_generated_old.txt", "test_input": "DATA/attribution_input_old.txt"}
dataset = datasets.load_dataset("text", data_files=data_files)
tokenizer.enable_padding(length=512)

def tokenize_function(examples):
    outputs = tokenizer.encode_batch(examples["text"])
    example = {
        "input_ids": [c.ids for c in outputs]
    }
    # The 🤗 Transformers library apply the shifting to the right, so we don't need to do it manually.
    #example["labels"] = example["input_ids"].copy()

    #example["x"] = example["train"].copy()
    #example["x^~"] = example["generated"].copy()
    return example


tokenized_datasets = dataset.map(
    tokenize_function, batched=True, remove_columns=["text"])
data_x = tokenized_datasets["input"]
data_x_tilde = tokenized_datasets["generated"]

device = 'cpu'

def ntxent(t, s, v):
       #iterate over all is
       L_cont = 0
       for i in range(len(t)):
  
            # compute NTXENT Loss
            A = torch.dot(t[i], s[i]) / v
            B = torch.matmul(t[i], s.transpose(0,1)) / v
            C = torch.matmul(t, s[i]) / v

        
            A1 = A
            B1 = torch.logsumexp(B, dim=0)
            C1 = torch.logsumexp(C, dim=0)

            L_cont += -( (A1 - B1) + (A1 - C1))
          
       return L_cont/len(t)


t = f(torch.tensor(data_x['input_ids']).to(device))
s = f_tilde(torch.tensor(data_x_tilde['input_ids']).to(device))

loss = ntxent(t, s, 1)

print(loss)