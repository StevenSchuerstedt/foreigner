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
import random
from AttributionHead import AttributionHead
import math

#load model

tokenizer = gpt2_composer.load_tokenizer("")
f = AttributionHead.from_pretrained('checkpoint_attribute_f_jan')
f_tilde = AttributionHead.from_pretrained('checkpoint_attribute_f_tilde_jan')

#load data 
data_files = {"generated": "DATA/attribution_generated.txt", "input": "DATA/attribution_input.txt"}
dataset = datasets.load_dataset("text", data_files=data_files)


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


#start with only input data, generated data, x and x_tilde?? calculate s in loss function? or beforehand


x_tilde_index = random.choice(range(len(data_x_tilde)))

x_tilde = data_x_tilde[x_tilde_index]

print(x_tilde_index)

input_ids_x_tilde = torch.tensor([x_tilde['input_ids']])

print(input_ids_x_tilde)


feature_vec_x_tilde = f_tilde(input_ids_x_tilde)

#calculate similarity scores

#list of similarit scores, => s = dot(F(x), F_tilde(x_tilde))
s = []

#list of ground truths (0 = false, 1 = true)
t = []

for j in range(2):
    for i in range(8):
        x_index = random.choice(range(100)) + i * 100 
        input_ids_x = torch.tensor([data_x[x_index]['input_ids']])
        feature_vec_x = f(input_ids_x)

        similarity_score = np.dot(feature_vec_x[0].detach().numpy(), feature_vec_x_tilde[0].detach().numpy())
    
        s.append(similarity_score)
        if(math.floor(x_tilde_index / 100) == math.floor(x_index / 100)):
            t.append(1)
        else:
            t.append(0)

        print("x_index:", x_index)

 

datapair = {"x_tilde": x_tilde_index,
       "similarity_scores": s,
       "ground_truths": t
       }

print(datapair)

datapairs = []


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

#S is identicator function
#S = ...


#TODO: does this work??
optimizer = torch.optim.SGD([tau, lámbda], lr=0.005)


#training loop
loss = klLoss(S, P)

optimizer.zero_grad()

loss.backward()

optimizer.step()

