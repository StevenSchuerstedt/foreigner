import copy
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
from ProbabilityScore import ProbabilityScore
import math

#load model

tokenizer = gpt2_composer.load_tokenizer("")
tokenizer.enable_padding(length=512)
f = AttributionHead("checkpoints/checkpoint-22500")
f_tilde = AttributionHead("checkpoints/checkpoint-22500")


f.load("checkpoint_attribute/f", "checkpoint_attribute/transformer_f")
f_tilde.load("checkpoint_attribute/f_tilde", "checkpoint_attribute/transformer_f_tilde")

#load data 
data_files = {"generated": "DATA/attribution_generated.txt", "input": "DATA/attribution_input.txt"}
dataset = datasets.load_dataset("text", data_files=data_files)


def tokenize_function(examples):
    outputs = tokenizer.encode_batch(examples["text"])
    example = {
        "input_ids": [c.ids for c in outputs],
        "attention_mask" : [c.attention_mask for c in outputs]
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

#print(x_tilde_index)

input_ids_x_tilde = torch.tensor([x_tilde['input_ids']])
attention_x_tilde = torch.tensor([x_tilde['attention_mask']])
#print(input_ids_x_tilde)


feature_vec_x_tilde = f_tilde(input_ids_x_tilde, attention_x_tilde)

#calculate similarity scores

#list of similarit scores, => s = dot(F(x), F_tilde(x_tilde))
s = []


for i in range(8):
    x_index = random.choice(range(100)) + i * 100 
    input_ids_x = torch.tensor([data_x[x_index]['input_ids']])
    attention_x = torch.tensor([data_x[x_index]['attention_mask']])
    feature_vec_x = f(input_ids_x, attention_x)

    similarity_score = np.dot(feature_vec_x[0].detach().numpy(), feature_vec_x_tilde[0].detach().numpy())
        
    s.append(similarity_score)
   


model = ProbabilityScore()

scores = model(torch.tensor(s))
print(s)
print(x_tilde_index)
print(scores * 100)