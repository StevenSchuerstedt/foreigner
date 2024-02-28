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

f = AttributionHead("checkpoints/checkpoint-22500")
f_tilde = AttributionHead("checkpoints/checkpoint-22500")
tokenizer.enable_padding(length=512, pad_id=f.transformer.config.pad_token_id)

f.load("checkpoint_attribute/checkpoint_attribute_f", "checkpoint_attribute/transformer_f")
f_tilde.load("checkpoint_attribute/checkpoint_attribute_f_tilde", "checkpoint_attribute/transformer_f_tilde")

#load data 

composers = ['bach',
              'beethoven',
                 'chopin',
                   'grieg',
                     'haydn',
                       'liszt',
                         'mendelssohn',
                           'rachmaninov'
                          ]

# load dataset
data_files = {
    "generated_bach": "DATA/attribution/generated/generated_bach.txt",
    "generated_beethoven": "DATA/attribution/generated/generated_beethoven.txt",
     "generated_chopin": "DATA/attribution/generated/generated_chopin.txt",
     "generated_grieg": "DATA/attribution/generated/generated_grieg.txt",
     "generated_haydn": "DATA/attribution/generated/generated_haydn.txt",
     "generated_liszt": "DATA/attribution/generated/generated_liszt.txt",
     "generated_mendelssohn": "DATA/attribution/generated/generated_mendelssohn.txt",
     #"generated_rachmaninov": "DATA/attribution/generated/generated_rachmaninov.txt",

    "input_bach": "DATA/attribution/input/input_bach.txt",
    "input_beethoven": "DATA/attribution/input/input_beethoven.txt",
     "input_chopin": "DATA/attribution/input/input_chopin.txt",
     "input_grieg": "DATA/attribution/input/input_grieg.txt",
     "input_haydn": "DATA/attribution/input/input_haydn.txt",
     "input_liszt": "DATA/attribution/input/input_liszt.txt",
     "input_mendelssohn": "DATA/attribution/input/input_mendelssohn.txt",
     #"input_rachmaninov": "DATA/attribution/input/input_rachmaninov_test.txt"
          }

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


dataset = datasets.load_dataset("text", data_files={
    "generated_rachmaninov": "DATA/attribution/generated/generated_rachmaninov.txt",
    "input_rachmaninov": "DATA/attribution/input/input_rachmaninov.txt"
    })
tokenized_datasets2 = dataset.map(
    tokenize_function, batched=True, remove_columns=["text"])

#alternativ concatenate_datasets
#datasets.combine

tokenized_datasets = {**tokenized_datasets, **tokenized_datasets2}

#start with only input data, generated data, x and x_tilde?? calculate s in loss function? or beforehand


#choose random composer
composer = random.choice(composers)
print(composer)

x_tilde = random.choice(tokenized_datasets['generated_' + composer])

#print(x_tilde_index)

input_ids_x_tilde = torch.tensor([x_tilde['input_ids']])
attention_x_tilde = torch.tensor([x_tilde['attention_mask']])
#print(input_ids_x_tilde)


feature_vec_x_tilde = f_tilde(input_ids_x_tilde, attention_x_tilde)


x_tilde2 = random.choice(tokenized_datasets['input_' + composer])

#print(x_tilde_index)

input_ids_x_tilde2 = torch.tensor([x_tilde2['input_ids']])
attention_x_tilde2 = torch.tensor([x_tilde2['attention_mask']])
#print(input_ids_x_tilde)


feature_vec_x_tilde2 = f(input_ids_x_tilde2, attention_x_tilde2)

ground_truth = np.dot(feature_vec_x_tilde2[0].detach().numpy(), feature_vec_x_tilde[0].detach().numpy())
#calculate similarity scores
print("ground_truth: ", ground_truth)
#list of similarit scores, => s = dot(F(x), F_tilde(x_tilde))
s = [ground_truth]

for c in composers:
   # if c == composer:
    #    continue
    for i in range(1):
      x = random.choice(tokenized_datasets['input_' + c])
      input_ids_x = torch.tensor([x['input_ids']])
      attention_x = torch.tensor([x['attention_mask']])
      feature_vec_x = f(input_ids_x, attention_x)

      similarity_score = np.dot(feature_vec_x[0].detach().numpy(), feature_vec_x_tilde[0].detach().numpy())
          
      s.append(similarity_score)
   


model = ProbabilityScore()

scores = model(torch.tensor(s))
print(s)
print(sum(scores))
print(scores * 100)