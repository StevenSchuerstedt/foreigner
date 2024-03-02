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
    tokenize_function, batched=True)


dataset = datasets.load_dataset("text", data_files={
    "generated_rachmaninov": "DATA/attribution/generated/generated_rachmaninov.txt",
    "input_rachmaninov": "DATA/attribution/input/input_rachmaninov.txt"
    })
tokenized_datasets2 = dataset.map(
    tokenize_function, batched=True)

#alternativ concatenate_datasets
#datasets.combine

tokenized_datasets = {**tokenized_datasets, **tokenized_datasets2}

#start with only input data, generated data, x and x_tilde?? calculate s in loss function? or beforehand


#choose specific piece
#chopin
#tokens = "PIECE_START TRACK_START INST=0 BAR_START NOTE_ON=81 NOTE_ON=94 TIME_DELTA=24 NOTE_OFF=81 NOTE_OFF=94 NOTE_ON=81 NOTE_ON=93 TIME_DELTA=12 NOTE_OFF=81 NOTE_OFF=93 NOTE_ON=79 NOTE_ON=91 TIME_DELTA=6 NOTE_OFF=79 NOTE_OFF=91 NOTE_ON=78 NOTE_ON=90 TIME_DELTA=6 NOTE_OFF=78 NOTE_OFF=90 BAR_END BAR_START NOTE_ON=79 NOTE_ON=91 TIME_DELTA=6 NOTE_OFF=79 NOTE_OFF=91 NOTE_ON=78 NOTE_ON=90 TIME_DELTA=6 NOTE_OFF=78 NOTE_OFF=90 NOTE_ON=76 NOTE_ON=88 TIME_DELTA=6 NOTE_OFF=76 NOTE_OFF=88 NOTE_ON=74 NOTE_ON=86 TIME_DELTA=6 NOTE_OFF=74 NOTE_OFF=86 NOTE_ON=72 NOTE_ON=84 TIME_DELTA=6 NOTE_OFF=72 NOTE_OFF=84 NOTE_ON=67 NOTE_ON=79 TIME_DELTA=6 NOTE_OFF=67 NOTE_OFF=79 NOTE_ON=66 NOTE_ON=78 TIME_DELTA=6 NOTE_OFF=66 NOTE_OFF=78 NOTE_ON=66 NOTE_ON=78 TIME_DELTA=6 NOTE_OFF=66 NOTE_OFF=78 BAR_END BAR_START NOTE_ON=67 NOTE_ON=79 TIME_DELTA=6 NOTE_OFF=67 NOTE_OFF=79 NOTE_ON=66 NOTE_ON=78 TIME_DELTA=5 NOTE_OFF=66 NOTE_OFF=78 NOTE_ON=64 NOTE_ON=76 TIME_DELTA=6 NOTE_OFF=64 NOTE_OFF=76 NOTE_ON=62 NOTE_ON=74 TIME_DELTA=6 NOTE_OFF=62 NOTE_OFF=74 NOTE_ON=64 NOTE_ON=76 TIME_DELTA=6 NOTE_OFF=64 NOTE_OFF=76 NOTE_ON=66 NOTE_ON=78 TIME_DELTA=6 NOTE_OFF=66 NOTE_OFF=78 NOTE_ON=67 NOTE_ON=79 TIME_DELTA=6 NOTE_OFF=67 NOTE_OFF=79 NOTE_ON=59 NOTE_ON=71 TIME_DELTA=15 NOTE_OFF=59 NOTE_OFF=71 BAR_END BAR_START NOTE_ON=61 NOTE_ON=73 TIME_DELTA=6 NOTE_OFF=61 NOTE_OFF=73 NOTE_ON=64 NOTE_ON=76 TIME_DELTA=3 NOTE_OFF=64 NOTE_OFF=76 NOTE_ON=67 NOTE_ON=79 TIME_DELTA=6 NOTE_OFF=67 NOTE_OFF=79 NOTE_ON=65 NOTE_ON=77 TIME_DELTA=6 NOTE_OFF=65 NOTE_OFF=77 NOTE_ON=67 NOTE_ON=79 TIME_DELTA=6 NOTE_OFF=67 NOTE_OFF=79 NOTE_ON=65 NOTE_ON=77 TIME_DELTA=6 NOTE_OFF=65 NOTE_OFF=77 NOTE_ON=67 NOTE_ON=79 TIME_DELTA=6 NOTE_OFF=67 NOTE_OFF=79 NOTE_ON=67 NOTE_ON=76 TIME_DELTA=6 NOTE_OFF=67 NOTE_OFF=76 BAR_END TRACK_END TRACK_START INST=0 BAR_START NOTE_ON=50 TIME_DELTA=6 NOTE_OFF=50 NOTE_ON=57 NOTE_ON=69 TIME_DELTA=6 NOTE_OFF=57 NOTE_OFF=69 NOTE_ON=55 TIME_DELTA=6 NOTE_OFF=55 NOTE_ON=43 TIME_DELTA=6 NOTE_OFF=43 NOTE_ON=50 TIME_DELTA=6 NOTE_OFF=50 NOTE_ON=57 NOTE_ON=66 TIME_DELTA=6 NOTE_OFF=57 NOTE_OFF=66 NOTE_ON=54 TIME_DELTA=6 NOTE_OFF=54 NOTE_ON=43 TIME_DELTA=6 NOTE_OFF=43 BAR_END BAR_START NOTE_ON=59 NOTE_ON=66 TIME_DELTA=6 NOTE_OFF=59 NOTE_OFF=66 NOTE_ON=54 TIME_DELTA=6 NOTE_OFF=54 NOTE_ON=55 NOTE_ON=59 NOTE_ON=64 TIME_DELTA=6 NOTE_OFF=55 NOTE_OFF=59 NOTE_OFF=64 NOTE_ON=45 TIME_DELTA=6 NOTE_OFF=45 NOTE_ON=50 NOTE_ON=57 NOTE_ON=62 TIME_DELTA=6 NOTE_OFF=50 NOTE_OFF=57 NOTE_OFF=62 NOTE_ON=43 TIME_DELTA=6 NOTE_OFF=43 NOTE_ON=59 NOTE_ON=66 TIME_DELTA=6 NOTE_OFF=59 NOTE_OFF=66 NOTE_ON=50 TIME_DELTA=6 NOTE_OFF=50 BAR_END BAR_START NOTE_ON=52 TIME_DELTA=6 NOTE_OFF=52 NOTE_ON=61 NOTE_ON=67 TIME_DELTA=6 NOTE_OFF=61 NOTE_OFF=67 NOTE_ON=50 TIME_DELTA=6 NOTE_OFF=50 NOTE_ON=57 NOTE_ON=64 TIME_DELTA=6 NOTE_OFF=57 NOTE_OFF=64 NOTE_ON=45 TIME_DELTA=6 NOTE_OFF=45 NOTE_ON=50 NOTE_ON=57 NOTE_ON=62 TIME_DELTA=6 NOTE_OFF=50 NOTE_OFF=57 NOTE_OFF=62 NOTE_ON=55 NOTE_ON=59 NOTE_ON=64 TIME_DELTA=6 NOTE_OFF=55 NOTE_OFF=59 NOTE_OFF=64 NOTE_ON=50 TIME_DELTA=6 NOTE_OFF=50 NOTE_ON=62 NOTE_ON=64 TIME_DELTA=6 NOTE_OFF=62 NOTE_OFF=64 BAR_END BAR_START NOTE_ON=43 TIME_DELTA=6 NOTE_OFF=43 NOTE_ON=57 NOTE_ON=61 TIME_DELTA=6 NOTE_OFF=57 NOTE_OFF=61 NOTE_ON=55 NOTE_ON=58 NOTE_ON=61 TIME_DELTA=6 NOTE_OFF=55 NOTE_OFF=58 NOTE_OFF=61 NOTE_ON=41 TIME_DELTA=6 NOTE_OFF=41 NOTE_ON=53 NOTE_ON=60 TIME_DELTA=6 NOTE_OFF=53 NOTE_OFF=60 NOTE_ON=56 NOTE_ON=59 NOTE_ON=65 TIME_DELTA=6 NOTE_OFF=56 NOTE_OFF=59 NOTE_OFF=65 NOTE_ON=43 TIME_DELTA=6 NOTE_OFF=43 NOTE_ON=55 NOTE_ON=58 NOTE_ON=61 TIME_DELTA=6 NOTE_OFF=55 NOTE_OFF=58 NOTE_OFF=61 BAR_END TRACK_END"

#all composers
tokens = "PIECE_START TRACK_START INST=0 BAR_START NOTE_ON=83 TIME_DELTA=3 NOTE_OFF=83 NOTE_ON=85 TIME_DELTA=3 NOTE_OFF=85 NOTE_ON=86 TIME_DELTA=3 NOTE_OFF=86 NOTE_ON=85 TIME_DELTA=3 NOTE_OFF=85 NOTE_ON=83 TIME_DELTA=3 NOTE_OFF=83 NOTE_ON=81 TIME_DELTA=3 NOTE_OFF=81 NOTE_ON=80 TIME_DELTA=3 NOTE_OFF=80 NOTE_ON=78 TIME_DELTA=3 NOTE_OFF=78 NOTE_ON=76 TIME_DELTA=12 NOTE_OFF=76 TIME_DELTA=12 BAR_END BAR_START NOTE_ON=63 TIME_DELTA=6 NOTE_OFF=63 TIME_DELTA=6 NOTE_ON=62 TIME_DELTA=6 NOTE_OFF=62 TIME_DELTA=6 NOTE_ON=65 TIME_DELTA=6 NOTE_OFF=65 NOTE_ON=63 TIME_DELTA=6 NOTE_OFF=63 TIME_DELTA=6 NOTE_ON=71 TIME_DELTA=6 NOTE_OFF=71 BAR_END BAR_START TIME_DELTA=6 NOTE_ON=71 TIME_DELTA=6 NOTE_OFF=71 TIME_DELTA=6 NOTE_ON=62 TIME_DELTA=6 NOTE_OFF=62 TIME_DELTA=6 NOTE_ON=61 TIME_DELTA=6 NOTE_OFF=61 TIME_DELTA=6 NOTE_ON=65 TIME_DELTA=6 NOTE_OFF=65 NOTE_ON=61 TIME_DELTA=6 NOTE_OFF=61 BAR_END BAR_START TIME_DELTA=6 NOTE_ON=61 TIME_DELTA=6 NOTE_OFF=61 TIME_DELTA=6 NOTE_ON=49 TIME_DELTA=6 NOTE_OFF=49 TIME_DELTA=6 NOTE_ON=37 TIME_DELTA=6 NOTE_OFF=37 TIME_DELTA=6 NOTE_ON=56 TIME_DELTA=6 NOTE_OFF=56 TIME_DELTA=6 BAR_END TRACK_END TRACK_START INST=0 BAR_START TIME_DELTA=18 NOTE_ON=71 TIME_DELTA=6 NOTE_OFF=71 NOTE_ON=71 TIME_DELTA=6 NOTE_OFF=71 NOTE_ON=73 TIME_DELTA=6 NOTE_OFF=73 NOTE_ON=71 TIME_DELTA=6 NOTE_OFF=71 BAR_END BAR_START TIME_DELTA=6 NOTE_ON=71 TIME_DELTA=6 NOTE_OFF=71 NOTE_ON=71 TIME_DELTA=6 NOTE_OFF=71 NOTE_ON=69 TIME_DELTA=6 NOTE_OFF=69 NOTE_ON=71 TIME_DELTA=6 NOTE_OFF=71 NOTE_ON=68 TIME_DELTA=6 NOTE_OFF=68 NOTE_ON=61 TIME_DELTA=6 NOTE_OFF=61 NOTE_ON=59 TIME_DELTA=6 NOTE_OFF=59 BAR_END BAR_START TIME_DELTA=6 NOTE_ON=58 TIME_DELTA=6 NOTE_OFF=58 NOTE_ON=59 TIME_DELTA=6 NOTE_OFF=59 NOTE_ON=49 TIME_DELTA=6 NOTE_OFF=49 NOTE_ON=37 TIME_DELTA=6 NOTE_OFF=37 TIME_DELTA=6 NOTE_ON=53 TIME_DELTA=6 NOTE_OFF=53 NOTE_ON=56 TIME_DELTA=6 NOTE_OFF=56 BAR_END BAR_START NOTE_ON=49 TIME_DELTA=6 NOTE_OFF=49 NOTE_ON=37 TIME_DELTA=6 NOTE_OFF=37 TIME_DELTA=6 NOTE_ON=53 TIME_DELTA=6 NOTE_OFF=53 NOTE_ON=56 TIME_DELTA=6 NOTE_OFF=56 NOTE_ON=49 TIME_DELTA=6 NOTE_OFF=49 NOTE_ON=38 TIME_DELTA=6 NOTE_OFF=38 TIME_DELTA=6 BAR_END TRACK_END"

input_ids_x_tilde = torch.tensor([tokenizer.encode(tokens).ids])
attention_x_tilde = torch.tensor([tokenizer.encode(tokens).attention_mask])
#composer = 'chopin'

results = []
resultcomposer = []

#resultcomposer.append()
#choose random piece
#choose random composer
# composer = random.choice(composers)
# print(composer)

# x_tilde = random.choice(tokenized_datasets['generated_' + composer])

# #print(x_tilde_index)

# input_ids_x_tilde = torch.tensor([x_tilde['input_ids']])
# attention_x_tilde = torch.tensor([x_tilde['attention_mask']])



feature_vec_x_tilde = f_tilde(input_ids_x_tilde, attention_x_tilde)


#x_tilde2 = random.choice(tokenized_datasets['input_' + composer])

#results.append(x_tilde2['text'])

#print(x_tilde_index)

#input_ids_x_tilde2 = torch.tensor([x_tilde2['input_ids']])
#attention_x_tilde2 = torch.tensor([x_tilde2['attention_mask']])
#print(input_ids_x_tilde)


#feature_vec_x_tilde2 = f(input_ids_x_tilde2, attention_x_tilde2)

#ground_truth = np.dot(feature_vec_x_tilde2[0].detach().numpy(), feature_vec_x_tilde[0].detach().numpy())
#calculate similarity scores
#print("ground_truth: ", ground_truth)
#list of similarit scores, => s = dot(F(x), F_tilde(x_tilde))
s = []

for c in composers:
   # if c == composer:
    #    continue
    for i in range(20):
      x = random.choice(tokenized_datasets['input_' + c])
      results.append(x['text'])
      resultcomposer.append(c)
      input_ids_x = torch.tensor([x['input_ids']])
      attention_x = torch.tensor([x['attention_mask']])
      feature_vec_x = f(input_ids_x, attention_x)

      similarity_score = np.dot(feature_vec_x[0].detach().numpy(), feature_vec_x_tilde[0].detach().numpy())
          
      s.append(similarity_score)
   


model = ProbabilityScore()

scores = model(torch.tensor(s))

for i in range(len(scores)):
   if(scores[i] > 0):
      print("Score: ", scores[i])
      print(resultcomposer[i])
      print(results[i])


print(s)
print(sum(scores))
print(scores * 100)