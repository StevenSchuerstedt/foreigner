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
from AttributionHead import AttributionHead

dirname = os.path.dirname(__file__)

import gpt2_composer

composers = ['bach', 'beethoven', 'chopin', 'grieg', 'haydn', 'liszt', 'mendelssohn', 'rachmaninov']

# data_files = {"input": "DATA/data.json"}
# dataset = datasets.load_dataset("json", data_files=data_files)

# print(dataset['input']['bach'][1])


DATASET_PATH = os.path.join(dirname, 'DATA\\chopin')
TOKENS_PATH = os.path.join(dirname, 'DATA\\TOKENS')





tokenizer = gpt2_composer.load_tokenizer("")


f = AttributionHead("checkpoints/checkpoint-22500")
f_tilde = AttributionHead("checkpoints/checkpoint-22500")


tokenizer.enable_padding(length=512, pad_id=f.transformer.config.pad_token_id)
print(f.transformer.config.pad_token_id)

#f.load("checkpoint_attribute/f", "checkpoint_attribute/transformer_f")
#f_tilde.load("checkpoint_attribute/f_tilde", "checkpoint_attribute/transformer_f_tilde")

# batched_ids = [
#     [200, 200, 200],
#     [200, 200]
# ]

# torch.tensor(batched_ids)


x = "PIECE_START TRACK_START INST=0 BAR_START NOTE_ON=77 NOTE_ON=79 TIME_DELTA=12 NOTE_OFF=76"
y = "PIECE_START TRACK_START INST=0 BAR_START NOTE_ON=77 NOTE_ON=79 TIME_DELTA=12 NOTE_OFF=76 NOTE_ON=77 INST=0 BAR_START NOTE_ON=77 NOTE_ON=79 TIME_DELTA=12 NOTE_OFF=76 NOTE_ON=77"
z = "PIECE_START TRACK_START INST=0 BAR_START NOTE_ON=77 NOTE_ON=79 TIME_DELTA=12 NOTE_OFF=76 TIME_DELTA=12 TIME_DELTA=12 TIME_DELTA=12"

# print(tokenizer.encode(x).ids)

# print(tokenizer.encode(x).ids)

input_ids_x = torch.tensor([tokenizer.encode(x).ids,tokenizer.encode(y).ids, tokenizer.encode(z).ids ])

#input_ids_x = torch.tensor([tokenizer.encode(z).ids])

attention_x = torch.tensor([tokenizer.encode(x).attention_mask, tokenizer.encode(y).attention_mask, tokenizer.encode(z).attention_mask])

#attention_x = torch.tensor([tokenizer.encode(z).attention_mask])

print(x)
print(input_ids_x)
#print(tokenizer.encode(x).attention_mask)

feature_vec_x = f(input_ids_x, attention_x)

print(feature_vec_x)
print(feature_vec_x.shape)