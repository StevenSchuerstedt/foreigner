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



# from transformers import AutoTokenizer, BertModel
# import torch

# tokenizer = gpt2_composer.load_tokenizer("")
# model = BertModel.from_pretrained("checkpoint_bert/checkpoint-22500")
# #tokenizer.enable_padding(length=512, pad_id=model.config.pad_token_id)
# primer = "PIECE_START TRACK_START INST=0 BAR_START NOTE_ON=44 NOTE_ON=44 NOTE_ON=47 NOTE_ON=50"

# input_ids = torch.tensor([tokenizer.encode(primer).ids])

# print(input_ids)

# generated_ids = model(input_ids)


# print(generated_ids.last_hidden_state[0][-1])
# print(generated_ids.last_hidden_state.shape)

#last_hidden_states = outputs.last_hidden_state

#print(outputs[1])

# data_files = {"input": "DATA/data.json"}
# dataset = datasets.load_dataset("json", data_files=data_files)

# print(dataset['input']['bach'][1])


# DATASET_PATH = os.path.join(dirname, 'DATA\\chopin')
# TOKENS_PATH = os.path.join(dirname, 'DATA\\TOKENS')


# # load dataset
# data_files = {

#     "input_bach": "DATA/attribution/input/input_bach.txt",
#     "input_beethoven": "DATA/attribution/input/input_beethoven.txt",
#     "input_chopin": "DATA/attribution/input/input_chopin.txt",
#     "input_grieg": "DATA/attribution/input/input_grieg.txt",
#     "input_haydn": "DATA/attribution/input/input_haydn.txt",
#     "input_liszt": "DATA/attribution/input/input_liszt.txt",
#     "input_mendelssohn": "DATA/attribution/input/input_mendelssohn.txt",
#     "input_rachmaninov": "DATA/attribution/input/input_rachmaninov.txt",

#           }

# dataset = datasets.load_dataset("text", data_files=data_files)
# tokenizer = gpt2_composer.load_tokenizer("")

# def tokenize_function(examples):
#     outputs = tokenizer.encode_batch(examples["text"])
#     example = {
#         "input_ids": [c.ids for c in outputs]
#     }

#     return example


# tokenized_datasets = dataset.map(
#     tokenize_function, batched=True, remove_columns=["text"])


# index = 0

# #print(len(tokenized_datasets['input_bach'][1]['input_ids']))

# for e in tokenized_datasets['input_rachmaninov']:
#     index += len(e['input_ids'])
# print(index)


# tokenizer = gpt2_composer.load_tokenizer("")


# f = AttributionHead("checkpoints/checkpoint-22500")
# f_tilde = AttributionHead("checkpoints/checkpoint-22500")


# tokenizer.enable_padding(length=512, pad_id=f.transformer.config.pad_token_id)
# print(f.transformer.config.pad_token_id)

# #f.load("checkpoint_attribute/f", "checkpoint_attribute/transformer_f")
# #f_tilde.load("checkpoint_attribute/f_tilde", "checkpoint_attribute/transformer_f_tilde")

# # batched_ids = [
# #     [200, 200, 200],
# #     [200, 200]
# # ]

# # torch.tensor(batched_ids)


# x = "PIECE_START TRACK_START INST=0 BAR_START NOTE_ON=77 NOTE_ON=79 TIME_DELTA=12 NOTE_OFF=76"
# y = "PIECE_START TRACK_START INST=0 BAR_START NOTE_ON=77 NOTE_ON=79 TIME_DELTA=12 NOTE_OFF=76 NOTE_ON=77 INST=0 BAR_START NOTE_ON=77 NOTE_ON=79 TIME_DELTA=12 NOTE_OFF=76 NOTE_ON=77"
# z = "PIECE_START TRACK_START INST=0 BAR_START NOTE_ON=77 NOTE_ON=79 TIME_DELTA=12 NOTE_OFF=76 TIME_DELTA=12 TIME_DELTA=12 TIME_DELTA=12"

# # print(tokenizer.encode(x).ids)

# # print(tokenizer.encode(x).ids)

# input_ids_x = torch.tensor([tokenizer.encode(x).ids,tokenizer.encode(y).ids, tokenizer.encode(z).ids ])

# #input_ids_x = torch.tensor([tokenizer.encode(z).ids])

# attention_x = torch.tensor([tokenizer.encode(x).attention_mask, tokenizer.encode(y).attention_mask, tokenizer.encode(z).attention_mask])

# #attention_x = torch.tensor([tokenizer.encode(z).attention_mask])

# print(x)
# print(input_ids_x)
# #print(tokenizer.encode(x).attention_mask)

# feature_vec_x = f(input_ids_x, attention_x)

# print(feature_vec_x)
# print(feature_vec_x.shape)



#TODO:
# - setup new training with train / test / validation 
# - fix generate scores, softplus to relu -> just use relu when training? similarity scores not always correct so training cannot work?
# - try out bart for generation
# - finish draft of main chapter in thesis -> jep 
# - evaluate current attribute classifier on some data (beethoven, chopin leftover etc) generate new stuff and test

#TODO: (14.02.2024)
# - check similarity scores, should work but doesnt? => basis for probability calibration
# - write new chapter for gpt2 for music in thesis (lets take time, because correction is too much work...will take long anyways)
# - should retrain everything with pad token id?? dunno...
# - ..TODO
# - normalizing flows, model neu trainineren, auf einzelne composer condition kp


#- Precision, False Negatives, True Positives etc, 4 Werte Berechnen
# - confusion matrix

# - future work kapitel (?) conditional flow matching

#- ganzes stück von beethoven zusammen betrachten mit wahrscheinlichkeiten

#- näher am paper schreiben

#- noch mehr erklären, verständis der materie, NTXENT Loss, vergleich zu anderen loss, andere paper lesen
#links, rechts 

#kapitel contrastive loss 

#vllt bessere Notation? statt x tilde, x+ 

#gleichungen erklären

#- vergleich wang paper bilder, zu transformer unterschied ausarbeiten

#2methoden, resampling, loss ändern



# true pos:  {'bach': 99, 'beethoven': 63, 'chopin': 42, 'grieg': 82, 'haydn': 88, 'liszt': 82, 'mendelssohn': 85, 'rachmaninov': 79}
# false neg:  {'bach': 1, 'beethoven': 37, 'chopin': 58, 'grieg': 18, 'haydn': 12, 'liszt': 18, 'mendelssohn': 15, 'rachmaninov': 21}


#print( (99 + 63 + 42 + 82 + 88 + 82 + 85 + 79) / (99 + 63 + 42 + 82 + 88 + 82 + 85 + 79 + 1 +37 + 58 + 18 + 12 + 18 + 15 + 21))


data_count = {
    'bach': 0,
    'beethoven': 0,
    'chopin': 0,
    'grieg': 0,
    'haydn': 0,
    'liszt': 0,
    'mendelssohn': 0,
    'rachmaninov': 0,
}

data_list = {
    'bach': {
        'bach': 0,
        'beethoven': 0,
        'chopin': 0,
        'grieg': 0,
        'haydn': 0,
        'liszt': 0,
        'mendelssohn': 0,
        'rachmaninov': 0,
    },
    'beethoven': {
        'bach': 0,
        'beethoven': 0,
        'chopin': 0,
        'grieg': 0,
        'haydn': 0,
        'liszt': 0,
        'mendelssohn': 0,
        'rachmaninov': 0,
    },
    'chopin': {
        'bach': 0,
        'beethoven': 0,
        'chopin': 0,
        'grieg': 0,
        'haydn': 0,
        'liszt': 0,
        'mendelssohn': 0,
        'rachmaninov': 0,
    },
    'grieg': {
        'bach': 0,
        'beethoven': 0,
        'chopin': 0,
        'grieg': 0,
        'haydn': 0,
        'liszt': 0,
        'mendelssohn': 0,
        'rachmaninov': 0,
    },
    'haydn': {
        'bach': 0,
        'beethoven': 0,
        'chopin': 0,
        'grieg': 0,
        'haydn': 0,
        'liszt': 0,
        'mendelssohn': 0,
        'rachmaninov': 0,
    },
    'liszt': {
        'bach': 0,
        'beethoven': 0,
        'chopin': 0,
        'grieg': 0,
        'haydn': 0,
        'liszt': 0,
        'mendelssohn': 0,
        'rachmaninov': 0,
    },
    'mendelssohn': {
        'bach': 0,
        'beethoven': 0,
        'chopin': 0,
        'grieg': 0,
        'haydn': 0,
        'liszt': 0,
        'mendelssohn': 0,
        'rachmaninov': 0,
    },
    'rachmaninov': {
        'bach': 0,
        'beethoven': 0,
        'chopin': 0,
        'grieg': 0,
        'haydn': 0,
        'liszt': 0,
        'mendelssohn': 0,
        'rachmaninov': 0,
    },
}

data_list['bach']['bach'] += 1

data_list['rachmaninov']['liszt'] += 1
print(data_list)