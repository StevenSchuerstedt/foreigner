import numpy as np
import note_seq
import transformers
import datasets
import torch
from typing import Optional
from AttributionHeadBert import AttributionHeadBert

import gpt2_composer

from AttributionHead import AttributionHead


tokenizer = gpt2_composer.load_tokenizer("")



f = AttributionHead("checkpoints/checkpoint-22500")
f_tilde = AttributionHead("checkpoints/checkpoint-22500")
#TODO: PADDING IMPOORTANT
tokenizer.enable_padding(length=512, pad_id=f.transformer.config.pad_token_id)

#f.load("checkpoint_attribute/checkpoint_attribute_f", "checkpoint_attribute/transformer_f")
#f_tilde.load("checkpoint_attribute/checkpoint_attribute_f_tilde", "checkpoint_attribute/transformer_f_tilde")

#28
#46

# load dataset
data_files = {"generated": "DATA/attribution_generated_old.txt", "input": "DATA/attribution_input_old.txt"}
dataset = datasets.load_dataset("text", data_files=data_files)

grand_score = 0
for i in range(8):
  score = 0
  #get x_tilde
  x_tilde = dataset['generated'][i]['text']
  x = dataset['input'][i]['text']

  input_ids_x_tilde = torch.tensor([tokenizer.encode(x_tilde).ids])
  attention_x_tilde = torch.tensor([tokenizer.encode(x_tilde).attention_mask])


  input_ids_x = torch.tensor([tokenizer.encode(x).ids])
  attention_x = torch.tensor([tokenizer.encode(x).attention_mask])

  feature_vec_x = f(input_ids_x, attention_x)
  feature_vec_x_tilde = f_tilde(input_ids_x_tilde, attention_x_tilde)

  output_x_xtilde = np.dot(feature_vec_x[0].detach().numpy(), feature_vec_x_tilde[0].detach().numpy())

  print(i)
  print("Ground Truth: ", output_x_xtilde)
  for j in range(8):
    if i == j: 
      continue
    y = dataset['input'][j]['text']

    input_ids_y = torch.tensor([tokenizer.encode(y).ids])
    attention_y = torch.tensor([tokenizer.encode(y).attention_mask])
    feature_vec_y = f(input_ids_y, attention_y)
    output_y_xtilde = np.dot(feature_vec_y[0].detach().numpy(), feature_vec_x_tilde[0].detach().numpy())
    print("Score for: ", j)
    print(output_y_xtilde)
    if output_x_xtilde > output_y_xtilde:
      score = score + 1
  print("TOTAL SCORE: ", score)
  grand_score = grand_score + score


print("GRAND SCORE: ", grand_score)


# #generated piece with finetuned bach
# x_tilde = "PIECE_START TRACK_START INST=0 BAR_START NOTE_ON=71 TIME_DELTA=12 NOTE_OFF=71 NOTE_ON=73 TIME_DELTA=12 NOTE_OFF=73 NOTE_ON=75 TIME_DELTA=12 NOTE_OFF=75 NOTE_ON=76 TIME_DELTA=12 NOTE_OFF=76 BAR_END BAR_START NOTE_ON=75 TIME_DELTA=12 NOTE_OFF=75 NOTE_ON=73 TIME_DELTA=12 NOTE_OFF=73 NOTE_ON=73 TIME_DELTA=24 NOTE_OFF=73 BAR_END BAR_START NOTE_ON=71 TIME_DELTA=12 NOTE_OFF=71 NOTE_ON=71 TIME_DELTA=12 NOTE_OFF=71 NOTE_ON=70 TIME_DELTA=12 NOTE_OFF=70 NOTE_ON=71 TIME_DELTA=12 NOTE_OFF=71 BAR_END BAR_START NOTE_ON=73 TIME_DELTA=12 NOTE_OFF=73 NOTE_ON=73 TIME_DELTA=12 NOTE_OFF=73 NOTE_ON=71 TIME_DELTA=24 NOTE_OFF=71 BAR_END TRACK_END TRACK_START INST=0 BAR_START NOTE_ON=66 TIME_DELTA=12 NOTE_OFF=66 NOTE_ON=66 TIME_DELTA=12 NOTE_OFF=66 NOTE_ON=66 TIME_DELTA=12 NOTE_OFF=66 NOTE_ON=68 TIME_DELTA=12 NOTE_OFF=68 BAR_END BAR_START NOTE_ON=66 TIME_DELTA=12 NOTE_OFF=66 NOTE_ON=66 TIME_DELTA=12 NOTE_OFF=66 NOTE_ON=65 TIME_DELTA=24 NOTE_OFF=65 BAR_END BAR_START NOTE_ON=66 TIME_DELTA=18 NOTE_OFF=66 NOTE_ON=64 TIME_DELTA=6 NOTE_OFF=64 NOTE_ON=64 TIME_DELTA=6 NOTE_OFF=64 NOTE_ON=66 TIME_DELTA=12 NOTE_OFF=66 NOTE_ON=66 TIME_DELTA=6 NOTE_OFF=66 NOTE_ON=66 TIME_DELTA=6 NOTE_OFF=66 BAR_END BAR_START NOTE_ON=64 TIME_DELTA=12 NOTE_OFF=64 NOTE_ON=66 TIME_DELTA=6 NOTE_OFF=66 NOTE_ON=68 TIME_DELTA=6 NOTE_OFF=68 NOTE_ON=69 TIME_DELTA=6 NOTE_OFF=69 NOTE_ON=68 TIME_DELTA=24 NOTE_OFF=68 BAR_END TRACK_END TRACK_START INST=0 BAR_START NOTE_ON=63 TIME_DELTA=12 NOTE_OFF=63 NOTE_ON=61 TIME_DELTA=12 NOTE_OFF=61 NOTE_ON=59 TIME_DELTA=12 NOTE_OFF=59 NOTE_ON=61 TIME_DELTA=12 NOTE_OFF=61 BAR_END BAR_START NOTE_ON=61 TIME_DELTA=12 NOTE_OFF=61 NOTE_ON=59 TIME_DELTA=6 NOTE_OFF=59 NOTE_ON=58 TIME_DELTA=6 NOTE_OFF=58 NOTE_ON=56 TIME_DELTA=24 NOTE_OFF=56 BAR_END BAR_START NOTE_ON=62 TIME_DELTA=12 NOTE_OFF=62 NOTE_ON=61 TIME_DELTA=12 NOTE_OFF=61 NOTE_ON=61 TIME_DELTA=12 NOTE_OFF=61 NOTE_ON=62 TIME_DELTA=12 NOTE_OFF=62 BAR_END BAR_START NOTE_ON=61 TIME_DELTA=12 NOTE_OFF=61 NOTE_ON=61 TIME_DELTA=12 NOTE_OFF=61 NOTE_ON=62 TIME_DELTA=24 NOTE_OFF=62 BAR_END TRACK_END TRACK_START INST=0 BAR_START NOTE_ON=59 TIME_DELTA=12 NOTE_OFF=59 NOTE_ON=58 TIME_DELTA=12 NOTE_OFF=58 NOTE_ON=59 TIME_DELTA=12 NOTE_OFF=59 NOTE_ON=49 TIME_DELTA=12 NOTE_OFF=49 BAR_END BAR_START NOTE_ON=54 TIME_DELTA=12 NOTE_OFF=54 NOTE_ON=54 TIME_DELTA=12 NOTE_OFF=54 NOTE_ON=49 TIME_DELTA=24 NOTE_OFF=49 BAR_END BAR_START NOTE_ON=59 TIME_DELTA=6 NOTE_OFF=59 NOTE_ON=61 TIME_DELTA=6 NOTE_OFF=61 NOTE_ON=62 TIME_DELTA=6 NOTE_OFF=62 NOTE_ON=61 TIME_DELTA=6 NOTE_OFF=61 NOTE_ON=58 TIME_DELTA=6 NOTE_OFF=58 NOTE_ON=52 TIME_DELTA=6 NOTE_OFF=52 NOTE_ON=50 TIME_DELTA=12 NOTE_OFF=50 BAR_END BAR_START NOTE_ON=58 TIME_DELTA=12 NOTE_OFF=58 NOTE_ON=54 TIME_DELTA=6 NOTE_OFF=54 NOTE_ON=56 TIME_DELTA=6 NOTE_OFF=56 NOTE_ON=58 TIME_DELTA=24 NOTE_OFF=58 BAR_END TRACK_END"


# #data of bach
# x = "PIECE_START TRACK_START INST=0 BAR_START NOTE_ON=67 TIME_DELTA=12 NOTE_OFF=67 NOTE_ON=65 TIME_DELTA=12 NOTE_OFF=65 NOTE_ON=67 TIME_DELTA=12 NOTE_OFF=67 NOTE_ON=69 TIME_DELTA=12 NOTE_OFF=69 BAR_END BAR_START NOTE_ON=69 TIME_DELTA=12 NOTE_OFF=69 NOTE_ON=70 TIME_DELTA=12 NOTE_OFF=70 NOTE_ON=72 TIME_DELTA=12 NOTE_OFF=72 NOTE_ON=70 TIME_DELTA=6 NOTE_OFF=70 NOTE_ON=69 TIME_DELTA=6 NOTE_OFF=69 BAR_END BAR_START NOTE_ON=67 TIME_DELTA=6 NOTE_OFF=67 NOTE_ON=69 TIME_DELTA=6 NOTE_OFF=69 NOTE_ON=70 TIME_DELTA=18 NOTE_OFF=70 NOTE_ON=72 TIME_DELTA=6 NOTE_OFF=72 NOTE_ON=69 TIME_DELTA=12 NOTE_OFF=69 BAR_END BAR_START NOTE_ON=74 TIME_DELTA=12 NOTE_OFF=74 NOTE_ON=75 TIME_DELTA=6 NOTE_OFF=75 NOTE_ON=74 TIME_DELTA=6 NOTE_OFF=74 NOTE_ON=72 TIME_DELTA=12 NOTE_OFF=72 NOTE_ON=74 TIME_DELTA=6 NOTE_OFF=74 NOTE_ON=72 TIME_DELTA=6 NOTE_OFF=72 BAR_END TRACK_END TRACK_START INST=0 BAR_START NOTE_ON=63 TIME_DELTA=6 NOTE_OFF=63 NOTE_ON=62 TIME_DELTA=6 NOTE_OFF=62 NOTE_ON=60 TIME_DELTA=6 NOTE_OFF=60 NOTE_ON=62 TIME_DELTA=6 NOTE_OFF=62 NOTE_ON=62 TIME_DELTA=6 NOTE_OFF=62 NOTE_ON=64 TIME_DELTA=6 NOTE_OFF=64 NOTE_ON=65 TIME_DELTA=12 NOTE_OFF=65 BAR_END BAR_START NOTE_ON=65 TIME_DELTA=12 NOTE_OFF=65 NOTE_ON=65 TIME_DELTA=12 NOTE_OFF=65 NOTE_ON=65 TIME_DELTA=12 NOTE_OFF=65 NOTE_ON=65 TIME_DELTA=12 NOTE_OFF=65 BAR_END BAR_START NOTE_ON=63 TIME_DELTA=12 NOTE_OFF=63 NOTE_ON=62 TIME_DELTA=18 NOTE_OFF=62 NOTE_ON=64 TIME_DELTA=6 NOTE_OFF=64 NOTE_ON=66 TIME_DELTA=12 NOTE_OFF=66 BAR_END BAR_START NOTE_ON=67 TIME_DELTA=6 NOTE_OFF=67 NOTE_ON=65 TIME_DELTA=6 NOTE_OFF=65 NOTE_ON=63 TIME_DELTA=12 NOTE_OFF=63 NOTE_ON=65 TIME_DELTA=12 NOTE_OFF=65 NOTE_ON=65 TIME_DELTA=6 NOTE_OFF=65 NOTE_ON=63 TIME_DELTA=6 NOTE_OFF=63 BAR_END TRACK_END TRACK_START INST=0 BAR_START NOTE_ON=58 TIME_DELTA=12 NOTE_OFF=58 NOTE_ON=57 TIME_DELTA=6 NOTE_OFF=57 NOTE_ON=58 TIME_DELTA=3 NOTE_OFF=58 NOTE_ON=57 TIME_DELTA=3 NOTE_OFF=57 NOTE_ON=58 TIME_DELTA=6 NOTE_OFF=58 NOTE_ON=60 TIME_DELTA=6 NOTE_OFF=60 NOTE_ON=60 TIME_DELTA=12 NOTE_OFF=60 BAR_END BAR_START NOTE_ON=60 TIME_DELTA=12 NOTE_OFF=60 NOTE_ON=62 TIME_DELTA=12 NOTE_OFF=62 NOTE_ON=63 TIME_DELTA=12 NOTE_OFF=63 NOTE_ON=62 TIME_DELTA=12 NOTE_OFF=62 BAR_END BAR_START NOTE_ON=55 TIME_DELTA=6 NOTE_OFF=55 NOTE_ON=54 TIME_DELTA=6 NOTE_OFF=54 NOTE_ON=55 TIME_DELTA=6 NOTE_OFF=55 NOTE_ON=50 TIME_DELTA=6 NOTE_OFF=50 NOTE_ON=55 TIME_DELTA=12 NOTE_OFF=55 NOTE_ON=57 TIME_DELTA=12 NOTE_OFF=57 BAR_END BAR_START NOTE_ON=55 TIME_DELTA=12 NOTE_OFF=55 NOTE_ON=60 TIME_DELTA=6 NOTE_OFF=60 NOTE_ON=58 TIME_DELTA=6 NOTE_OFF=58 NOTE_ON=57 TIME_DELTA=12 NOTE_OFF=57 NOTE_ON=57 TIME_DELTA=12 NOTE_OFF=57 BAR_END TRACK_END TRACK_START INST=0 BAR_START NOTE_ON=51 TIME_DELTA=18 NOTE_OFF=51 NOTE_ON=50 TIME_DELTA=3 NOTE_OFF=50 NOTE_ON=48 TIME_DELTA=3 NOTE_OFF=48 NOTE_ON=46 TIME_DELTA=6 NOTE_OFF=46 NOTE_ON=48 TIME_DELTA=6 NOTE_OFF=48 NOTE_ON=41 TIME_DELTA=12 NOTE_OFF=41 BAR_END BAR_START NOTE_ON=53 TIME_DELTA=6 NOTE_OFF=53 NOTE_ON=51 TIME_DELTA=6 NOTE_OFF=51 NOTE_ON=50 TIME_DELTA=6 NOTE_OFF=50 NOTE_ON=48 TIME_DELTA=6 NOTE_OFF=48 NOTE_ON=46 TIME_DELTA=6 NOTE_OFF=46 NOTE_ON=45 TIME_DELTA=6 NOTE_OFF=45 NOTE_ON=46 TIME_DELTA=12 NOTE_OFF=46 BAR_END BAR_START NOTE_ON=48 TIME_DELTA=12 NOTE_OFF=48 NOTE_ON=43 TIME_DELTA=6 NOTE_OFF=43 NOTE_ON=45 TIME_DELTA=6 NOTE_OFF=45 NOTE_ON=46 TIME_DELTA=6 NOTE_OFF=46 NOTE_ON=43 TIME_DELTA=6 NOTE_OFF=43 NOTE_ON=50 TIME_DELTA=12 NOTE_OFF=50 BAR_END BAR_START NOTE_ON=47 TIME_DELTA=12 NOTE_OFF=47 NOTE_ON=48 TIME_DELTA=12 NOTE_OFF=48 NOTE_ON=53 TIME_DELTA=6 NOTE_OFF=53 NOTE_ON=51 TIME_DELTA=6 NOTE_OFF=51 NOTE_ON=50 TIME_DELTA=12 NOTE_OFF=50 BAR_END TRACK_END"


# #no data of bach
# y = "PIECE_START TRACK_START INST=0 BAR_START NOTE_ON=59 TIME_DELTA=3 NOTE_OFF=59 NOTE_ON=71 NOTE_ON=63 TIME_DELTA=3 NOTE_OFF=71 NOTE_OFF=63 NOTE_ON=69 NOTE_ON=57 TIME_DELTA=3 NOTE_OFF=69 NOTE_OFF=57 NOTE_ON=68 NOTE_ON=64 TIME_DELTA=3 NOTE_OFF=68 NOTE_OFF=64 NOTE_ON=63 NOTE_ON=56 TIME_DELTA=3 NOTE_OFF=63 NOTE_OFF=56 NOTE_ON=64 NOTE_ON=59 TIME_DELTA=3 NOTE_OFF=64 NOTE_OFF=59 NOTE_ON=66 TIME_DELTA=12 NOTE_OFF=66 NOTE_ON=66 TIME_DELTA=3 NOTE_OFF=66 NOTE_ON=68 TIME_DELTA=3 NOTE_OFF=68 NOTE_ON=68 TIME_DELTA=3 NOTE_OFF=68 NOTE_ON=66 TIME_DELTA=3 NOTE_OFF=66 NOTE_ON=64 TIME_DELTA=6 NOTE_OFF=64 BAR_END BAR_START TIME_DELTA=6 NOTE_ON=71 NOTE_ON=62 NOTE_ON=66 TIME_DELTA=3 NOTE_OFF=71 NOTE_OFF=62 NOTE_OFF=66 NOTE_ON=73 NOTE_ON=64 NOTE_ON=68 TIME_DELTA=3 NOTE_OFF=73 NOTE_OFF=64 NOTE_OFF=68 NOTE_ON=73 NOTE_ON=62 NOTE_ON=66 TIME_DELTA=3 NOTE_OFF=73 NOTE_OFF=62 NOTE_OFF=66 NOTE_ON=71 NOTE_ON=64 NOTE_ON=68 TIME_DELTA=3 NOTE_OFF=71 NOTE_OFF=64 NOTE_OFF=68 NOTE_ON=69 NOTE_ON=61 TIME_DELTA=3 NOTE_OFF=69 NOTE_OFF=61 NOTE_ON=71 NOTE_ON=64 TIME_DELTA=3 NOTE_OFF=71 NOTE_OFF=64 NOTE_ON=68 NOTE_ON=61 TIME_DELTA=3 NOTE_OFF=68 NOTE_OFF=61 NOTE_ON=69 NOTE_ON=64 TIME_DELTA=3 NOTE_OFF=69 NOTE_OFF=64 NOTE_ON=75 NOTE_ON=66 NOTE_ON=72 TIME_DELTA=3 NOTE_OFF=75 NOTE_OFF=66 NOTE_OFF=72 NOTE_ON=76 NOTE_ON=68 TIME_DELTA=3 NOTE_OFF=76 NOTE_OFF=68 NOTE_ON=76 NOTE_ON=66 NOTE_ON=70 TIME_DELTA=3 NOTE_OFF=76 NOTE_OFF=66 NOTE_OFF=70 NOTE_ON=75 NOTE_ON=68 NOTE_ON=72 TIME_DELTA=3 NOTE_OFF=75 NOTE_OFF=68 NOTE_OFF=72 NOTE_ON=73 NOTE_ON=64 TIME_DELTA=3 NOTE_OFF=73 NOTE_OFF=64 NOTE_ON=75 NOTE_ON=68 TIME_DELTA=3 NOTE_OFF=75 NOTE_OFF=68 BAR_END BAR_START NOTE_ON=72 NOTE_ON=64 TIME_DELTA=3 NOTE_OFF=72 NOTE_OFF=64 NOTE_ON=73 NOTE_ON=68 TIME_DELTA=3 NOTE_OFF=73 NOTE_OFF=68 NOTE_ON=64 NOTE_ON=70 NOTE_ON=76 TIME_DELTA=3 NOTE_OFF=64 NOTE_OFF=70 NOTE_OFF=76 NOTE_ON=66 NOTE_ON=70 NOTE_ON=78 TIME_DELTA=3 NOTE_OFF=66 NOTE_OFF=70 NOTE_OFF=78 NOTE_ON=63 NOTE_ON=70 NOTE_ON=75 TIME_DELTA=3 NOTE_OFF=63 NOTE_OFF=70 NOTE_OFF=75 NOTE_ON=63 NOTE_ON=70 NOTE_ON=76 TIME_DELTA=3 NOTE_OFF=63 NOTE_OFF=70 NOTE_OFF=76 NOTE_ON=66 NOTE_ON=70 NOTE_ON=78 TIME_DELTA=3 NOTE_OFF=66 NOTE_OFF=70 NOTE_OFF=78 NOTE_ON=68 NOTE_ON=70 NOTE_ON=80 TIME_DELTA=3 NOTE_OFF=68 NOTE_OFF=70 NOTE_OFF=80 NOTE_ON=64 NOTE_ON=70 NOTE_ON=76 TIME_DELTA=3 NOTE_OFF=64 NOTE_OFF=70 NOTE_OFF=76 NOTE_ON=66 NOTE_ON=70 NOTE_ON=78 TIME_DELTA=3 NOTE_OFF=66 NOTE_OFF=70 NOTE_OFF=78 NOTE_ON=68 NOTE_ON=76 NOTE_ON=80 TIME_DELTA=3 NOTE_OFF=68 NOTE_ON=71 TIME_DELTA=3 NOTE_OFF=71 NOTE_ON=68 TIME_DELTA=3 NOTE_OFF=68 NOTE_ON=71 TIME_DELTA=3 NOTE_OFF=71 NOTE_ON=68 TIME_DELTA=3 NOTE_OFF=76 NOTE_OFF=80 NOTE_OFF=68 NOTE_ON=78 NOTE_ON=71 TIME_DELTA=3 NOTE_OFF=78 NOTE_OFF=71 BAR_END BAR_START NOTE_ON=76 NOTE_ON=68 TIME_DELTA=3 NOTE_OFF=76 NOTE_OFF=68 NOTE_ON=73 NOTE_ON=71 TIME_DELTA=3 NOTE_OFF=73 NOTE_OFF=71 NOTE_ON=63 NOTE_ON=75 TIME_DELTA=3 NOTE_OFF=63 NOTE_ON=66 TIME_DELTA=3 NOTE_OFF=66 NOTE_ON=63 TIME_DELTA=3 NOTE_OFF=63 NOTE_ON=66 TIME_DELTA=3 NOTE_OFF=75 NOTE_OFF=66 NOTE_ON=76 NOTE_ON=64 TIME_DELTA=3 NOTE_OFF=76 NOTE_OFF=64 NOTE_ON=75 NOTE_ON=68 TIME_DELTA=3 NOTE_OFF=75 NOTE_OFF=68 NOTE_ON=73 NOTE_ON=64 TIME_DELTA=3 NOTE_OFF=73 NOTE_OFF=64 NOTE_ON=68 NOTE_ON=68 TIME_DELTA=3 NOTE_OFF=68 NOTE_ON=59 NOTE_ON=71 TIME_DELTA=3 NOTE_OFF=59 NOTE_ON=63 TIME_DELTA=3 NOTE_OFF=63 NOTE_ON=59 TIME_DELTA=3 NOTE_OFF=59 NOTE_ON=63 TIME_DELTA=3 NOTE_OFF=71 NOTE_OFF=63 NOTE_ON=73 NOTE_ON=61 TIME_DELTA=3 NOTE_OFF=73 NOTE_OFF=61 NOTE_ON=71 NOTE_ON=64 TIME_DELTA=3 NOTE_OFF=71 NOTE_OFF=64 BAR_END TRACK_END TRACK_START INST=0 BAR_START TIME_DELTA=3 NOTE_ON=47 TIME_DELTA=3 NOTE_OFF=47 NOTE_ON=40 NOTE_ON=40 TIME_DELTA=3 NOTE_OFF=40 NOTE_ON=47 TIME_DELTA=6 NOTE_OFF=47 NOTE_ON=47 TIME_DELTA=3 NOTE_OFF=47 NOTE_ON=57 NOTE_ON=61 NOTE_ON=35 NOTE_ON=35 TIME_DELTA=3 NOTE_OFF=57 NOTE_OFF=61 NOTE_OFF=35 NOTE_ON=59 NOTE_ON=63 NOTE_ON=47 TIME_DELTA=3 NOTE_OFF=59 NOTE_OFF=63 NOTE_ON=57 NOTE_ON=61 TIME_DELTA=3 NOTE_OFF=47 NOTE_OFF=57 NOTE_OFF=61 NOTE_ON=59 NOTE_ON=63 NOTE_ON=47 TIME_DELTA=3 NOTE_OFF=59 NOTE_OFF=63 NOTE_OFF=47 NOTE_ON=57 NOTE_ON=61 NOTE_ON=35 NOTE_ON=35 TIME_DELTA=3 NOTE_OFF=57 NOTE_OFF=61 NOTE_OFF=35 NOTE_ON=59 NOTE_ON=63 NOTE_ON=47 TIME_DELTA=3 NOTE_OFF=59 NOTE_OFF=63 NOTE_ON=57 NOTE_ON=61 TIME_DELTA=3 NOTE_OFF=47 NOTE_OFF=57 NOTE_OFF=61 NOTE_ON=59 NOTE_ON=63 NOTE_ON=47 TIME_DELTA=3 NOTE_OFF=59 NOTE_OFF=63 NOTE_OFF=47 NOTE_ON=56 NOTE_ON=40 NOTE_ON=40 TIME_DELTA=3 NOTE_OFF=56 NOTE_OFF=40 NOTE_ON=59 NOTE_ON=47 TIME_DELTA=3 NOTE_OFF=59 NOTE_OFF=47 BAR_END BAR_START NOTE_ON=56 TIME_DELTA=3 NOTE_OFF=56 NOTE_ON=59 NOTE_ON=47 TIME_DELTA=3 NOTE_OFF=59 NOTE_OFF=47 NOTE_ON=40 NOTE_ON=40 TIME_DELTA=3 NOTE_OFF=40 NOTE_ON=52 TIME_DELTA=6 NOTE_OFF=52 NOTE_ON=52 TIME_DELTA=3 NOTE_OFF=52 NOTE_ON=45 NOTE_ON=45 TIME_DELTA=3 NOTE_OFF=45 NOTE_ON=52 TIME_DELTA=6 NOTE_OFF=52 NOTE_ON=52 TIME_DELTA=3 NOTE_OFF=52 NOTE_ON=44 NOTE_ON=44 TIME_DELTA=3 NOTE_OFF=44 NOTE_ON=56 TIME_DELTA=6 NOTE_OFF=56 NOTE_ON=56 TIME_DELTA=3 NOTE_OFF=56 NOTE_ON=49 NOTE_ON=49 TIME_DELTA=3 NOTE_OFF=49 NOTE_ON=56 TIME_DELTA=3 NOTE_OFF=56 BAR_END BAR_START TIME_DELTA=3 NOTE_ON=56 TIME_DELTA=3 NOTE_OFF=56 NOTE_ON=49 NOTE_ON=52 NOTE_ON=54 NOTE_ON=58 TIME_DELTA=3 NOTE_OFF=49 NOTE_OFF=52 NOTE_OFF=54 NOTE_OFF=58 NOTE_ON=49 NOTE_ON=52 NOTE_ON=54 NOTE_ON=58 TIME_DELTA=3 NOTE_OFF=49 NOTE_OFF=52 NOTE_OFF=54 NOTE_OFF=58 NOTE_ON=49 NOTE_ON=52 NOTE_ON=54 NOTE_ON=58 TIME_DELTA=3 NOTE_OFF=49 NOTE_OFF=52 NOTE_OFF=54 NOTE_OFF=58 NOTE_ON=49 NOTE_ON=52 NOTE_ON=54 NOTE_ON=58 TIME_DELTA=3 NOTE_OFF=49 NOTE_OFF=52 NOTE_OFF=54 NOTE_OFF=58 NOTE_ON=48 NOTE_ON=52 NOTE_ON=54 NOTE_ON=58 TIME_DELTA=3 NOTE_OFF=48 NOTE_OFF=52 NOTE_OFF=54 NOTE_OFF=58 NOTE_ON=48 NOTE_ON=52 NOTE_ON=54 NOTE_ON=58 TIME_DELTA=3 NOTE_OFF=48 NOTE_OFF=52 NOTE_OFF=54 NOTE_OFF=58 NOTE_ON=48 NOTE_ON=52 NOTE_ON=54 NOTE_ON=58 TIME_DELTA=3 NOTE_OFF=48 NOTE_OFF=52 NOTE_OFF=54 NOTE_OFF=58 NOTE_ON=48 NOTE_ON=52 NOTE_ON=54 NOTE_ON=58 TIME_DELTA=3 NOTE_OFF=48 NOTE_OFF=52 NOTE_OFF=54 NOTE_OFF=58 NOTE_ON=47 NOTE_ON=52 NOTE_ON=56 NOTE_ON=59 TIME_DELTA=18 NOTE_OFF=47 NOTE_OFF=52 NOTE_OFF=56 NOTE_OFF=59 BAR_END BAR_START TIME_DELTA=6 NOTE_ON=54 TIME_DELTA=3 NOTE_OFF=54 NOTE_ON=59 TIME_DELTA=3 NOTE_OFF=59 NOTE_ON=47 TIME_DELTA=3 NOTE_OFF=47 NOTE_ON=59 TIME_DELTA=3 NOTE_OFF=59 NOTE_ON=49 TIME_DELTA=3 NOTE_OFF=49 NOTE_ON=56 TIME_DELTA=3 NOTE_OFF=56 NOTE_ON=44 TIME_DELTA=3 NOTE_OFF=44 NOTE_ON=56 TIME_DELTA=3 NOTE_OFF=56 NOTE_ON=51 TIME_DELTA=3 NOTE_OFF=51 NOTE_ON=56 TIME_DELTA=3 NOTE_OFF=56 NOTE_ON=44 TIME_DELTA=3 NOTE_OFF=44 NOTE_ON=56 TIME_DELTA=3 NOTE_OFF=56 NOTE_ON=45 TIME_DELTA=3 NOTE_OFF=45 NOTE_ON=52 TIME_DELTA=3 NOTE_OFF=52 BAR_END TRACK_END"


# input_ids_x_tilde = torch.tensor([tokenizer.encode(x_tilde).ids])
# input_ids_x = torch.tensor([tokenizer.encode(x).ids])
# input_ids_y = torch.tensor([tokenizer.encode(y).ids])

# attention_mask_x_tilde = torch.tensor([tokenizer.encode(x_tilde).attention_mask])
# attention_mask_x = torch.tensor([tokenizer.encode(x).attention_mask])
# attention_mask_y = torch.tensor([tokenizer.encode(y).attention_mask])

# feature_vec_x = f(input_ids_x, attention_mask_x)
# feature_vec_x_tilde = f_tilde(input_ids_x_tilde, attention_mask_x_tilde)

# feature_vec_y = f(input_ids_y, attention_mask_y)

# output_x_xtilde = np.dot(feature_vec_x[0].detach().numpy(), feature_vec_x_tilde[0].detach().numpy())

# output_y_xtilde = np.dot(feature_vec_y[0].detach().numpy(), feature_vec_x_tilde[0].detach().numpy())
# #print("feature_vec_x", feature_vec_x[0].detach().numpy())
# #print("feature_vec_x_tilde", feature_vec_x_tilde[0].detach().numpy())
# print("output_x_xtilde: ", output_x_xtilde)

# print("output_y_xtilde: ", output_y_xtilde)

# if output_x_xtilde > output_y_xtilde:
#   print("X näher an X Tilde!!")
# else:
#   print("Y näher an X Tilde")