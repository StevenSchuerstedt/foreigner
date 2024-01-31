import numpy as np
import note_seq
import transformers
import datasets
import torch
from typing import Optional

import gpt2_composer

from AttributionHead import AttributionHead


tokenizer = gpt2_composer.load_tokenizer("")

#TODO: WHY IS PADDING SO IMPOORTANT????? ARGHHHHH
tokenizer.enable_padding(length=512)
f = AttributionHead("checkpoints/checkpoint-22500_new_basemodel")
f_tilde = AttributionHead("checkpoints/checkpoint-22500_new_basemodel")


f.load("checkpoint_attribute/f", "checkpoint_attribute/transformer_f")
f_tilde.load("checkpoint_attribute/f_tilde", "checkpoint_attribute/transformer_f_tilde")

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
  input_ids_x = torch.tensor([tokenizer.encode(x).ids])

  feature_vec_x = f(input_ids_x)
  feature_vec_x_tilde = f_tilde(input_ids_x_tilde)

  output_x_xtilde = np.dot(feature_vec_x[0].detach().numpy(), feature_vec_x_tilde[0].detach().numpy())

  print(i)
  print("Ground Truth: ", output_x_xtilde)
  for j in range(8):
    if i == j: 
      continue
    y = dataset['input'][j]['text']

    input_ids_y = torch.tensor([tokenizer.encode(y).ids])
    feature_vec_y = f(input_ids_y)
    output_y_xtilde = np.dot(feature_vec_y[0].detach().numpy(), feature_vec_x_tilde[0].detach().numpy())
    print("Score for: ", j)
    print(output_y_xtilde)
    if output_x_xtilde > output_y_xtilde:
      score = score + 1
  print("TOTAL SCORE: ", score)
  grand_score = grand_score + score


print("GRAND SCORE: ", grand_score)


# #generated piece with finetuned bach
# x_tilde = "PIECE_START TRACK_START INST=0 BAR_START NOTE_ON=76 NOTE_ON=79 TIME_DELTA=12 NOTE_OFF=76 NOTE_OFF=79 NOTE_ON=72 NOTE_ON=76 TIME_DELTA=12 NOTE_OFF=72 NOTE_OFF=76 NOTE_ON=81 NOTE_ON=85 TIME_DELTA=12 NOTE_OFF=81 NOTE_OFF=85 NOTE_ON=75 NOTE_ON=78 TIME_DELTA=12 NOTE_OFF=75 NOTE_OFF=78 BAR_END BAR_START NOTE_ON=80 NOTE_ON=83 TIME_DELTA=12 NOTE_OFF=80 NOTE_OFF=83 NOTE_ON=75 NOTE_ON=80 TIME_DELTA=12 NOTE_OFF=75 NOTE_OFF=80 NOTE_ON=81 NOTE_ON=85 TIME_DELTA=12 NOTE_OFF=81 NOTE_OFF=85 NOTE_ON=75 NOTE_ON=78 TIME_DELTA=12 NOTE_OFF=75 NOTE_OFF=78 BAR_END BAR_START NOTE_ON=80 NOTE_ON=83 TIME_DELTA=12 NOTE_OFF=80 NOTE_OFF=83 NOTE_ON=76 NOTE_ON=80 TIME_DELTA=12 NOTE_OFF=76 NOTE_OFF=80 NOTE_ON=80 NOTE_ON=83 TIME_DELTA=12 NOTE_OFF=80 NOTE_OFF=83 NOTE_ON=76 NOTE_ON=80 TIME_DELTA=12 NOTE_OFF=76 NOTE_OFF=80 BAR_END BAR_START NOTE_ON=80 NOTE_ON=83 TIME_DELTA=12 NOTE_OFF=80 NOTE_OFF=83 NOTE_ON=76 NOTE_ON=80 TIME_DELTA=12 NOTE_OFF=76 NOTE_OFF=80 NOTE_ON=80 NOTE_ON=83 TIME_DELTA=12 NOTE_OFF=80 NOTE_OFF=83 NOTE_ON=76 NOTE_ON=80 TIME_DELTA=12 NOTE_OFF=76 NOTE_OFF=80 BAR_END TRACK_END TRACK_START INST=0 BAR_START NOTE_ON=60 NOTE_ON=67 TIME_DELTA=12 NOTE_OFF=60 NOTE_OFF=67 NOTE_ON=60 NOTE_ON=67 TIME_DELTA=12 NOTE_OFF=60 NOTE_OFF=67 NOTE_ON=64 NOTE_ON=71 TIME_DELTA=12 NOTE_OFF=64 NOTE_OFF=71 NOTE_ON=64 NOTE_ON=71 TIME_DELTA=12 NOTE_OFF=64 NOTE_OFF=71 BAR_END BAR_START NOTE_ON=64 NOTE_ON=71 TIME_DELTA=12 NOTE_OFF=64 NOTE_OFF=71 NOTE_ON=64 NOTE_ON=71 TIME_DELTA=12 NOTE_OFF=64 NOTE_OFF=71 NOTE_ON=64 NOTE_ON=71 TIME_DELTA=12 NOTE_OFF=64 NOTE_OFF=71 NOTE_ON=64 NOTE_ON=71 TIME_DELTA=12 NOTE_OFF=64 NOTE_OFF=71 BAR_END BAR_START NOTE_ON=64 NOTE_ON=71 TIME_DELTA=12 NOTE_OFF=64 NOTE_OFF=71 NOTE_ON=64 NOTE_ON=71 TIME_DELTA=12 NOTE_OFF=64 NOTE_OFF=71 NOTE_ON=64 NOTE_ON=71 TIME_DELTA=12 NOTE_OFF=64 NOTE_OFF=71 NOTE_ON=64 NOTE_ON=71 TIME_DELTA=12 NOTE_OFF=64 NOTE_OFF=71 BAR_END BAR_START NOTE_ON=64 NOTE_ON=71 TIME_DELTA=12 NOTE_OFF=64 NOTE_OFF=71 NOTE_ON=64 NOTE_ON=71 TIME_DELTA=12 NOTE_OFF=64 NOTE_OFF=71 NOTE_ON=64 NOTE_ON=71 TIME_DELTA=12 NOTE_OFF=64 NOTE_OFF=71 NOTE_ON=64 NOTE_ON=71 TIME_DELTA=12 NOTE_OFF=64 NOTE_OFF=71 BAR_END TRACK_END"

# #data of bach
# x = "PIECE_START TRACK_START INST=0 BAR_START NOTE_ON=71 TIME_DELTA=12 NOTE_OFF=71 NOTE_ON=64 TIME_DELTA=9 NOTE_ON=66 TIME_DELTA=3 NOTE_OFF=64 NOTE_OFF=66 NOTE_ON=67 TIME_DELTA=9 NOTE_OFF=67 NOTE_ON=66 NOTE_ON=69 TIME_DELTA=3 NOTE_OFF=66 NOTE_OFF=69 NOTE_ON=71 NOTE_ON=67 TIME_DELTA=9 NOTE_OFF=71 NOTE_ON=76 TIME_DELTA=3 NOTE_OFF=67 NOTE_OFF=76 BAR_END BAR_START NOTE_ON=79 NOTE_ON=72 TIME_DELTA=9 NOTE_OFF=79 NOTE_ON=78 TIME_DELTA=3 NOTE_OFF=72 NOTE_OFF=78 NOTE_ON=74 NOTE_ON=66 TIME_DELTA=9 NOTE_OFF=74 NOTE_ON=72 TIME_DELTA=3 NOTE_OFF=66 NOTE_OFF=72 NOTE_ON=76 NOTE_ON=69 TIME_DELTA=9 NOTE_OFF=76 NOTE_OFF=69 TIME_DELTA=15 BAR_END TRACK_END TRACK_START INST=0 BAR_START TIME_DELTA=24 NOTE_ON=62 NOTE_ON=64 TIME_DELTA=12 NOTE_OFF=62 NOTE_ON=61 TIME_DELTA=12 NOTE_OFF=64 NOTE_OFF=61 BAR_END BAR_START NOTE_ON=64 NOTE_ON=60 TIME_DELTA=12 NOTE_OFF=64 NOTE_OFF=60 NOTE_ON=62 NOTE_ON=59 TIME_DELTA=12 NOTE_OFF=62 NOTE_OFF=59 NOTE_ON=60 NOTE_ON=57 TIME_DELTA=9 NOTE_OFF=60 NOTE_OFF=57 TIME_DELTA=15 BAR_END TRACK_END"
# #no data of bach
# y = "PIECE_START TRACK_START INST=0 BAR_START NOTE_ON=69 TIME_DELTA=12 NOTE_OFF=69 NOTE_ON=69 TIME_DELTA=12 NOTE_OFF=69 NOTE_ON=74 TIME_DELTA=12 NOTE_OFF=74 NOTE_ON=72 TIME_DELTA=12 NOTE_OFF=72 BAR_END BAR_START NOTE_ON=72 TIME_DELTA=12 NOTE_OFF=72 NOTE_ON=74 TIME_DELTA=12 NOTE_OFF=74 NOTE_ON=72 TIME_DELTA=24 NOTE_OFF=72 BAR_END BAR_START NOTE_ON=72 TIME_DELTA=12 NOTE_OFF=72 NOTE_ON=74 TIME_DELTA=12 NOTE_OFF=74 NOTE_ON=75 TIME_DELTA=12 NOTE_OFF=75 NOTE_ON=74 TIME_DELTA=12 NOTE_OFF=74 BAR_END BAR_START NOTE_ON=72 TIME_DELTA=24 NOTE_OFF=72 NOTE_ON=70 TIME_DELTA=24 NOTE_OFF=70 BAR_END TRACK_END TRACK_START INST=0 BAR_START NOTE_ON=65 TIME_DELTA=12 NOTE_OFF=65 NOTE_ON=65 TIME_DELTA=12 NOTE_OFF=65 NOTE_ON=65 TIME_DELTA=12 NOTE_OFF=65 NOTE_ON=65 TIME_DELTA=12 NOTE_OFF=65 BAR_END BAR_START NOTE_ON=65 TIME_DELTA=12 NOTE_OFF=65 NOTE_ON=70 TIME_DELTA=6 NOTE_OFF=70 NOTE_ON=69 TIME_DELTA=6 NOTE_OFF=69 NOTE_ON=67 TIME_DELTA=24 NOTE_OFF=67 BAR_END BAR_START NOTE_ON=69 TIME_DELTA=12 NOTE_OFF=69 NOTE_ON=70 TIME_DELTA=18 NOTE_OFF=70 NOTE_ON=69 TIME_DELTA=6 NOTE_OFF=69 NOTE_ON=70 TIME_DELTA=12 NOTE_OFF=70 BAR_END BAR_START TIME_DELTA=6 NOTE_ON=69 TIME_DELTA=3 NOTE_OFF=69 NOTE_ON=67 TIME_DELTA=3 NOTE_OFF=67 NOTE_ON=69 TIME_DELTA=12 NOTE_OFF=69 NOTE_ON=65 TIME_DELTA=24 NOTE_OFF=65 BAR_END TRACK_END TRACK_START INST=0 BAR_START NOTE_ON=60 TIME_DELTA=12 NOTE_OFF=60 NOTE_ON=60 TIME_DELTA=12 NOTE_OFF=60 NOTE_ON=58 TIME_DELTA=12 NOTE_OFF=58 NOTE_ON=57 TIME_DELTA=6 NOTE_OFF=57 NOTE_ON=58 TIME_DELTA=6 NOTE_OFF=58 BAR_END BAR_START NOTE_ON=60 TIME_DELTA=12 NOTE_OFF=60 NOTE_ON=65 TIME_DELTA=12 NOTE_OFF=65 NOTE_ON=64 TIME_DELTA=24 NOTE_OFF=64 BAR_END BAR_START NOTE_ON=65 TIME_DELTA=12 NOTE_OFF=65 NOTE_ON=65 TIME_DELTA=12 NOTE_OFF=65 NOTE_ON=63 TIME_DELTA=6 NOTE_OFF=63 NOTE_ON=60 TIME_DELTA=6 NOTE_OFF=60 NOTE_ON=65 TIME_DELTA=6 NOTE_OFF=65 NOTE_ON=67 TIME_DELTA=6 NOTE_OFF=67 BAR_END BAR_START NOTE_ON=60 TIME_DELTA=6 NOTE_OFF=60 NOTE_ON=65 TIME_DELTA=6 NOTE_OFF=65 NOTE_ON=65 TIME_DELTA=6 NOTE_OFF=65 NOTE_ON=63 TIME_DELTA=6 NOTE_OFF=63 NOTE_ON=62 TIME_DELTA=24 NOTE_OFF=62 BAR_END TRACK_END TRACK_START INST=0 BAR_START NOTE_ON=53 TIME_DELTA=12 NOTE_OFF=53 NOTE_ON=53 TIME_DELTA=12 NOTE_OFF=53 NOTE_ON=58 TIME_DELTA=12 NOTE_OFF=58 NOTE_ON=53 TIME_DELTA=6 NOTE_OFF=53 NOTE_ON=55 TIME_DELTA=6 NOTE_OFF=55 BAR_END BAR_START NOTE_ON=57 TIME_DELTA=6 NOTE_OFF=57 NOTE_ON=53 TIME_DELTA=6 NOTE_OFF=53 NOTE_ON=58 TIME_DELTA=12 NOTE_OFF=58 NOTE_ON=60 TIME_DELTA=24 NOTE_OFF=60 BAR_END BAR_START NOTE_ON=53 TIME_DELTA=6 NOTE_OFF=53 NOTE_ON=52 TIME_DELTA=6 NOTE_OFF=52 NOTE_ON=50 TIME_DELTA=12 NOTE_OFF=50 NOTE_ON=48 TIME_DELTA=12 NOTE_OFF=48 NOTE_ON=50 TIME_DELTA=6 NOTE_OFF=50 NOTE_ON=51 TIME_DELTA=6 NOTE_OFF=51 BAR_END BAR_START NOTE_ON=53 TIME_DELTA=12 NOTE_OFF=53 NOTE_ON=41 TIME_DELTA=12 NOTE_OFF=41 NOTE_ON=46 TIME_DELTA=24 NOTE_OFF=46 BAR_END TRACK_END"

# input_ids_x_tilde = torch.tensor([tokenizer.encode(x_tilde).ids])
# input_ids_x = torch.tensor([tokenizer.encode(x).ids])
# input_ids_y = torch.tensor([tokenizer.encode(y).ids])

# feature_vec_x = f(input_ids_x)
# feature_vec_x_tilde = f_tilde(input_ids_x_tilde)

# feature_vec_y = f(input_ids_y)

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