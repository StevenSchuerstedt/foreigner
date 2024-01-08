print("hello world")

import os
import glob
from copy import deepcopy
import json
import sklearn

from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np

import note_seq

dirname = os.path.dirname(__file__)

import gpt2_composer

composers = ['bach', 'beethoven', 'chopin', 'grieg', 'haydn', 'liszt', 'mendelssohn', 'rachmaninov']

DATASET_PATH = os.path.join(dirname, 'DATA\\chopin')
TOKENS_PATH = os.path.join(dirname, 'DATA\\TOKENS')
#midi_files = glob.glob(os.path.join(DATASET_PATH, "*.mid"))


midi_files = []

#midi_files.append(glob.glob(os.path.join(DATASET_PATH, "chpn_op25_e1.mid")))

#midi_files.append(glob.glob(os.path.join(DATASET_PATH, "chpn_op25_e11.mid")))

midi_files.append(glob.glob(os.path.join(DATASET_PATH, "025-1b-LE-001.mid")))


midi_files = sum(midi_files, [])


seq = note_seq.midi_file_to_note_sequence(midi_files[0])
#print(seq)

# split on tempo and time signature changes
sub_seqs = note_seq.sequences_lib.split_note_sequence_on_time_changes(seq)
i = 0
for sub_seq in sub_seqs:
      note_seq.sequence_proto_to_midi_file(sub_seq, "output/test/" + str(i) + ".mid")
      i = i + 1

      # # quantize sequence
      seq_quant = note_seq.quantize_note_sequence(sub_seq, steps_per_quarter=12)
        
      # # split to 4bars parts
      # #seq_bars = gpt2_composer.split_note_seq_nbars(seq_quant)

      total_bars = seq_quant.total_quantized_steps

      print(total_bars)
      # #for seq_bar in seq_bars:
      # #      print("hi")



# for fn_mid in tqdm(midi_files):
#   fn_txt = os.path.join(TOKENS_PATH, os.path.splitext(os.path.basename(fn_mid))[0] + ".txt")
#   note_seq_examples = gpt2_composer.extract_4bar_sections(fn_mid)

#   with open(fn_txt, "w") as txt_file:
#     for seq in note_seq_examples:
#       token_string = " ".join(gpt2_composer.note_sequence_to_token_sequence(seq))
#       txt_file.write(token_string + "\n")



# txt_files = glob.glob(os.path.join(TOKENS_PATH, "*.txt"))
# # train test split
# txt_files_train, txt_files_test = sklearn.model_selection.train_test_split(txt_files, test_size=0.1, random_state=42)

# # put txt files in large file
# with open("DATA/bacht.txt", "w") as f2:
#   for fn_txt in tqdm(txt_files_train):
#     with open(fn_txt, "r") as f1:
#       f2.write(f1.read())
      
# with open("DATA/bachtt.txt", "w") as f2:
#   for fn_txt in tqdm(txt_files_test):
#     with open(fn_txt, "r") as f1:
#       f2.write(f1.read())