import os
import glob
from copy import deepcopy
import json
import sklearn

from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np

dirname = os.path.dirname(__file__)

import gpt2_composer

DATASET_PATH = os.path.join(dirname, 'DATA\\beethoven')
TOKENS_PATH = os.path.join(dirname, 'DATA\\TOKENS')
midi_files = glob.glob(os.path.join(DATASET_PATH, "*.mid"))

for fn_mid in tqdm(midi_files):
  fn_txt = os.path.join(TOKENS_PATH, os.path.splitext(os.path.basename(fn_mid))[0] + ".txt")
  note_seq_examples = gpt2_composer.extract_4bar_sections(fn_mid)

  with open(fn_txt, "w") as txt_file:
    for seq in note_seq_examples:
      token_string = " ".join(gpt2_composer.note_sequence_to_token_sequence(seq))
      txt_file.write(token_string + "\n")



txt_files = glob.glob(os.path.join(TOKENS_PATH, "*.txt"))
# train test split
txt_files_train, txt_files_test = sklearn.model_selection.train_test_split(txt_files, test_size=0.1, random_state=42)

# put txt files in large file
with open("DATA/train.txt", "w") as f2:
  for fn_txt in tqdm(txt_files_train):
    with open(fn_txt, "r") as f1:
      f2.write(f1.read())
      
with open("DATA/test.txt", "w") as f2:
  for fn_txt in tqdm(txt_files_test):
    with open(fn_txt, "r") as f1:
      f2.write(f1.read())