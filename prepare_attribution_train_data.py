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

composers = ['bach', 'beethoven', 'chopin', 'grieg', 'haydn', 'liszt', 'mendelssohn', 'rachmaninov']

#DATASET_PATH = os.path.join(dirname, 'DATA\\chopin')
TOKENS_PATH = os.path.join(dirname, 'DATA\\TOKENS')
#midi_files = glob.glob(os.path.join(DATASET_PATH, "*.mid"))


TOKENS_PATH = os.path.join(dirname, 'output')
#midi_files = glob.glob(os.path.join(DATASET_PATH, "*.mid"))

txt_files = []
for composer in composers:
  txt_files.append(glob.glob(os.path.join(TOKENS_PATH + '\\-' + composer, "*.txt")))

#dirty flatten
txt_files = sum(txt_files, [])

# train test split
#txt_files_train, txt_files_test = sklearn.model_selection.train_test_split(txt_files, test_size=0.1, random_state=42)

# put txt files in large file
with open("DATA/attribution_input.txt", "w") as f2:
  for fn_txt in tqdm(txt_files):
    with open(fn_txt, "r") as f1:
      f2.write(f1.read() + '\n')
      
# with open("DATA/attribution_input_test.txt", "w") as f2:
#   for fn_txt in tqdm(txt_files_test):
#     with open(fn_txt, "r") as f1:
#       f2.write(f1.read())