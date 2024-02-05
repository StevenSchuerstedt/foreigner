import os
import glob
from copy import deepcopy
import json
import sklearn
import json
import pandas as pd

from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np

dirname = os.path.dirname(__file__)

import gpt2_composer

composers = ['bach', 'beethoven', 'chopin', 'grieg', 'haydn', 'liszt', 'mendelssohn', 'rachmaninov']

#DATASET_PATH = os.path.join(dirname, 'DATA\\chopin')
TOKENS_PATH = os.path.join(dirname, 'DATA\\TOKENS')
#midi_files = glob.glob(os.path.join(DATASET_PATH, "*.mid"))


dic = {}
for composer in composers:
  midi_files = []
  # if composer == 'rachmaninov':
  #   continue

  midi_files.append(glob.glob(os.path.join(os.path.join(dirname, 'DATA\\' + composer), "*.mid")))

  midi_files = sum(midi_files, [])
  strr = []
  for fn_mid in tqdm(midi_files):
    
    note_seq_examples = gpt2_composer.extract_4bar_sections(fn_mid)
    for seq in note_seq_examples:
      token_string = " ".join(gpt2_composer.note_sequence_to_token_sequence(seq))
      strr.append(token_string)

  items = ['Mango', 'Orange', 'Apple', 'Lemon']
  file = open("DATA/attribution/input_" + composer + ".txt", "w")
  for item in strr:
	  file.write(item+"\n")
  
  #file.close()


#print(dic)

#txt_files = glob.glob(os.path.join(TOKENS_PATH, "*.txt"))
# train test split
#txt_files_train, txt_files_test = sklearn.model_selection.train_test_split(txt_files, test_size=0.1, random_state=42)

# with open('data.json', 'w') as fp:
#     json.dump(dic, fp, indent=4, sort_keys=False)


#df = pd.DataFrame(data=dic)
#df.to_csv('input.csv')

# put txt files in large file
# with open("DATA/attribution_input.txt", "w") as f2:
#   for fn_txt in tqdm(txt_files):
#     with open(fn_txt, "r") as f1:
#       f2.write(f1.read())
      
# with open("DATA/test_-rachmaninov.txt", "w") as f2:
#   for fn_txt in tqdm(txt_files_test):
#     with open(fn_txt, "r") as f1:
#       f2.write(f1.read())