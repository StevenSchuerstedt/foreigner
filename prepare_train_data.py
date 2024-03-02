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

from gpt2_composer import extract_4bar_sections
from gpt2_composer.token_generator import note_sequence_to_token_sequence

dirname = os.path.dirname(__file__)

#import gpt2_composer

composers = [
     #'bach',
       #'beethoven',
        #'chopin',
          # 'grieg',
          #   'haydn',
               'liszt',
          #       'mendelssohn',
          #         'rachmaninov'
                  ]

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
    
    note_seq_examples = extract_4bar_sections(fn_mid)
    for seq in note_seq_examples:
      token_string = " ".join(note_sequence_to_token_sequence(seq))

      if token_string == "PIECE_START TRACK_START INST=0 BAR_START NOTE_ON=84 NOTE_ON=96 TIME_DELTA=6 NOTE_OFF=84 NOTE_OFF=96 NOTE_ON=83 NOTE_ON=95 TIME_DELTA=6 NOTE_OFF=83 NOTE_OFF=95 NOTE_ON=82 NOTE_ON=94 TIME_DELTA=6 NOTE_OFF=82 NOTE_OFF=94 NOTE_ON=81 NOTE_ON=93 TIME_DELTA=6 NOTE_OFF=81 NOTE_OFF=93 NOTE_ON=79 NOTE_ON=91 TIME_DELTA=6 NOTE_OFF=79 NOTE_OFF=91 NOTE_ON=78 NOTE_ON=90 TIME_DELTA=6 NOTE_OFF=78 NOTE_OFF=90 NOTE_ON=77 NOTE_ON=89 TIME_DELTA=6 NOTE_OFF=77 NOTE_OFF=89 NOTE_ON=74 NOTE_ON=86 TIME_DELTA=6 NOTE_OFF=74 NOTE_OFF=86 BAR_END BAR_START NOTE_ON=73 NOTE_ON=85 TIME_DELTA=6 NOTE_OFF=73 NOTE_OFF=85 NOTE_ON=72 NOTE_ON=84 TIME_DELTA=6 NOTE_OFF=72 NOTE_OFF=84 NOTE_ON=71 NOTE_ON=83 TIME_DELTA=6 NOTE_OFF=71 NOTE_OFF=83 NOTE_ON=70 NOTE_ON=82 TIME_DELTA=6 NOTE_OFF=70 NOTE_OFF=82 NOTE_ON=69 NOTE_ON=81 TIME_DELTA=6 NOTE_OFF=69 NOTE_OFF=81 NOTE_ON=70 NOTE_ON=82 TIME_DELTA=6 NOTE_OFF=70 NOTE_OFF=82 NOTE_ON=71 NOTE_ON=83 TIME_DELTA=6 NOTE_OFF=71 NOTE_OFF=83 NOTE_ON=72 NOTE_ON=84 TIME_DELTA=6 NOTE_OFF=72 NOTE_OFF=84 BAR_END BAR_START NOTE_ON=73 NOTE_ON=85 TIME_DELTA=6 NOTE_OFF=73 NOTE_OFF=85 NOTE_ON=74 NOTE_ON=86 TIME_DELTA=6 NOTE_OFF=74 NOTE_OFF=86 NOTE_ON=75 NOTE_ON=87 TIME_DELTA=6 NOTE_OFF=75 NOTE_OFF=87 NOTE_ON=76 NOTE_ON=88 TIME_DELTA=6 NOTE_OFF=76 NOTE_OFF=88 NOTE_ON=77 NOTE_ON=89 TIME_DELTA=6 NOTE_OFF=77 NOTE_OFF=89 NOTE_ON=78 NOTE_ON=90 TIME_DELTA=6 NOTE_OFF=78 NOTE_OFF=90 NOTE_ON=79 NOTE_ON=91 TIME_DELTA=6 NOTE_OFF=79 NOTE_OFF=91 NOTE_ON=80 NOTE_ON=92 TIME_DELTA=6 NOTE_OFF=80 NOTE_OFF=92 BAR_END BAR_START NOTE_ON=81 NOTE_ON=93 TIME_DELTA=6 NOTE_OFF=81 NOTE_OFF=93 NOTE_ON=83 NOTE_ON=95 TIME_DELTA=6 NOTE_OFF=83 NOTE_OFF=95 NOTE_ON=84 NOTE_ON=96 TIME_DELTA=6 NOTE_OFF=84 NOTE_OFF=96 NOTE_ON=85 NOTE_ON=97 TIME_DELTA=6 NOTE_OFF=85 NOTE_OFF=97 NOTE_ON=86 NOTE_ON=98 TIME_DELTA=6 NOTE_OFF=86 NOTE_OFF=98 NOTE_ON=87 NOTE_ON=99 TIME_DELTA=6 NOTE_OFF=87 NOTE_OFF=99 NOTE_ON=88 NOTE_ON=100 TIME_DELTA=6 NOTE_OFF=88 NOTE_OFF=100 TIME_DELTA=6 BAR_END TRACK_END TRACK_START INST=0 BAR_START NOTE_ON=39 NOTE_ON=43 NOTE_ON=51 TIME_DELTA=6 NOTE_OFF=39 NOTE_OFF=43 NOTE_OFF=51 NOTE_ON=39 NOTE_ON=43 NOTE_ON=51 TIME_DELTA=6 NOTE_OFF=39 NOTE_OFF=43 NOTE_OFF=51 NOTE_ON=39 NOTE_ON=43 NOTE_ON=51 TIME_DELTA=12 NOTE_OFF=39 NOTE_OFF=43 NOTE_OFF=51 NOTE_ON=51 NOTE_ON=63 TIME_DELTA=12 NOTE_OFF=51 NOTE_OFF=63 NOTE_ON=50 NOTE_ON=62 TIME_DELTA=12 NOTE_OFF=50 NOTE_OFF=62 BAR_END BAR_START NOTE_ON=52 NOTE_ON=64 TIME_DELTA=12 NOTE_OFF=52 NOTE_OFF=64 NOTE_ON=55 NOTE_ON=67 TIME_DELTA=12 NOTE_OFF=55 NOTE_OFF=67 NOTE_ON=53 NOTE_ON=65 TIME_DELTA=12 NOTE_OFF=53 NOTE_OFF=65 NOTE_ON=52 NOTE_ON=64 TIME_DELTA=12 NOTE_OFF=52 NOTE_OFF=64 BAR_END BAR_START NOTE_ON=50 NOTE_ON=62 TIME_DELTA=12 NOTE_OFF=50 NOTE_OFF=62 NOTE_ON=48 NOTE_ON=60 TIME_DELTA=12 NOTE_OFF=48 NOTE_OFF=60 NOTE_ON=47 NOTE_ON=59 TIME_DELTA=12 NOTE_OFF=47 NOTE_OFF=59 NOTE_ON=45 NOTE_ON=57 TIME_DELTA=12 NOTE_OFF=45 NOTE_OFF=57 BAR_END BAR_START NOTE_ON=43 NOTE_ON=55 TIME_DELTA=12 NOTE_OFF=43 NOTE_OFF=55 NOTE_ON=42 NOTE_ON=54 TIME_DELTA=12 NOTE_OFF=42 NOTE_OFF=54 NOTE_ON=41 NOTE_ON=53 TIME_DELTA=12 NOTE_OFF=41 NOTE_OFF=53 NOTE_ON=40 NOTE_ON=52 TIME_DELTA=6 NOTE_OFF=40 NOTE_OFF=52 TIME_DELTA=6 BAR_END TRACK_END":
          print(fn_mid)
          print("found")
          break

      strr.append(token_string)

  #write to file
  #file = open("DATA/attribution/input_" + composer + ".txt", "w")
  #for item in strr:
	  #file.write(item+"\n")
  
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