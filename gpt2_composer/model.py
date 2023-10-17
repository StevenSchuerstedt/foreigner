import os

import tokenizers
import transformers


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config/")

def load_tokenizer(dataset_name):
  fn = os.path.join(CONFIG_PATH, dataset_name, "tokenizer.json")
  return tokenizers.Tokenizer.from_file(fn)

def create_model(dataset_name):
  fn = os.path.join(CONFIG_PATH, dataset_name, "config.json")  
  config = transformers.GPT2Config.from_json_file(fn)
  return transformers.GPT2LMHeadModel(config)