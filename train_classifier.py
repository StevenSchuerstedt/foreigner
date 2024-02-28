from typing import Optional
import numpy as np
import note_seq
import transformers
import torch
import datasets
from AttributionHeadBert import AttributionHeadBert
import gpt2_composer
from torch import nn
from transformers import Trainer
import random
import math
from AttributionHead import AttributionHead


tokenizer = gpt2_composer.load_tokenizer("")
f = AttributionHeadBert("checkpoint_bert\checkpoint-22500")

f_tilde = AttributionHeadBert("checkpoint_bert\checkpoint-22500")

# padding needed?
tokenizer.enable_padding(length=512, pad_id=f.transformer.config.pad_token_id)


# o = f(torch.tensor(tokenizer.encode("PIECE_START").ids))
# print(o.shape)

# print(o)
composers = ['bach',
              'beethoven',
               # 'chopin',
                 # 'grieg',
                  #  'haydn',
                  #    'liszt',
                   #     'mendelssohn',
                    #      'rachmaninov'
                    ]

# load dataset
data_files = {
    "generated_bach": "DATA/attribution/generated/generated_bach.txt",
    "generated_beethoven": "DATA/attribution/generated/generated_beethoven.txt",
    "generated_chopin": "DATA/attribution/generated/generated_chopin.txt",
    "generated_grieg": "DATA/attribution/generated/generated_grieg.txt",
    "generated_haydn": "DATA/attribution/generated/generated_haydn.txt",
    "generated_liszt": "DATA/attribution/generated/generated_liszt.txt",
    "generated_mendelssohn": "DATA/attribution/generated/generated_mendelssohn.txt",
    "generated_rachmaninov": "DATA/attribution/generated/generated_rachmaninov.txt",

    "input_bach": "DATA/attribution/input/input_bach.txt",
    "input_beethoven": "DATA/attribution/input/input_beethoven.txt",
    "input_chopin": "DATA/attribution/input/input_chopin.txt",
    "input_grieg": "DATA/attribution/input/input_grieg.txt",
    "input_haydn": "DATA/attribution/input/input_haydn.txt",
    "input_liszt": "DATA/attribution/input/input_liszt.txt",
    "input_mendelssohn": "DATA/attribution/input/input_mendelssohn.txt",
    "input_rachmaninov": "DATA/attribution/input/input_rachmaninov.txt",

    "generated_bach_test": "DATA/attribution/generated/generated_bach_test.txt",
    "generated_beethoven_test": "DATA/attribution/generated/generated_beethoven_test.txt",
    "generated_chopin_test": "DATA/attribution/generated/generated_chopin_test.txt",
    "generated_grieg_test": "DATA/attribution/generated/generated_grieg_test.txt",
    "generated_haydn_test": "DATA/attribution/generated/generated_haydn_test.txt",
    "generated_liszt_test": "DATA/attribution/generated/generated_liszt_test.txt",
    "generated_mendelssohn_test": "DATA/attribution/generated/generated_mendelssohn_test.txt",
    "generated_rachmaninov_test": "DATA/attribution/generated/generated_rachmaninov_test.txt",

    "input_bach_test": "DATA/attribution/input/input_bach_test.txt",
    "input_beethoven_test": "DATA/attribution/input/input_beethoven_test.txt",
    "input_chopin_test": "DATA/attribution/input/input_chopin_test.txt",
    "input_grieg_test": "DATA/attribution/input/input_grieg_test.txt",
    "input_haydn_test": "DATA/attribution/input/input_haydn_test.txt",
    "input_liszt_test": "DATA/attribution/input/input_liszt_test.txt",
    "input_mendelssohn_test": "DATA/attribution/input/input_mendelssohn_test.txt",
    "input_rachmaninov_test": "DATA/attribution/input/input_rachmaninov_test.txt",
          }

dataset = datasets.load_dataset("text", data_files=data_files)



def tokenize_function(examples):
    outputs = tokenizer.encode_batch(examples["text"])
    example = {
        "input_ids": [c.ids for c in outputs]
    }
    # The 🤗 Transformers library apply the shifting to the right, so we don't need to do it manually.
    #example["labels"] = example["input_ids"].copy()

    #example["x"] = example["train"].copy()
    #example["x^~"] = example["generated"].copy()
    return example


tokenized_datasets = dataset.map(
    tokenize_function, batched=True, remove_columns=["text"])


# data_x = tokenized_datasets["input"]
# data_x_tilde = tokenized_datasets["generated"]


def ntxent(t, s, v):
       #iterate over all is
       L_cont = 0
       for i in range(len(t)):
            #t_i = F(x^+) => exemplar Set
            #s_i = F^~(x^~) => generated Example

            # compute NTXENT Loss
            A = torch.dot(t[i], s[i]) / v
            B = torch.matmul(t[i], s.transpose(0,1)) / v
            C = torch.matmul(t, s[i]) / v


            #!! use logsumexp to avoid NAN gradients, as exp() will produce numbers outside floating point range
        
            A1 = A
            B1 = torch.logsumexp(B, dim=0)
            C1 = torch.logsumexp(C, dim=0)

            L_cont += -( (A1 - B1) + (A1 - C1))

       return L_cont/len(t)


def calculate_validation_loss():
   t = f(torch.tensor(test_x['input_ids']).to(device))
   s = f_tilde(torch.tensor(test_x_tilde['input_ids']).to(device))
   v = 1
   loss = ntxent(t, s, v)
   return loss



optimizer_F = torch.optim.Adam(f.parameters(), betas=[0.9, 0.999], lr=0.00001)
optimizer_F_tilde = torch.optim.Adam(f_tilde.parameters(), betas=[0.9, 0.999], lr=0.00001)


n_epochs = 100    # number of epochs to run

batch_size = 8  # size of each batch
batches_per_epoch = 100

device = 'cpu'

# if torch.cuda.is_available:
#    device = 'cuda'

f = f.to(device)
f_tilde = f_tilde.to(device)

def calculate_regularizer():
   #L1-Regularization
   f_param = []

   for param in f.parameters():
    f_param.append(param)
   f_tilde_param = []
   for param in f_tilde.parameters():
    f_tilde_param.append(param)
   w = f_param[-1]
   w_tilde = f_tilde_param[-1]
   
   regularizer_loss = 0.5 * (torch.norm(torch.t(w) * w) + torch.norm(torch.t(w_tilde) * w_tilde))
   return regularizer_loss

#torch.autograd.set_detect_anomaly(True)

#TODO: implement regularizer?

print("***START TRAINING***")
for i in range(n_epochs):
  for j in range(batches_per_epoch):

    optimizer_F.zero_grad()
    optimizer_F_tilde.zero_grad()

    start = j * batch_size
    # take a batch
    data_x = []
    data_x_tilde = []
    for composer in composers:
      data_x.append(random.choice(tokenized_datasets['input_' + composer]['input_ids']))
      data_x_tilde.append(random.choice(tokenized_datasets['generated_' + composer]['input_ids']))
    X_batch = data_x
    X_tilde_batch = data_x_tilde


    #TODO: test custom training loop with optimizing two models with one combined loss
    t = f(torch.tensor(X_batch).to(device))
    s = f_tilde(torch.tensor(X_tilde_batch).to(device))
    #temperature
    v = 1
    loss = ntxent(t, s, v)

    l = 0.05
    loss = loss + l * calculate_regularizer()
    

    loss.backward()

    #clip gradients TODO: alternative: register hook to clip DURING backpropagation
    #torch.nn.utils.clip_grad_norm_(f.parameters(), 100, error_if_nonfinite=True)
    #torch.nn.utils.clip_grad_norm_(f_tilde.parameters(), 100, error_if_nonfinite=True)

    # debug gradients
    # p = []
    # for param in f.parameters():
    #    #print(param)
    #    p.append(param)
    
    # print(p[-1].grad)
    

    optimizer_F.step()
    optimizer_F_tilde.step()

    print("LOSS: ", loss)
    print("VALIDATION LOSS: ", calculate_validation_loss())
    print("BATCH " + str(j) + " FINISHED")
  print("EPOCH " + str(i) + " FINISHED")


#f.save('checkpoint_attribute_f')
#f_tilde.save_pretrained('checkpoint_attribute_f_tilde')
