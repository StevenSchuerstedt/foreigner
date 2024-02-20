import copy
from typing import Optional
import numpy as np
import note_seq
import transformers
import torch
import datasets
import gpt2_composer
from torch import nn 
import torch.nn.functional as F
from transformers import Trainer
import random
from AttributionHead import AttributionHead
from ProbabilityScore import ProbabilityScore
import math

#load model

tokenizer = gpt2_composer.load_tokenizer("")

f = AttributionHead("checkpoints/checkpoint-22500")
f_tilde = AttributionHead("checkpoints/checkpoint-22500")
tokenizer.enable_padding(length=512, pad_id=f.transformer.config.pad_token_id)

f.load("checkpoint_attribute/checkpoint_attribute_f", "checkpoint_attribute/transformer_f")
f_tilde.load("checkpoint_attribute/checkpoint_attribute_f_tilde", "checkpoint_attribute/transformer_f_tilde")

composers = ['bach',
              'beethoven',
                'chopin',
                  'grieg',
                    'haydn',
                      'liszt',
                        'mendelssohn',
                          'rachmaninov']
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
    "input_rachmaninov": "DATA/attribution/input/input_rachmaninov.txt"
          }
dataset = datasets.load_dataset("text", data_files=data_files)


def tokenize_function(examples):
    outputs = tokenizer.encode_batch(examples["text"])
    example = {
        "input_ids": [c.ids for c in outputs],
        "attention_mask" : [c.attention_mask for c in outputs]
    }
    # The 🤗 Transformers library apply the shifting to the right, so we don't need to do it manually.
    #example["labels"] = example["input_ids"].copy()

    #example["x"] = example["train"].copy()
    #example["x^~"] = example["generated"].copy()
    return example


tokenized_datasets = dataset.map(
    tokenize_function, batched=True, remove_columns=["text"])


#start with only input data, generated data, x and x_tilde?? calculate s in loss function? or beforehand
datapairs = []


for n in range(10):

    composer = random.choice(composers)

    x_tilde = random.choice(tokenized_datasets['generated_' + composers])

    #print(x_tilde_index)

    input_ids_x_tilde = torch.tensor([x_tilde['input_ids']])
    attention_x_tilde = torch.tensor([x_tilde['attention_mask']])

    #print(input_ids_x_tilde)


    feature_vec_x_tilde = f_tilde(input_ids_x_tilde, attention_x_tilde)

    #calculate similarity scores

    #list of similarit scores, => s = dot(F(x), F_tilde(x_tilde))
    s = []

    #list of ground truths (0 = false, 1 = true)
    t = []

    for j in range(2):
        for i in range(8):

           

            composer_c = random.choice(composers)
            x_index = random.choice(tokenized_datasets['input_' + composers])

            input_ids_x = torch.tensor([x_index['input_ids']])
            attention_x = torch.tensor([x_index['attention_mask']])
            feature_vec_x = f(input_ids_x, attention_x)

            similarity_score = np.dot(feature_vec_x[0].detach().numpy(), feature_vec_x_tilde[0].detach().numpy())
        
            s.append(similarity_score)
            if(composer_c == composer):
                t.append(1)
            else:
                t.append(0)

            #print("x_index:", x_index)

    

    datapair = {"x_tilde": 0,
        "similarity_scores": s,
        "ground_truths": t
        }


    datapairs.append(datapair)

#print(datapairs)

# P = torch.tensor(np.ones(len(datapairs[0]['similarity_scores'])))

# print("similarity_scores: ", datapairs[0]['similarity_scores'])
# print("similarity_scores sorted: ", np.flip(np.sort(datapairs[0]['similarity_scores'])))

# for i in range(len(datapairs[0]['similarity_scores'])):
        
#             sorted = np.flip(np.sort(datapairs[0]['similarity_scores']))
#             n = F.softplus(torch.exp(torch.tensor(datapairs[0]['similarity_scores'][i] - sorted[0])/1) - 0)
#             print("n: ", n)


#             A = torch.exp( torch.tensor((sorted - sorted[0]))/1 )
#             print("A: ", A)
#             B = F.softplus( A- 0)
#             d = torch.sum(B)
        
#             P[i] = n / d

# print("P:", P)

def klLoss(input, target):
    expectation_value = 0
    for datapair in datapairs:

        #construct ground truths
        S = torch.tensor(np.zeros(len(datapair['ground_truths'])))

        count_ground_truth = np.count_nonzero(datapair['ground_truths'])

        for index, element in enumerate(datapair['ground_truths']):
            if element == 1:
                S[index] = 1 / count_ground_truth

        #construct probabilites
        #print("grounrd truth: ", S)
        #print("sscores: ", datapair['similarity_scores'])
        P = model(datapair['similarity_scores'])

        #print("S", S)
        #print("P", P)
        #calculate kl difference
        #TODO: test order ( P, S)
        #BEWARE: target in kl div shall be in log probs
        kl = F.kl_div(P.log(), S)
        #print("kl:", kl)
        expectation_value = expectation_value + kl
    #print("expectation_value", expectation_value / len(datapairs))
    return expectation_value / len(datapairs)



model = ProbabilityScore()
# tau = torch.nn.Parameter(torch.Tensor([1.0]))
# lámbda = torch.nn.Parameter(torch.Tensor([0.0]))
optimizer = torch.optim.Adam([
    {'params': model.tau, 'lr': 0.5},
    {'params': model.lámbda, 'lr': 0.0005}
], lr=0.0001)


#training loop
steps = 3000
print("Start TRAINING!!")
print("lámbda", model.lámbda)
print("tau", model.tau)
for i in range(steps):
    
    loss = klLoss(0,0)

    loss.backward()

    optimizer.step()

    optimizer.zero_grad()
    if i % 100 == 0:
        print("STEP ", str(i), "Finished!!")
        print("LOSS: ", loss)
    #print("lámbda", model.lámbda)
    #print("tau", model.tau)

print("FINISHED!! RESULTS:")
print("lámbda", model.lámbda)
print("tau", model.tau)

torch.save(model, 'probability_score/P')