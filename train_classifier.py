from typing import Optional
import numpy as np
import note_seq
import transformers
import torch
import datasets
import gpt2_composer
from torch import nn
from transformers import Trainer


# create Attribution model
class AttributionHead(transformers.GPT2PreTrainedModel):
  
  def __init__(self, config):
    super().__init__(config)

    #does this work???
    self.transformer = transformers.GPT2Model(config)


    #freeze GPT2 layers
    for param in self.transformer.parameters():
      param.requires_grad = False

    self.model_parallel = False
    #what is size of features going out??
    out_features = 512
    self.attribution_head = torch.nn.Linear(config.n_embd, out_features)

  def forward(self, input_ids, labels: Optional[torch.LongTensor] = None,):
    transformer_outputs = self.transformer(input_ids)

    hidden_states = transformer_outputs[0]

    #only select last token TODO: change model to BERT? 
    hidden_state = hidden_states[:, -1]

    attribution_logits = self.attribution_head(hidden_state)
    return attribution_logits 


tokenizer = gpt2_composer.load_tokenizer("")
f = AttributionHead.from_pretrained("checkpoints\checkpoint-15000-basemodel")

f_tilde = AttributionHead.from_pretrained("checkpoints\checkpoint-15000-basemodel")

# padding needed?
tokenizer.enable_padding(length=512)


# o = f(torch.tensor(tokenizer.encode("PIECE_START").ids))
# print(o.shape)

# print(o)


# load dataset
data_files = {"generated": "DATA/attribution_generated_old.txt", "input": "DATA/attribution_input_old.txt"}
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
data_x = tokenized_datasets["input"]
data_x_tilde = tokenized_datasets["generated"]


#custom training loop to train two models (F and F_tilde) simultanously?
# compute loss, depending on both models
# call loss.backward to compute gradients for both models?
# does this work? => I guess

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
            #TODO: logsumexp(A, 0) = A ??

            A1 = torch.logsumexp(A, 0)
            B1 = torch.logsumexp(B, 0)
            C1 = torch.logsumexp(C, 0)

            L_cont += -( (A1 - B1) + (A1 - C1))
            #L_cont += -(torch.log(numerator / denomerator) + torch.log(numerator / denomerator2))

       return L_cont/len(t)

def ntxent_version2(t, s, v):
       #iterate over all is
       L_cont = 0
       #8 composers
       for i in range(8):
            #all data of one composer
            for j in range(100):
               #generate a bach
               x_tilde = s[i * 100 + j]
               x_plus = t[i * 100 + j]
               A = torch.dot(x_plus, x_tilde)
               B = torch.matmul(x_plus, t.transpose(0,1))
               C = torch.matmul(t, x_tilde)

               A1 = A
               B1 = B
               C1 = torch.logsumexp(C, dim=0)

              #TODO: count how many nan values, division by zero ?? how many nan in sum of logsumexp, in iterations

               L_cont += -( (A1 - B1) + (A1 - C1))
       return L_cont/len(t)

optimizer_F = torch.optim.SGD(f.parameters(), lr=0.5)
optimizer_F_tilde = torch.optim.SGD(f_tilde.parameters(), lr=0.5)


n_epochs = 100    # number of epochs to run
batch_size = 8  # size of each batch
batches_per_epoch = len(data_x) // batch_size

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
    start = j * batch_size
    # take a batch
    X_batch = data_x[start:start+batch_size]
    X_tilde_batch = data_x_tilde[start:start+batch_size]

    #TODO: test custom training loop with optimizing two models with one combined loss
    t = f(torch.tensor(X_batch['input_ids']).to(device))
    s = f_tilde(torch.tensor(X_tilde_batch['input_ids']).to(device))
    #temperature
    v = 1
    loss = ntxent(t, s, v)

    #TODO: add L1 (?) Regularization
    l = 0.05
    loss = loss + l * calculate_regularizer()
    

    loss.backward()

    #clip gradients TODO: alternative: register hook to clip DURING backpropagation
    torch.nn.utils.clip_grad_norm_(f.parameters(), 100, error_if_nonfinite=True)
    torch.nn.utils.clip_grad_norm_(f_tilde.parameters(), 100, error_if_nonfinite=True)

    # debug gradients
    # p = []
    # for param in f.parameters():
    #    #print(param)
    #    p.append(param)
    
    # print(p[-1].grad)
    

    optimizer_F.step()
    optimizer_F_tilde.step()

    optimizer_F.zero_grad()
    optimizer_F_tilde.zero_grad()
    print("BATCH " + str(j) + " FINISHED")
  print("STEP " + str(i) + " FINISHED")


f.save_pretrained('checkpoint_attribute_f')
f_tilde.save_pretrained('checkpoint_attribute_f_tilde')
