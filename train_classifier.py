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
    #TODO freeze later, optimizer only over linear layer parameter
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
data_files = {"generated": "DATA/attribution_generated.txt", "input": "DATA/attribution_input.txt"}
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



#Attribution Trainer

class AttributionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        
        #temperature 
        v = 1

        #inputs are shaped according to batch size (default=8)

        #sample one i randomly to draw according data?
        i = 3
        #t_i = F(x^+)
        #s_i = F^~(x^~)

        #Forward Pass
        f = model
        ftilde = model

        t = f(inputs['x^-'])
        s = ftilde(inputs['x^~'])

        # compute NTXENT Loss
        numerator = torch.exp(torch.dot(t[i], s[i]) / v)
        denomerator = torch.sum(torch.exp(torch.mul(t[i], s)) / v)

        denomerator2 = torch.sum(torch.exp(torch.mul(t, s[i])) / v)

        L_cont = -(torch.log(numerator / denomerator) + torch.log(numerator / denomerator2))

        return L_cont


# output = f(torch.tensor([tokenizer.encode("PIECE_START").ids]))

# print(output.shape)

# train
# training_args = transformers.TrainingArguments("checkpoints", num_train_epochs=10000, save_steps=1000, remove_unused_columns=False)
# trainer = AttributionTrainer(
#     model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset)
# trainer.train()


#custom training loop to train two models (F and F_tilde) simultanously?
# compute loss, depending on both models
# call loss.backward to compute gradients for both models?
# does this work?


def ntxent(t, s, v):
       #sample one i randomly to draw according data?
       #iterate over all is
       L_cont = 0
       for i in range(len(t)):
       #i = 0
       #t_i = F(x^+)
       #s_i = F^~(x^~)
       # compute NTXENT Loss
            numerator = torch.exp(torch.dot(t[i], s[i]) / v)
            denomerator = torch.sum(torch.exp(torch.mul(t[i], s)) / v)

            denomerator2 = torch.sum(torch.exp(torch.mul(t, s[i])) / v)

            L_cont += -(torch.log(numerator / denomerator) + torch.log(numerator / denomerator2))

       return L_cont/len(t)

optimizer_F = torch.optim.SGD(f.parameters(), lr=0.5)
optimizer_F_tilde = torch.optim.SGD(f_tilde.parameters(), lr=0.5)


n_epochs = 10    # number of epochs to run
batch_size = 32  # size of each batch
batches_per_epoch = len(data_x) // batch_size


f = f.to('cuda')
f_tilde = f_tilde.to('cuda')

print("***START TRAINING***")
for i in range(n_epochs):
  for j in range(batches_per_epoch):
    start = j * batch_size
    # take a batch
    X_batch = data_x[start:start+batch_size]
    X_tilde_batch = data_x_tilde[start:start+batch_size]

    #TODO: test custom training loop with optimizing two models with one combined loss
    t = f(torch.tensor(X_batch['input_ids']).to('cuda'))
    s = f_tilde(torch.tensor(X_tilde_batch['input_ids']).to('cuda'))
    #temperature
    v = 1
    loss = ntxent(t, s, v)
    loss.backward()

    #TODO does this exist parameter.grad 

    optimizer_F.step()
    optimizer_F_tilde.step()

    optimizer_F.zero_grad()
    optimizer_F_tilde.zero_grad()
    print("BATCH " + str(j) + " FINISHED")
  print("STEP " + str(i) + " FINISHED")


f.save_pretrained('checkpoint_attribute')
f_tilde.save_pretrained('checkpoint_attribute')

#TODO nan: pytorch detect anomaly