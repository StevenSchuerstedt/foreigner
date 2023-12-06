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

    #TODO: freeze transformer when training
    self.transformer = transformers.GPT2Model(config)

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
f = AttributionHead.from_pretrained("checkpoints\checkpoint-1000")

f_tilde = AttributionHead.from_pretrained("checkpoints\checkpoint-1000")
# padding needed?
tokenizer.enable_padding(length=512)


# load dataset
data_files = {"train": "DATA/train.txt", "test": "DATA/test.txt"}
dataset = datasets.load_dataset("text", data_files=data_files)


def tokenize_function(examples):
    outputs = tokenizer.encode_batch(examples["text"])
    example = {
        "input_ids": [c.ids for c in outputs]
    }
    # The 🤗 Transformers library apply the shifting to the right, so we don't need to do it manually.
    example["labels"] = example["input_ids"].copy()

    example["x^-"] = example["input_ids"].copy()
    example["x^~"] = example["input_ids"].copy()
    return example


tokenized_datasets = dataset.map(
    tokenize_function, batched=True, remove_columns=["text"])
train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["test"]



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

# train
training_args = transformers.TrainingArguments("checkpoints", num_train_epochs=10000, save_steps=1000, remove_unused_columns=False)
trainer = AttributionTrainer(
    model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset)
trainer.train()


#custom training loop to train two models (F and F_tilde) simultanously?
# compute loss, depending on both models
# call loss.backward to compute gradients for both models?
# does this work?

