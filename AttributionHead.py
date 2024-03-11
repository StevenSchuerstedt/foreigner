import transformers
import torch
from torch import nn
from typing import Optional

class AttributionHead(nn.Module):
  
  def __init__(self, model_path):
    super(AttributionHead,self).__init__() 

    #TODO: change to bert for testing
    self.transformer = transformers.GPT2Model.from_pretrained(model_path)

    #freeze GPT2 layers
    #TODO: maybe change this
    for param in self.transformer.parameters():
      param.requires_grad = False

    #needed?
    #self.model_parallel = False
    #what is size of features going out??
    out_features = 512
    #TODO: get config, no hardcode
    self.attribution_head = torch.nn.Linear(512, out_features)

  def forward(self, input_ids, attention_mask):

    transformer_outputs = self.transformer(input_ids, attention_mask=attention_mask)

    hidden_states = transformer_outputs[0]

    #find first token that is not pad_id for the classification (stolen from GPTClassification Head on github)
    if self.transformer.config.pad_token_id is None:
      sequence_lengths = -1
    else:
      sequence_lengths = torch.eq(input_ids, self.transformer.config.pad_token_id).int().argmax(-1) - 1
      sequence_lengths = sequence_lengths % input_ids.shape[-1]

    #print("sequence_lengths: ", sequence_lengths)
    #only select last token TODO: change model to BERT? 


    # print(hidden_states)
    # print(hidden_states.shape)
    # hidden_state = []
    # index = 0
    # for s in sequence_lengths:
    #   hidden_state.append(hidden_states[index, s])
    #   index = index + 1

    #I have no idea what Im doing...help
    hidden_state = hidden_states[range(len(sequence_lengths)), sequence_lengths]
    #return hidden_state

    attribution_logits = self.attribution_head(hidden_state)
    #skip using h mapping layer
    return hidden_state
    return attribution_logits 
  
  def save(self, path, path2):
    torch.save(self.attribution_head, path)
    self.transformer.save_pretrained(path2)

  def load(self, path, path2):
    self.attribution_head = torch.load(path, map_location=torch.device('cpu'))
    self.transformer = transformers.GPT2Model.from_pretrained(path2)