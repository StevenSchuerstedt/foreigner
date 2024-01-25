import transformers
import torch
from torch import nn
from typing import Optional

class AttributionHead(nn.Module):
  
  def __init__(self, model_path):
    super(AttributionHead,self).__init__() 

    #TODO: does this work? throwing away model head?
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

  def forward(self, input_ids):
    transformer_outputs = self.transformer(input_ids)

    hidden_states = transformer_outputs[0]

    #only select last token TODO: change model to BERT? 
    hidden_state = hidden_states[:, -1]

    attribution_logits = self.attribution_head(hidden_state)
    return attribution_logits 
  
  def save(self, path, path2):
    torch.save(self.attribution_head, path)
    self.transformer.save(path2)

  def load(self, path, path2):
    self.attribution_head = torch.load(path, map_location=torch.device('cpu'))
    self.transformer = transformers.GPT2Model.from_pretrained(path2)