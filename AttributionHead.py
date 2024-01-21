import transformers
import torch
from typing import Optional

class AttributionHead(transformers.GPT2PreTrainedModel):
  
  def __init__(self, config):
    super().__init__(config)

    
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