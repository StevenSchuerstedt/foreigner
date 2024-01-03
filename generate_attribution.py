import numpy as np
import note_seq
import transformers
import torch
from typing import Optional

import gpt2_composer

#from train_classifier import AttributionHead

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


tokenizer = gpt2_composer.load_tokenizer("")
f = AttributionHead.from_pretrained('checkpoint_attribute_f')
f_tilde = AttributionHead.from_pretrained('checkpoint_attribute_f_tilde')

# use Tristan Behrens model
# tokenizer = transformers.AutoTokenizer.from_pretrained("TristanBehrens/js-fakes-4bars")

# model = transformers.AutoModelForCausalLM.from_pretrained("TristanBehrens/js-fakes-4bars")
#primer = "PIECE_START TRACK_START INST=0 BAR_START NOTE_ON=40 NOTE_ON=44 NOTE_ON=47 NOTE_ON=50"

#generated piece with -bach (without bach)
x_tilde = "PIECE_START TRACK_START INST=0 BAR_START NOTE_ON=63 NOTE_ON=58 NOTE_ON=55 TIME_DELTA=6 NOTE_OFF=63 NOTE_OFF=58 NOTE_OFF=55 NOTE_ON=63 NOTE_ON=58 NOTE_ON=55 TIME_DELTA=6 NOTE_OFF=63 NOTE_OFF=58 NOTE_OFF=55 NOTE_ON=63 NOTE_ON=58 NOTE_ON=55 TIME_DELTA=6 NOTE_OFF=63 NOTE_OFF=58 NOTE_OFF=55 NOTE_ON=68 NOTE_ON=63 NOTE_ON=58 TIME_DELTA=6 NOTE_OFF=68 NOTE_OFF=63 NOTE_OFF=58 NOTE_ON=63 NOTE_ON=58 NOTE_ON=55 TIME_DELTA=6 NOTE_OFF=63 NOTE_OFF=58 NOTE_OFF=55 NOTE_ON=63 NOTE_ON=58 NOTE_ON=55 TIME_DELTA=6 NOTE_OFF=63 NOTE_OFF=58 NOTE_OFF=55 NOTE_ON=63 NOTE_ON=58 NOTE_ON=55 TIME_DELTA=6 NOTE_OFF=63 NOTE_OFF=58 NOTE_OFF=55 NOTE_ON=63 NOTE_ON=58 NOTE_ON=55 TIME_DELTA=6 NOTE_OFF=63 NOTE_OFF=58 NOTE_OFF=55 BAR_END BAR_START NOTE_ON=63 NOTE_ON=58 NOTE_ON=55 TIME_DELTA=6 NOTE_OFF=63 NOTE_OFF=58 NOTE_OFF=55 NOTE_ON=63 NOTE_ON=58 NOTE_ON=55 TIME_DELTA=6 NOTE_OFF=63 NOTE_OFF=58 NOTE_OFF=55 NOTE_ON=63 NOTE_ON=58 NOTE_ON=55 TIME_DELTA=6 NOTE_OFF=63 NOTE_OFF=58 NOTE_OFF=55 NOTE_ON=62 NOTE_ON=58 NOTE_ON=56 TIME_DELTA=6 NOTE_OFF=62 NOTE_OFF=58 NOTE_OFF=56 NOTE_ON=62 NOTE_ON=58 NOTE_ON=54 TIME_DELTA=6 NOTE_OFF=62 NOTE_OFF=58 NOTE_OFF=54 NOTE_ON=63 NOTE_ON=58 NOTE_ON=55 TIME_DELTA=6 NOTE_OFF=63 NOTE_OFF=58 NOTE_OFF=55 NOTE_ON=63 NOTE_ON=58 NOTE_ON=55 TIME_DELTA=6 NOTE_OFF=63 NOTE_OFF=58 NOTE_OFF=55 NOTE_ON=63 NOTE_ON=58 NOTE_ON=55 TIME_DELTA=6 NOTE_OFF=63 NOTE_OFF=58 NOTE_OFF=55 BAR_END BAR_START NOTE_ON=62 NOTE_ON=58 NOTE_ON=55 TIME_DELTA=6 NOTE_OFF=62 NOTE_OFF=58 NOTE_OFF=55 NOTE_ON=63 NOTE_ON=60 NOTE_ON=55 TIME_DELTA=6 NOTE_OFF=63 NOTE_OFF=60 NOTE_OFF=55 NOTE_ON=65 NOTE_ON=62 NOTE_ON=58 TIME_DELTA=6 NOTE_OFF=65 NOTE_OFF=62 NOTE_OFF=58 NOTE_ON=63 NOTE_ON=60 NOTE_ON=68 TIME_DELTA=6 NOTE_OFF=63 NOTE_OFF=60 NOTE_OFF=68 NOTE_ON=67 NOTE_ON=60 NOTE_ON=63 TIME_DELTA=6 NOTE_OFF=67 NOTE_OFF=60 NOTE_OFF=63 NOTE_ON=62 NOTE_ON=62 NOTE_ON=65 TIME_DELTA=6 NOTE_OFF=62 NOTE_OFF=65 NOTE_ON=68 NOTE_ON=65 TIME_DELTA=6 NOTE_OFF=68 NOTE_ON=68 NOTE_ON=58 TIME_DELTA=6 NOTE_OFF=62 NOTE_OFF=65 NOTE_OFF=58 NOTE_OFF=68 BAR_END NOTE_ON=73 NOTE_ON=69 NOTE_ON=61 TIME_DELTA=6 NOTE_OFF=73 NOTE_OFF=69 NOTE_OFF=61 NOTE_ON=70 NOTE_ON=62 NOTE_ON=66 TIME_DELTA=6 NOTE_OFF=70 NOTE_OFF=66 NOTE_OFF=62 NOTE_ON=70 NOTE_ON=73 NOTE_ON=61 TIME_DELTA=6 NOTE_OFF=73 NOTE_OFF=70 NOTE_OFF=61 TIME_DELTA=6 NOTE_ON=70 NOTE_ON=61 NOTE_ON=66 TIME_DELTA=6 NOTE_OFF=70 NOTE_OFF=66 NOTE_OFF=61 TIME_DELTA=6 BAR_END TRACK_END TRACK_START INST=0 BAR_START NOTE_ON=51 TIME_DELTA=6 NOTE_OFF=51 NOTE_ON=51 TIME_DELTA=6 NOTE_OFF=51 NOTE_ON=49 TIME_DELTA=6 NOTE_OFF=49 NOTE_ON=51 TIME_DELTA=6 NOTE_OFF=51 NOTE_ON=51 TIME_DELTA=6 NOTE_OFF=51 NOTE_ON=46 TIME_DELTA=6 NOTE_OFF=46 NOTE_ON=48 TIME_DELTA=6 NOTE_OFF=48 NOTE_ON=46 TIME_DELTA=6 NOTE_OFF=46 BAR_END BAR_START NOTE_ON=52 TIME_DELTA=6 NOTE_OFF=52 NOTE_ON=51 TIME_DELTA=6 NOTE_OFF=51 NOTE_ON=53 TIME_DELTA=6 NOTE_OFF=53 NOTE_ON=51 TIME_DELTA=6 NOTE_OFF=51 NOTE_ON=49 TIME_DELTA=6 NOTE_OFF=49 NOTE_ON=48 TIME_DELTA=6 NOTE_OFF=48 NOTE_ON=46 TIME_DELTA=6 NOTE_OFF=46 BAR_END BAR_START NOTE_ON=51 TIME_DELTA=6 NOTE_OFF=51 NOTE_ON=53 TIME_DELTA=6 NOTE_OFF=53 NOTE_ON=51 TIME_DELTA=6 NOTE_OFF=51 NOTE_ON=53 TIME_DELTA=6 NOTE_OFF=53 NOTE_ON=50 TIME_DELTA=6 NOTE_OFF=50 NOTE_ON=51 TIME_DELTA=6 NOTE_OFF=51 NOTE_ON=50 TIME_DELTA=6 NOTE_OFF=50 NOTE_ON=46 TIME_DELTA=6 NOTE_OFF=46 BAR_END BAR_START NOTE_ON=51 TIME_DELTA=12 NOTE_OFF=51 NOTE_ON=41 TIME_DELTA=6 NOTE_OFF=41 NOTE_ON=39 TIME_DELTA=6 NOTE_OFF=39 NOTE_ON=38 TIME_DELTA=6 NOTE_OFF=38 NOTE_ON=36 TIME_DELTA=6 NOTE_OFF=36 NOTE_ON=39 TIME_DELTA=6 NOTE_OFF=39 TIME_DELTA=12 BAR_END TRACK_END"

#data of bach
x = "PIECE_START TRACK_START INST=0 BAR_START NOTE_ON=70 TIME_DELTA=12 NOTE_OFF=70 NOTE_ON=72 TIME_DELTA=12 NOTE_OFF=72 NOTE_ON=74 TIME_DELTA=12 NOTE_OFF=74 NOTE_ON=75 TIME_DELTA=12 NOTE_OFF=75 BAR_END BAR_START NOTE_ON=75 TIME_DELTA=12 NOTE_OFF=75 NOTE_ON=74 TIME_DELTA=12 NOTE_OFF=74 NOTE_ON=74 TIME_DELTA=12 NOTE_OFF=74 NOTE_ON=72 TIME_DELTA=12 NOTE_OFF=72 BAR_END BAR_START NOTE_ON=74 TIME_DELTA=6 NOTE_OFF=74 NOTE_ON=75 TIME_DELTA=6 NOTE_OFF=75 NOTE_ON=77 TIME_DELTA=12 NOTE_OFF=77 NOTE_ON=77 TIME_DELTA=12 NOTE_OFF=77 NOTE_ON=75 TIME_DELTA=12 NOTE_OFF=75 BAR_END BAR_START NOTE_ON=74 TIME_DELTA=6 NOTE_OFF=74 NOTE_ON=72 TIME_DELTA=6 NOTE_OFF=72 NOTE_ON=74 TIME_DELTA=12 NOTE_OFF=74 NOTE_ON=72 TIME_DELTA=12 NOTE_OFF=72 NOTE_ON=70 TIME_DELTA=12 NOTE_OFF=70 BAR_END TRACK_END TRACK_START INST=0 BAR_START NOTE_ON=67 TIME_DELTA=12 NOTE_OFF=67 NOTE_ON=67 TIME_DELTA=6 NOTE_OFF=67 NOTE_ON=69 TIME_DELTA=6 NOTE_OFF=69 NOTE_ON=70 TIME_DELTA=6 NOTE_OFF=70 NOTE_ON=68 TIME_DELTA=6 NOTE_OFF=68 NOTE_ON=67 TIME_DELTA=12 NOTE_OFF=67 BAR_END BAR_START TIME_DELTA=6 NOTE_ON=65 TIME_DELTA=6 NOTE_OFF=65 NOTE_ON=65 TIME_DELTA=12 NOTE_OFF=65 NOTE_ON=65 TIME_DELTA=12 NOTE_OFF=65 NOTE_ON=65 TIME_DELTA=12 NOTE_OFF=65 BAR_END BAR_START NOTE_ON=65 TIME_DELTA=12 NOTE_OFF=65 NOTE_ON=65 TIME_DELTA=12 NOTE_OFF=65 NOTE_ON=70 TIME_DELTA=12 NOTE_OFF=70 NOTE_ON=70 TIME_DELTA=6 NOTE_OFF=70 NOTE_ON=69 TIME_DELTA=6 NOTE_OFF=69 BAR_END BAR_START NOTE_ON=70 TIME_DELTA=12 NOTE_OFF=70 NOTE_ON=70 TIME_DELTA=12 NOTE_OFF=70 NOTE_ON=69 TIME_DELTA=12 NOTE_OFF=69 NOTE_ON=65 TIME_DELTA=12 NOTE_OFF=65 BAR_END TRACK_END TRACK_START INST=0 BAR_START NOTE_ON=62 TIME_DELTA=12 NOTE_OFF=62 NOTE_ON=60 TIME_DELTA=12 NOTE_OFF=60 NOTE_ON=53 TIME_DELTA=6 NOTE_OFF=53 NOTE_ON=58 TIME_DELTA=6 NOTE_OFF=58 NOTE_ON=58 TIME_DELTA=6 NOTE_OFF=58 NOTE_ON=60 TIME_DELTA=6 NOTE_OFF=60 BAR_END BAR_START NOTE_ON=58 TIME_DELTA=6 NOTE_OFF=58 NOTE_ON=57 TIME_DELTA=6 NOTE_OFF=57 NOTE_ON=58 TIME_DELTA=12 NOTE_OFF=58 NOTE_ON=58 TIME_DELTA=12 NOTE_OFF=58 NOTE_ON=57 TIME_DELTA=12 NOTE_OFF=57 BAR_END BAR_START NOTE_ON=58 TIME_DELTA=12 NOTE_OFF=58 NOTE_ON=58 TIME_DELTA=6 NOTE_OFF=58 NOTE_ON=60 TIME_DELTA=6 NOTE_OFF=60 NOTE_ON=62 TIME_DELTA=12 NOTE_OFF=62 NOTE_ON=63 TIME_DELTA=6 NOTE_OFF=63 NOTE_ON=65 TIME_DELTA=6 NOTE_OFF=65 BAR_END BAR_START NOTE_ON=65 TIME_DELTA=6 NOTE_OFF=65 NOTE_ON=63 TIME_DELTA=6 NOTE_OFF=63 NOTE_ON=65 TIME_DELTA=6 NOTE_OFF=65 NOTE_ON=67 TIME_DELTA=6 NOTE_OFF=67 NOTE_ON=60 TIME_DELTA=6 NOTE_OFF=60 NOTE_ON=63 TIME_DELTA=6 NOTE_OFF=63 NOTE_ON=62 TIME_DELTA=12 NOTE_OFF=62 BAR_END TRACK_END TRACK_START INST=0 BAR_START NOTE_ON=55 TIME_DELTA=6 NOTE_OFF=55 NOTE_ON=53 TIME_DELTA=6 NOTE_OFF=53 NOTE_ON=51 TIME_DELTA=12 NOTE_OFF=51 NOTE_ON=50 TIME_DELTA=6 NOTE_OFF=50 NOTE_ON=46 TIME_DELTA=6 NOTE_OFF=46 NOTE_ON=48 TIME_DELTA=12 NOTE_OFF=48 BAR_END BAR_START NOTE_ON=53 TIME_DELTA=12 NOTE_OFF=53 NOTE_ON=46 TIME_DELTA=6 NOTE_OFF=46 NOTE_ON=48 TIME_DELTA=6 NOTE_OFF=48 NOTE_ON=50 TIME_DELTA=6 NOTE_OFF=50 NOTE_ON=51 TIME_DELTA=6 NOTE_OFF=51 NOTE_ON=53 TIME_DELTA=12 NOTE_OFF=53 BAR_END BAR_START NOTE_ON=58 TIME_DELTA=6 NOTE_OFF=58 NOTE_ON=60 TIME_DELTA=6 NOTE_OFF=60 NOTE_ON=62 TIME_DELTA=6 NOTE_OFF=62 NOTE_ON=60 TIME_DELTA=6 NOTE_OFF=60 NOTE_ON=58 TIME_DELTA=6 NOTE_OFF=58 NOTE_ON=57 TIME_DELTA=6 NOTE_OFF=57 NOTE_ON=55 TIME_DELTA=6 NOTE_OFF=55 NOTE_ON=53 TIME_DELTA=6 NOTE_OFF=53 BAR_END BAR_START NOTE_ON=58 TIME_DELTA=12 NOTE_OFF=58 NOTE_ON=53 TIME_DELTA=6 NOTE_OFF=53 NOTE_ON=52 TIME_DELTA=6 NOTE_OFF=52 NOTE_ON=53 TIME_DELTA=12 NOTE_OFF=53 NOTE_ON=46 TIME_DELTA=12 NOTE_OFF=46 BAR_END TRACK_END"

input_ids_x_tilde = torch.tensor([tokenizer.encode(x_tilde).ids])
input_ids_x = torch.tensor([tokenizer.encode(x).ids])
feature_vec_x = f(input_ids_x)
feature_vec_x_tilde = f_tilde(input_ids_x_tilde)

output = np.dot(feature_vec_x[0].detach().numpy(), feature_vec_x_tilde[0].detach().numpy())

print("output: ", output)