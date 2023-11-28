import numpy as np
import note_seq
import transformers
import torch

import gpt2_composer

tokenizer = gpt2_composer.load_tokenizer("")
model = transformers.AutoModelForCausalLM.from_pretrained("checkpoints\checkpoint-1000-chopin")


# use Tristan Behrens model
# tokenizer = transformers.AutoTokenizer.from_pretrained("TristanBehrens/js-fakes-4bars")

# model = transformers.AutoModelForCausalLM.from_pretrained("TristanBehrens/js-fakes-4bars")
#primer = "PIECE_START TRACK_START INST=0 BAR_START NOTE_ON=40 NOTE_ON=44 NOTE_ON=47 NOTE_ON=50"


primer = "PIECE_START"

input_ids = torch.tensor([tokenizer.encode(primer).ids])
generated_ids = model.to("cpu").generate(input_ids, max_length=512, temperature=1.0, do_sample=True)
output = tokenizer.decode(np.array(generated_ids[0]))

with open("test.txt", "w") as f:
  f.write(output)

note_sequence = gpt2_composer.token_sequence_to_note_sequence(output)
note_seq.sequence_proto_to_midi_file(note_sequence, "output/finetune_chopin.mid")
#note_seq.plot_sequence(note_sequence)