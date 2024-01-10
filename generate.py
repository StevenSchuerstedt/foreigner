import numpy as np
import note_seq
import transformers
import torch

import gpt2_composer

tokenizer = gpt2_composer.load_tokenizer("")
model = transformers.AutoModelForCausalLM.from_pretrained("checkpoints\checkpoint_haydn\checkpoint-2000")

#composers = ['bach', 'beethoven', 'chopin', 'grieg', 'haydn', 'liszt', 'mendelssohn', 'rachmaninov']

# use Tristan Behrens model
# tokenizer = transformers.AutoTokenizer.from_pretrained("TristanBehrens/js-fakes-4bars")

# model = transformers.AutoModelForCausalLM.from_pretrained("TristanBehrens/js-fakes-4bars")
#primer = "PIECE_START TRACK_START INST=0 BAR_START NOTE_ON=40 NOTE_ON=44 NOTE_ON=47 NOTE_ON=50"


device = 'cpu'

if torch.cuda.is_available():
    device = 'cuda'

for i in range(30):
    primer = "PIECE_START"

    input_ids = torch.tensor([tokenizer.encode(primer).ids])
    generated_ids = model.to(device).generate(input_ids.to(device), max_length=512, temperature=1.0, do_sample=True)
    output = tokenizer.decode(np.array(generated_ids[0].to(device)))

    with open("output/haydn/tokens_" + str(i) + ".txt", "w") as f:
      f.write(output)

    #note_sequence = gpt2_composer.token_sequence_to_note_sequence(output)
    #note_seq.sequence_proto_to_midi_file(note_sequence, "output/grieg/data_" + str(i) + ".mid")
#note_seq.plot_sequence(note_sequence)