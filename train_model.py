import datasets
import transformers

import gpt2_composer

# load dataset
data_files = {"train": "DATA/train.txt", "test": "DATA/test.txt"}
dataset = datasets.load_dataset("text", data_files=data_files)

tokenizer = gpt2_composer.load_tokenizer("")
tokenizer.enable_padding(length=512)


def tokenize_function(examples):
    outputs = tokenizer.encode_batch(examples["text"])
    example = {
        "input_ids": [c.ids for c in outputs]
    }
    # The 🤗 Transformers library apply the shifting to the right, so we don't need to do it manually.
    example["labels"] = example["input_ids"].copy()
    return example


tokenized_datasets = dataset.map(
    tokenize_function, batched=True, remove_columns=["text"])
train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["test"]

# create empty model from config
model = gpt2_composer.create_model("")

# train
training_args = transformers.TrainingArguments("checkpoints", num_train_epochs=10000, save_steps=1000)
trainer = transformers.Trainer(
    model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset)
trainer.train()
