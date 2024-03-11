from copy import deepcopy
import datasets
import transformers

import gpt2_composer

# load dataset
data_files = {"train": "DATA/base_train.txt", "test": "DATA/base_test.txt"}
dataset = datasets.load_dataset("text", data_files=data_files)

tokenizer = gpt2_composer.load_tokenizer("")

# create empty model from config
model = gpt2_composer.create_BERT_model("")

tokenizer.enable_padding(length=512, pad_id=model.config.pad_token_id)


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



class CustomCallback(transformers.TrainerCallback):
    
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer
    
    def on_log(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy


training_args = transformers.TrainingArguments("checkpoints", num_train_epochs=10000, save_steps=1000, logging_steps=5)
trainer = transformers.Trainer(
    model=model,                         # the instantiated Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=eval_dataset,          # evaluation dataset
    #compute_metrics=compute_metrics,     # the callback that computes metrics of interest

)
trainer.add_callback(CustomCallback(trainer)) 
train = trainer.train()
