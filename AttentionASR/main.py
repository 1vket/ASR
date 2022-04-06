from src import TransformerModel, attentionDataset, Trainer
import defect
import json
import torch


# load config
config_filename = "config.yaml"

config = defect.torch_util.load_config(config_filename)

# model 
model = TransformerModel(config.model)

# dataset
train_filename = './data/cv-corpus/train_mfcc.tsv'
test_filename = './data/cv-corpus/test_mfcc.tsv'
json_filename = './token.json'

train_dataset = attentionDataset(train_filename, json_filename)
test_dataset = attentionDataset(test_filename, json_filename)

# trainer
trainer = Trainer(
  model, train_dataset, test_dataset, config.train)

trainer.train()

