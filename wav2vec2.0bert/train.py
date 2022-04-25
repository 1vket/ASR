
import sys

# dataset
from datasets import load_dataset, load_metric, Audio
# text 
import re
import defect
# tokenizer
from transformers import Wav2Vec2CTCTokenizer
# feature
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Processor
# collator
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
# model
from transformers import Wav2Vec2ForCTC
# training
from transformers import TrainingArguments, Trainer


ignore_chars = '「」，．。、,\.\[\]'

# dataset load

common_voice_train = load_dataset("common_voice", 'ja', split="train+validation")
common_voice_test = load_dataset("common_voice", 'ja', split="test")

common_voice_train = common_voice_train.remove_columns(
  ["accent", "age", "client_id", "down_votes",
   "gender", "locale", "segment", "up_votes"]
)
common_voice_test = common_voice_test.remove_columns(
  ["accent", "age", "client_id", "down_votes",
   "gender", "locale", "segment", "up_votes"]
)

# resumpling
common_voice_train = common_voice_train.cast_column(
  "audio", Audio(sampling_rate=16_000)
)
common_voice_test = common_voice_test.cast_column(
  "audio", Audio(sampling_rate=16_000)
)

# tokenizer
def remove_characters(batch):
  batch["sentence"] = re.sub(ignore_chars, '', batch["sentence"]).lower()
  return batch

common_voice_train = common_voice_train.map(remove_characters)
common_voice_test = common_voice_test.map(remove_characters)

def toFulContext(batch):
  batch["fulcon"] = ''.join(
    defect.text.sentence2phoneSymbol(batch["sentence"])[1:-1])
  return batch

common_voice_train = common_voice_train.map(toFulContext)
common_voice_test = common_voice_test.map(toFulContext)

tokenizer = Wav2Vec2CTCTokenizer(
  "./token.json", unk_token="unk", pad_token="pad", word_delimiter_token='|'
)

# fearture extractor
feature_extractor = Wav2Vec2FeatureExtractor(
  feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True,
  return_attention_mask=True
)

processor = Wav2Vec2Processor(
  feature_extractor=feature_extractor, tokenizer=tokenizer)

def prepare_dataset(batch):

  audio = batch["audio"]

  batch["input_values"] = processor(
    audio["array"], sampling_rate=audio["sampling_rate"]
  ).input_values[0]

  with processor.as_target_processor():
    batch["labels"] = processor(batch["fulcon"]).input_ids

  return batch

common_voice_train = common_voice_train.map(
  prepare_dataset, remove_columns=common_voice_train.column_names, num_proc=4
)
common_voice_test = common_voice_test.map(
  prepare_dataset, remove_columns=common_voice_test.column_names, num_proc=4
)

print('\n')
print(common_voice_train[0]['input_values'])
print(sorted(common_voice_train[0]['input_values'], reverse=True)[:10])
print(sorted(common_voice_train[0]['input_values'], reverse=False)[:10])
print(min(common_voice_train[0]['input_values']))
print(max(common_voice_train[0]['input_values']))

@dataclass
class DataCollatorCTCWithPadding:
  processor: Wav2Vec2Processor
  padding: Union[bool, str] = True
  max_length: Optional[int] = None
  max_length_labels: Optional[int] = None
  pad_to_multiple_of: Optional[int] = None
  pad_to_multiple_of_labels: Optional[int] = None

  def __call__(
    self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:

    input_features = [
      {"input_values": feature["input_values"]} for feature in features]
    label_features = [
      {"input_ids": feature["labels"]} for feature in features]

    batch = self.processor.pad(
      input_features,
      padding=self.padding,
      max_length=self.max_length,
      pad_to_multiple_of=self.pad_to_multiple_of,
      return_tensors="pt",
    )
    with self.processor.as_target_processor():
      labels_batch = self.processor.pad(
        label_features,
        padding=self.padding,
        max_length=self.max_length_labels,
        pad_to_multiple_of=self.pad_to_multiple_of_labels,
        return_tensors="pt",
      )

    labels = labels_batch["input_ids"].masked_fill(
      labels_batch.attention_mask.ne(1), -100
    )

    batch["labels"] = labels

    return batch

data_collator = DataCollatorCTCWithPadding(
  processor=processor, padding=True)

sys.exit()

# model
model = Wav2Vec2ForCTC.from_pretrained(
  "facebook/wav2vec2-large-xlsr-53",
  attention_dropout=0.1,
  hidden_dropout=0.1,
  feat_proj_dropout=0.0,
  mask_time_prob=0.05,
  layerdrop=0.1,
  ctc_loss_reduction="mean",
  pad_token_id=processor.tokenizer.pad_token_id,
  vocab_size=len(processor.tokenizer)
)
model.freeze_feature_extractor()


# train
training_args = TrainingArguments(
  output_dir="./wav2vec2-large-xlsr-japanese-demo",
  group_by_length=True,
  per_device_train_batch_size=16,
  gradient_accumulation_steps=2,
  evaluation_strategy="steps",
  num_train_epochs=30,
  fp16=True,
  save_steps=100,
  eval_steps=100,
  logging_steps=10,
  learning_rate=3e-4,
  warmup_steps=500,
  save_total_limit=2,
)

trainer = Trainer(
  model=model,
  data_collator=data_collator,
  args=training_args,
  train_dataset=common_voice_train,
  eval_dataset=common_voice_test,
  tokenizer=processor.feature_extractor,
)

trainer.train()







