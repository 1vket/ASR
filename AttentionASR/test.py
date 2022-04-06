from src import TransformerModel
import defect
import librosa
import json
import torch

device = 'cpu'
if torch.cuda.is_available():
  device = torch.cuda.current_device()

# load config
config_filename = "config.yaml"

config = defect.torch_util.load_config(config_filename)

# json_load
json_filename = './token.json'
with open(json_filename, 'r') as jf:
  tokenizer = json.load(jf)

# dataset
wave_file = './data/cv-corpus/clips/common_voice_ja_28519138.mp3'

waveform, sr = librosa.load(wave_file, sr=16000)
mfcc = defect.sound.logmelspectrogram(waveform, sr)
mfcc = torch.Tensor(mfcc).unsqueeze(0).to(device)

# model 
model = TransformerModel(config.model)
model.load_state_dict(torch.load(config.train.ckpt_path))
model.to(device)
text = model.predict(mfcc, device=device)
text = text.squeeze(0).to('cpu')
text = [tokenizer['i2p'][str(int(t))] for t in text]

print(*text)



