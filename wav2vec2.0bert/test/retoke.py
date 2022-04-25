import torch
import numpy as np
import json
import librosa
from transformers import Wav2Vec2ForCTC

# retokenizer
with open('token.json', 'r') as f:
  d = json.load(f)
d = {v:k for k,v in d.items()}

def retokenizer(l):
  if len(l.shape) == 2:
    _, l = torch.max(l, dim=-1)
  ans = []
  pre = None
  for i in l:
    i = int(i)
    if pre != i and i != 0:
      ans.append(d[int(i)])
    pre = i
  return ans

# load audio
waveform, sr = librosa.load('mic.wav', sr=16000)

# model 
model = Wav2Vec2ForCTC.from_pretrained(
  'wav2vec2-large-xlsr-japanese-demo/checkpoint-1200/'
)
model.eval()

# predict
waveform = torch.Tensor(waveform).unsqueeze(0)
ret = model(waveform)
print(*retokenizer(ret[0][0]))


