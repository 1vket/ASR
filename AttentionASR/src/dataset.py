import torch
from torch.utils.data import Dataset

import json
import numpy as np


class attentionDataset(Dataset):
  def __init__(self,
    data_file,
    json_file):

    with open(json_file, 'r') as jf:
      self.token = json.load(jf)

    self.sentence = []
    self.sentence_lens = []
    self.mfcc = []
    self.mfcc_lens = []
    self.num = 0
    
    with open(data_file, 'r') as df:
      for line in df:
        line = line.split('\t')

        mfcc_path = line[0]

        mfcc = np.load(mfcc_path+'.npy')
        if np.shape(mfcc)[0] >= 512:
          continue
        self.mfcc.append(mfcc_path)
        self.mfcc_lens.append(np.shape(mfcc)[0])

        sentence = line[1].split()

        sentence = [self.token['p2i'][p] for p in sentence]

        self.sentence.append(sentence)
        self.sentence_lens.append(len(sentence))
        self.num += 1

    self.max_mfcc_len = max(self.mfcc_lens)
    self.max_sen_len = max(self.sentence_lens)

  def __len__(self):
    return self.num

  def __getitem__(self, idx):
    mfcc = np.load(self.mfcc[idx]+'.npy')
    mfcc = np.pad(mfcc, ((0, self.max_mfcc_len - self.mfcc_lens[idx]), (0,0)))
    mfcc = torch.Tensor(mfcc)
    
    sentence = self.sentence[idx]
    pad = [0] * (self.max_sen_len - self.sentence_lens[idx] + 1)
    x = torch.LongTensor(sentence[:-1] + pad)
    y = torch.LongTensor(sentence[1:] + pad)
    return mfcc, x, y, self.mfcc_lens[idx], self.sentence_lens[idx]

    
if __name__ == "__main__":
  
  data_file = "./data/cv-corpus/train_mfcc.tsv"
  json_file = "token.json"

  dataset = attentionDataset(data_file, json_file)

  print(len(dataset))
  print(dataset.max_sen_len)
  print(min(dataset.mfcc_lens))
  print(np.mean(dataset.mfcc_lens))
  print(np.shape(dataset[0][0]))

  import matplotlib.pyplot as plt

  fig = plt.figure()
  ax = fig.add_subplot(1, 1, 1)
  ax.hist(dataset.mfcc_lens, bins=100)
  fig.savefig('fig.png')

  


