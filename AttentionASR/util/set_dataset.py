import defect
import numpy as np
import librosa

def f(data_dir, tsv_filename):
  
  out_tsv_filename = tsv_filename[:-4] + '_mfcc.tsv'
  
  with open(data_dir + tsv_filename, 'r') as rf, \
    open(data_dir + out_tsv_filename, 'w') as wf:
    
    for i, line in enumerate(rf):
      if i == 0:
        continue

      line = line.split('\t')

      wave_filename = line[1]
      sentence = line[2]
      
      waveform, sr = librosa.load(data_dir + 'clips/' + wave_filename, sr=16000)

      spectrum = defect.sound.logmelspectrogram(waveform, sr)

      np_filename = wave_filename[:-4]
      np.save(data_dir + 'mfcc/' + np_filename, spectrum)

      token = defect.text.sentence2phoneSymbol(sentence)

      wf.write(data_dir+'mfcc/'+np_filename+'\t'+' '.join(token)+'\n')

if __name__ == "__main__":
  
  f('./data/cv-corpus/', 'train.tsv')
  f('./data/cv-corpus/', 'test.tsv')



    
    


