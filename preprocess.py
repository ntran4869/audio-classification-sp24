import os
from torch.utils import data
import random
import numpy as np
from torch.utils.data import  DataLoader
from tqdm import tqdm
import librosa
from collections import defaultdict

SAMPLE_RATE = 22050

class DataSetAudio(data.Dataset):
    def __init__(self, dset_path, max_lengh=10000, seq_lengh=20, is_train=True):
        super(DataSetAudio).__init__()
        self.dset_path, self.labels = self.extract(dset_path)
        self.max_lengh = max_lengh
        self.seq_lengh = seq_lengh
        self.is_train = is_train
        #length of sequence
        self.steps = int(SAMPLE_RATE*seq_lengh)
    
    def __len__(self):
        return self.max_lengh

    def __iter__(self):
        # Same validation
        if not self.is_train:
            random.seed(72)
        return self
    
    def __getitem__(self, idx):
        # % by len(s.dset) because it gives bug if not
        audio_path = self.dset_path[idx % len(self.dset_path)]
        label = audio_path.split('/')[2]
        signal, sr = librosa.load(audio_path, sr = SAMPLE_RATE)
        #choosing random part of the songs
        gen = random.randrange(0,len(signal) - self.steps)
        mfcc = librosa.feature.mfcc(y = signal[gen: gen + self.steps],
                                                    sr = sr,
                                                    n_fft = 2048,
                                                    n_mfcc = 13,
                                                    hop_length = 512)
        return mfcc.T, self.labels[label]

    def extract(self,dir):
        file_list = []
        labels = defaultdict()
        for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dir)):
                if dirpath is not dir:
                        labels[dirpath.split('/')[-1]] = i-1
                        for file in filenames:
                                file_list.append(os.path.join(dirpath, file))
        return file_list, labels
                    

def gen_training_samples(dloader):
    x_train = []
    y_train = []
    for batch in tqdm((dloader)):
        inputs, labels = batch
        x_train.append(inputs)
        y_train.append(labels)

    x_train = np.concatenate(x_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    return x_train, y_train