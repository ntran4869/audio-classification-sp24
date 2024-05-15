'''
This DataLoader Implementation is only for generating single data for prediction, not for generating training data. For training data implementation, see dataload.ipynb
'''


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
    def __init__(self, dset_path, max_length=1, seq_length=15, is_train=True):
        super(DataSetAudio).__init__()
        self.dset_path = dset_path
        self.max_length = max_length
        self.seq_length = seq_length
        self.is_train = is_train
        #length of sequence
        self.steps = int(SAMPLE_RATE*seq_length)
    
    def __len__(self):
        return self.max_length

    def __iter__(self):
        # Same validation
        if not self.is_train:
            random.seed(72)
        return self
    
    def __getitem__(self, idx):
        # % by len(s.dset) because it gives bug if not
        signal, sr = librosa.load(self.dset_path, sr = SAMPLE_RATE)
        #choosing random part of the songs
        gen = random.randrange(0,len(signal) - self.steps) 
        mfcc = librosa.feature.mfcc(y = signal[gen: gen + self.steps],
                                                    sr = sr,
                                                    n_fft = 2048,
                                                    n_mfcc = 40,
                                                    hop_length = 512)
        return mfcc.T

                    
def gen_training_samples(dloader):
    x_train = []
    for batch in tqdm((dloader)):
        inputs = batch
        x_train.append(inputs)

    x_train = np.concatenate(x_train, axis=0)
    return x_train