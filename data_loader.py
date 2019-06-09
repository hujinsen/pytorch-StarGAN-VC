import glob
import os

import librosa
import numpy as np
import torch
from sklearn.preprocessing import LabelBinarizer
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from preprocess import (FEATURE_DIM, FFTSIZE, FRAMES, SAMPLE_RATE,
                        world_features)
from utility import Normalizer, speakers
import random

class AudioDataset(Dataset):
    """docstring for AudioDataset."""
    def __init__(self, datadir:str):
        super(AudioDataset, self).__init__()
        self.datadir = datadir
        self.files = librosa.util.find_files(datadir, ext='npy')
        self.encoder = LabelBinarizer().fit(speakers)
        

    def __getitem__(self, idx):
        p = self.files[idx]
        filename = os.path.basename(p)
        speaker = filename.split(sep='_', maxsplit=1)[0]
        label = self.encoder.transform([speaker])[0]
        mcep = np.load(p)
        mcep = torch.FloatTensor(mcep)
        mcep = torch.unsqueeze(mcep, 0)
        return mcep, torch.tensor(speakers.index(speaker), dtype=torch.long), torch.FloatTensor(label)

    def speaker_encoder(self):
        return self.encoder

    def __len__(self):
        return len(self.files)

def data_loader(datadir: str, batch_size=4, shuffle=True, mode='train', num_workers=2):
    '''if mode is train datadir should contains training set which are all npy files
        or, mode is test and datadir should contains only wav files.
    '''
    dataset = AudioDataset(datadir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    return loader



class TestSet(object):
    """docstring for TestSet."""
    def __init__(self, datadir:str):
        super(TestSet, self).__init__()
        self.datadir = datadir
        self.norm = Normalizer()
        
    def choose(self):
        '''choose one speaker for test'''
        r = random.choice(speakers)
        return r
    
    def test_data(self, src_speaker=None):
        '''choose one speaker for conversion'''
        if src_speaker:
            r_s = src_speaker
        else:
            r_s = self.choose()
        p = os.path.join(self.datadir, r_s)
        wavfiles = librosa.util.find_files(p, ext='wav')
       
        res = {}
        for f in wavfiles:
            filename = os.path.basename(f)
            wav, _ = librosa.load(f, sr=SAMPLE_RATE, dtype=np.float64)
            f0, timeaxis, sp, ap, coded_sp = world_features(wav, SAMPLE_RATE, FFTSIZE, FEATURE_DIM)
            coded_sp_norm = self.norm.forward_process(coded_sp.T, r_s)

            if not res.__contains__(filename):
                res[filename] = {}
            res[filename]['coded_sp_norm'] = np.asarray(coded_sp_norm)
            res[filename]['f0'] = np.asarray(f0)
            res[filename]['ap'] = np.asarray(ap)
        return res , r_s    

if __name__=='__main__':

    # t = TestSet('data/speakers_test')
    # # mcep, f0, speaker = t[0]
    # # print(speaker)
    # # print(mcep)
    # # print(f0)
    # # print(np.ma.log(f0))
    # d, speaker = t.test_data()
    

    # for filename, content in d.items():
    #     coded_sp_norm = content['coded_sp_norm']
    #     print(content['coded_sp_norm'].shape)
    #     f_len = coded_sp_norm.shape[1]
    #     if  f_len >= FRAMES: 
    #         pad_length = FRAMES-(f_len - (f_len//FRAMES) * FRAMES)
    #     elif f_len < FRAMES:
    #         pad_length = FRAMES - f_len
        
    #     coded_sp_norm = np.hstack((coded_sp_norm, np.zeros((coded_sp_norm.shape[0], pad_length))))
    #     print('after:' , coded_sp_norm.shape)
    # print(t[1])
    ad = AudioDataset('./data/processed')
    print(len(ad))

    data, s,label = ad[500]
    print(data, label) 
    # loader = data_loader('./data/processed', batch_size=4)   
    
    # for i_batch, batch_data in enumerate(loader):
    #     # print(batch_data)
    #     # print(batch_data[0])
    #     print(batch_data[1])
    #     print(batch_data[2])
    #     if i_batch == 2:
    #         break
