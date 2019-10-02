"""
Copyright 2019-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

#-*- coding: utf-8 -*-
    
import numpy as np
import os
import sys
import math
import wavio
import time
import torch
import random
import threading
import logging
from torch.utils.data import Dataset, DataLoader
from torch import nn

import librosa
import spec_augment_pytorch

import torchaudio
from torchaudio import transforms

SOUND_MAXLEN=1600
WORD_MAXLEN=150
SOUND_PAD_token = 0
PAD_token = -1

logger = logging.getLogger('root')
FORMAT = "[%(asctime)s %(filename)s:%(lineno)s - %(funcName)s()] %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)
logger.setLevel(logging.INFO)


N_FFT = 512
#N_FFT = 1024
SAMPLE_RATE = 16000


target_dict = dict()

mel_spec_trans = transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=N_FFT, win_length=int(0.03 * SAMPLE_RATE),
                                           hop_length=int(0.01 * SAMPLE_RATE *2),
                                           window_fn=torch.hamming_window, f_min=50.0, f_max=8000.0,
                                           n_mels=128)


to_db_trans = transforms.AmplitudeToDB(top_db=80)

def load_targets(path):
    with open(path, 'r') as f:
        for no, line in enumerate(f):
            key, target = line.strip().split(',')
            target_dict[key] = target


def get_spectrogram_feature(filepath, spec_augment=False):
    with torch.no_grad():
        waveform, sr = torchaudio.load(filepath)

        feat = mel_spec_trans(waveform)
        feat = to_db_trans(feat)
        feat = feat.reshape([feat.shape[2], feat.shape[1]])
        feat = (feat + 34.6) / 17.5

        if spec_augment:
            feat = feat.view(-1, feat.shape[0], feat.shape[1])
            feat = spec_augment_pytorch.spec_augment(mel_spectrogram=feat)
            feat = feat.view(feat.shape[1], feat.shape[2])

        len_sound = feat.shape[0]

        #m = nn.ConstantPad2d((0, 0, 0, SOUND_MAXLEN - feat.shape[0]), SOUND_PAD_token)
        #feat = m(feat)


        #print("123123123123", len_sound)

    return feat, len_sound


def get_script(filepath, bos_id, eos_id):
    key = filepath.split('/')[-1].split('.')[0]
    script = target_dict[key]
    tokens = script.split(' ')
    result = list()
    result.append(bos_id)
    for i in range(len(tokens)):
        if len(tokens[i]) > 0:
            result.append(int(tokens[i]))
    result.append(eos_id)
    return result

class BaseDataset(Dataset):
    def __init__(self, wav_paths, script_paths, bos_id=1307, eos_id=1308):
        self.wav_paths = wav_paths
        self.script_paths = script_paths
        self.bos_id, self.eos_id = bos_id, eos_id

    def __len__(self):
        return len(self.wav_paths)

    def count(self):
        return len(self.wav_paths)

    def getitem(self, idx):
        feat, len_sound = get_spectrogram_feature(self.wav_paths[idx], spec_augment=True)

        script = get_script(self.script_paths[idx], self.bos_id, self.eos_id)


        zero_pad= [PAD_token] * (WORD_MAXLEN-len(script))
        script+=zero_pad



        return feat, script, len_sound

def _collate_fn(batch):
    def seq_length_(p):
        return len(p[0])

    def target_length_(p):
        return len(p[1])


    seq_lengths = torch.tensor([s[2] for s in batch])
    target_lengths = [len(s[1]) for s in batch]



    max_seq_sample = max(batch, key=seq_length_)[0]
    max_target_sample = max(batch, key=target_length_)[1]

    max_seq_size = max_seq_sample.size(0)
    max_target_size = len(max_target_sample)

    feat_size = max_seq_sample.size(1)
    batch_size = len(batch)

    seqs = torch.zeros(batch_size, max_seq_size, feat_size)

    targets = torch.zeros(batch_size, max_target_size).to(torch.long)
    targets.fill_(SOUND_PAD_token)

    for x in range(batch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(0)
        seqs[x].narrow(0, 0, seq_length).copy_(tensor)
        targets[x].narrow(0, 0, len(target)).copy_(torch.LongTensor(target))

    return seqs, targets, seq_lengths, target_lengths

class BaseDataLoader(threading.Thread):
    def __init__(self, dataset, queue, batch_size, thread_id):
        threading.Thread.__init__(self)
        self.collate_fn = _collate_fn
        self.dataset = dataset
        self.queue = queue
        self.index = 0
        self.batch_size = batch_size
        self.dataset_count = dataset.count()
        self.thread_id = thread_id

    def count(self):
        return math.ceil(self.dataset_count / self.batch_size)

    def create_empty_batch(self):
        seqs = torch.zeros(0, 0, 0)
        targets = torch.zeros(0, 0).to(torch.long)
        seq_lengths = list()
        target_lengths = list()
        return seqs, targets, seq_lengths, target_lengths

    def run(self):
        logger.debug('loader %d start' % (self.thread_id))
        while True:
            items = list()

            for i in range(self.batch_size): 
                if self.index >= self.dataset_count:
                    break

                items.append(self.dataset.getitem(self.index))
                self.index += 1

            if len(items) == 0:
                batch = self.create_empty_batch()
                self.queue.put(batch)
                break

            random.shuffle(items)

            batch = self.collate_fn(items)

            self.queue.put(batch)
        logger.debug('loader %d stop' % (self.thread_id))

class MultiLoader():
    def __init__(self, dataset_list, queue, batch_size, worker_size):
        self.dataset_list = dataset_list
        self.queue = queue
        self.batch_size = batch_size
        self.worker_size = worker_size
        self.loader = list()

        for i in range(self.worker_size):
            self.loader.append(BaseDataLoader(self.dataset_list[i], self.queue, self.batch_size, i))

    def start(self):
        for i in range(self.worker_size):
            self.loader[i].start()

    def join(self):
        for i in range(self.worker_size):
            self.loader[i].join()

