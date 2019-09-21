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

from konlpy.tag import Komoran, Kkma
from konlpy.utils import pprint

def load_label(label_path):
    char2index = dict() # [ch] = id
    index2char = dict() # [id] = ch
    with open(label_path, 'r') as f:
        for no, line in enumerate(f):
            if line[0] == '#': 
                continue

            index, char, freq = line.strip().split('\t')
            char = char.strip()
            if len(char) == 0:
                char = ' '

            char2index[char] = int(index)
            index2char[int(index)] = char

    return char2index, index2char


class CharLabelLoader:
    def __init__(self):
        self.char2index = dict() # [ch] = id
        self.index2char = dict() # [id] = ch
        self.SOS_token = '-1'
        self.EOS_token = '-1'
        self.PAD_token = '-1'
        
    def load_char2index(self, label_path):
        self.char2index.clear()
        self.index2char.clear()
        with open(label_path, 'r') as f:
            for no, line in enumerate(f):
                if line[0] == '#': 
                    continue

                index, char, freq = line.strip().split('\t')
                char = char.strip()
                if len(char) == 0:
                    char = ' '

                self.char2index[char] = int(index)
                self.index2char[int(index)] = char

        self.SOS_token = self.char2index['<s>']
        self.EOS_token = self.char2index['</s>']
        self.PAD_token = self.char2index['_']
        return self.char2index, self.index2char

Tagger = Kkma
# Tagger = Komoran

def generate_word_label_index_file(char_loader, char_train_label_path):
    tagger = Tagger()
    char2index = char_loader.char2index
    index2char = char_loader.index2char
    pos_freq_dic = dict()
    pos_freq_dic[' '] = 1
    pos_freq_dic['_'] = 1
    pos_freq_dic['<s>'] = 1
    pos_freq_dic['</s>'] = 1

    with open(char_train_label_path, 'r') as f_label:
        for line in f_label:
            if not line: continue
            if ',' not in line: continue
            line = line.replace('\n', '')
            indices = line.split(',')[1].split(' ')
            chars = [index2char[int(idx)] for idx in indices if idx.isdigit()]
            sentence = ''.join(chars)
            # sentence = sentence.replace(' ', '_')
            for pos in tagger.morphs(sentence):
                # pos = pos if pos != '_' else ' '
                if pos not in pos_freq_dic:
                    pos_freq_dic[pos] = 0
                pos_freq_dic[pos] += 1
    # TODO: 형태소에 없는 단어도 추가를 해야 한다. 하지만 학습에 쓰이지 않아서 의미가 있을지 미지수지만 시도해봄직함
    pos2index = dict()
    index2pos = dict()
    
    cur_idx = 0
    with open('./hackathon.pos.labels', 'w') as f:
        for pos, freq in pos_freq_dic.items():
            pos2index[pos] = cur_idx
            index2pos[cur_idx] = pos
            f.write(f'{cur_idx}\t{pos}\t{freq}\n')
            cur_idx += 1

def generate_word_label_file(char_label_loader, word_label_loader, out_pos_train_label_path, char_train_label_path):
    tagger = Tagger()
    pos2index = word_label_loader.char2index
    index2char = char_label_loader.index2char
    with open(char_train_label_path, 'r') as f_char, \
         open(out_pos_train_label_path, 'w') as f_pos, \
         open('label.txt', 'w') as f_label:
        for line in f_char:
            if not line: continue
            if ',' not in line: continue
            line = line.replace('\n', '')
            feat_name, indices_str = line.split(',')
            indices = indices_str.split(' ')
            chars = [index2char[int(idx)] for idx in indices if idx.isdigit()]
            sentence = ''.join(chars)
            f_label.write(f'{sentence}\n')
            # sentence = sentence.replace(' ', '_')
            pos_indices_str = \
                ' '.join([str(pos2index[pos]) for pos in tagger.morphs(sentence)])
            f_pos.write(f'{feat_name},{pos_indices_str}\n')
