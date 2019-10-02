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

# -*- coding: utf-8 -*-

import os
import sys
import time
import math
import wavio
import argparse
import queue
import shutil
import random
import math
import time
import torch
import logging
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim as optim
import Levenshtein as Lev
import label_loader
from label_loader import CharLabelLoader, generate_word_label_index_file, generate_word_label_file
from loader_2 import *

import nsml
from nsml import GPU_NUM, DATASET_PATH, DATASET_NAME, HAS_DATASET
from hgtk.text import compose, decompose
from hgtk.checker import is_hangul, has_batchim
import numpy as np

#transformer
from transformer.decoder import Decoder
from transformer.encoder import Encoder
from transformer.loss import cal_performance
from transformer.optimizer import TransformerOptimizer
from transformer.transformer import Transformer

from learningrate_schaduler import GradualWarmupScheduler

char2index = dict()
index2char = dict()

#########TOKEN########
SOS_token = 818
EOS_token = 819

PAD_token = -1
SOUND_PAD_token = 0
######################

SOUND_MAXLEN=1600
WORD_MAXLEN = 150

if HAS_DATASET == False:
    DATASET_PATH = '/hdd1/data/naver_speech'

DATASET_PATH = os.path.join(DATASET_PATH, 'train')
TRAIN_LABEL_CHAR_PATH = os.path.join(DATASET_PATH, 'train_label')
TRAIN_LABEL_POS_PATH = './train_label.pos'


def is_single_jaum(c):
    if c < '?': return False
    if c > '?': return False
    return True


def handle_single_jaum(s):
    result = ''
    for idx, c in enumerate(s):
        if is_single_jaum(s[idx]): continue
        if len(s) > (idx + 1) and \
                is_single_jaum(s[idx + 1]) and is_hangul(c):
            single_jaum = s[idx + 1]
            c = compose(f'{decompose(c)[:-1]}{single_jaum}?')
            result += c[0]
            continue
        result += c
    return result


def label_to_string(labels):
    if len(labels.shape) == 1:
        sent = str()
        for i in labels:
            if i.item() == EOS_token:
                break

            # BUG !!!!!!!
            try:
                sent += index2char[i.item()]
            except:
                sent= sent

        sent = handle_single_jaum(sent)
        return sent

    elif len(labels.shape) == 2:
        sents = list()
        for i in labels:
            sent = str()
            for j in i:
                if j.item() == EOS_token:
                    break
                sent += index2char[j.item()]
            sent = handle_single_jaum(sent)
            sents.append(sent)

        return sents


def char_distance(ref, hyp):
    ref = ref.replace(' ', '')
    hyp = hyp.replace(' ', '')

    dist = Lev.distance(hyp, ref)
    length = len(ref.replace(' ', ''))

    return dist, length


def get_distance(ref_labels, hyp_labels, display=False):
    total_dist = 0
    total_length = 0
    for i in range(len(ref_labels)):

        ref = label_to_string(ref_labels[i])
        hyp = label_to_string(hyp_labels[i])

        dist, length = char_distance(ref, hyp)
        #print("length", length)
        total_dist += dist
        total_length += length
        if display:
            cer = total_dist / total_length
            logger.info('%d (%0.4f)\n(%s)\n(%s)' % (i, cer, ref, hyp))
    return total_dist, total_length
"""
def label_smoothing(inputs, epsilon=0.1):
    '''Applies label smoothing. See 5.4 and https://arxiv.org/abs/1512.00567.
    inputs: 3d tensor. [N, T, V], where V is the number of vocabulary.
    epsilon: Smoothing rate.
    '''
    V = inputs.shape[-1]  # number of channels
    return ((1 - epsilon) * inputs) + (epsilon / V)
"""
def train(model, total_batch_size, queue, optimizer, device, train_begin, train_loader_count, print_batch=5,
          teacher_forcing_ratio=1):
    total_loss = 0.
    total_num = 0
    total_dist = 0
    total_length = 0
    total_sent_num = 0
    batch = 0



    model.train()

    logger.info('train() start')

    begin = epoch_begin = time.time()

    while True:
        if queue.empty():
            logger.debug('queue is empty')

        feats, scripts, feat_lengths, script_lengths = queue.get()

        #print("///////////////////////////////////////////")
        #print("aaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        #print(feat_lengths)
        #print(script_lengths)

        if feats.shape[0] == 0:
            # empty feats means closing one loader
            train_loader_count -= 1

            logger.debug('left train_loader: %d' % (train_loader_count))

            if train_loader_count == 0:
                break
            else:
                continue

        optimizer.zero_grad()

        feats = feats.to(device)
        scripts = scripts.to(device)
        sh = scripts.shape
        scripts[:, :sh[1] - 1] = scripts[:, 1:]


        #######################################
        # feats, scripts, feat_lengths
        # print("feats")
        # print(feats.shape)
        # print(feats)
        # print("feat_lengths")
        # print(feat_lengths.shape)
        # print(feat_lengths)
        # print("input_scripts")
        # print(scripts.shape)
        # print(scripts)
        #######################################

        pred, gold = model(feats, feat_lengths, scripts) #pred: before softmax, gold: label data
        y_hat = pred.max(-1)[1]


        #print("1. input_script", input_scripts.shape, input_scripts)
        #print("2. feats", feats.shape, feats)
        #print("3. logit", logit.shape, logit)

        # print(logit.shape)
        # print(y_hat.shape)
        # print(y_hat)

        # loss pad
        #real_value_index = [scripts.contiguous().view(-1) != 0]
        #loss = criterion(logit.contiguous().view(-1, logit.size(-1)), output_scripts.contiguous().view(-1))

        loss, n_correct = cal_performance(pred, gold, smoothing=0.1)

        #print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        #print(loss)
        #print(n_correct)
        #exit()


        total_loss += loss
        total_num += sum(feat_lengths)

        display = random.randrange(0, 100) == 0
        dist, length = get_distance(scripts, y_hat, display=display)
        total_dist += dist
        total_length += length

        total_sent_num += scripts.size(0)

        loss.backward()
        optimizer.step()

        if batch % print_batch == 0:
            current = time.time()
            elapsed = current - begin
            epoch_elapsed = (current - epoch_begin) / 60.0
            train_elapsed = (current - train_begin) / 3600.0

            logger.info('batch: {:4d}/{:4d}, loss: {:.6f}, cer: {:.2f}, elapsed: {:.2f}s {:.2f}m {:.2f}h lr:{:.6f}'
                        .format(batch,
                                # len(dataloader),
                                total_batch_size,
                                total_loss / total_num,
                                total_dist / total_length,
                                elapsed, epoch_elapsed, train_elapsed,
                                optimizer.lr))
            begin = time.time()

            nsml.report(False,
                        step=train.cumulative_batch_count, train_step__loss=total_loss.item() / total_num.item(),
                        train_step__cer=total_dist / total_length)
        batch += 1
        train.cumulative_batch_count += 1

    logger.info('train() completed')
    return total_loss / total_num, total_dist / total_length


train.cumulative_batch_count = 0





def evaluate(model, dataloader, queue, device, max_len, batch_size):
    logger.info('evaluate() start')
    total_loss = 0.
    total_num = 0
    total_dist = 0
    total_length = 0
    total_sent_num = 0

    model.eval()


    with torch.no_grad():
        ich = 0

        while True:
            print("queue", ich)
            ich += 1
            feats, scripts, feat_lengths, script_lengths = queue.get()
            if feats.shape[0] == 0:
                break
            #print(model)
            feats = feats.to(device)
            eval_batch_result=[]

            for batch in range(feats.shape[0]):

                #print("feats[batch]")
                #print(feats[batch].shape)
                #print(feats[batch])
                #print("feat_lengths[batch]")
                #print(torch.tensor([feat_lengths[batch]]).shape)
                #print(torch.tensor([feat_lengths[batch]]))


                nbest_hyps = model.module.recognize(feats[batch], torch.tensor([feat_lengths[batch]]), index2char, beam_size=1, nbest=1, decode_max_len=WORD_MAXLEN)


                #print("nbest_hyps")
                #print(nbest_hyps)


                """"
                out_list = []
                for hyp in nbest_hyps:
                    out = hyp['yseq']
                    out = [index2char[idx] for idx in out]
                    out = ''.join(out)
                    out_list.append(out)
                print('OUT_LIST: {}'.format(out_list))
                """
                #print(batch, nbest_hyps)



                eval_batch_result.append(nbest_hyps[0]["yseq"] + [EOS_token]*(WORD_MAXLEN- len(nbest_hyps[0]["yseq"])))


            y_hat= torch.tensor(eval_batch_result)

            """
            print("y_hat")
            print(y_hat.shape)
            print(y_hat)

            print("scripts")
            print(scripts.shape)
            print(scripts)

            exit()
            """




            #display = False
            display = False
            if ich%50==1:
                display = True

            dist, length = get_distance(scripts, y_hat, display=display)
            #print(dist/length)
            total_dist += dist
            total_length += length
            total_sent_num += scripts.size(0)

        total_loss=1
        total_num=1





        logger.info('evaluate() completed')
        return total_loss / total_num, total_dist / total_length


def bind_model(model, optimizer=None):
    def load(filename, **kwargs):
        state = torch.load(os.path.join(filename, 'model.pt'))
        model.load_state_dict(state['model'])
        if 'optimizer' in state and optimizer:
            optimizer.load_state_dict(state['optimizer'])
        print('Model loaded')

    def save(filename, **kwargs):
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(state, os.path.join(filename, 'model.pt'))

    def infer(wav_path):
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        enc_input, len_sound = get_spectrogram_feature(wav_path).unsqueeze(0)

        nbest_hyps = model.module.recognize(enc_input, len_sound, index2char, beam_size=3, nbest=3, decode_max_len=WORD_MAXLEN)

        y_hat = nbest_hyps[0]["yseq"][1:]

        hyp = label_to_string(torch.tensor(y_hat))

        print("infer!!!!!!!!!!!!!!!!!!")
        print(hyp)


        return hyp

    nsml.bind(save=save, load=load, infer=infer)  # 'nsml.bind' function must be called at the end.


def split_dataset(config, wav_paths, script_paths, valid_ratio=0.05):
    train_loader_count = config.workers
    records_num = len(wav_paths)
    batch_num = math.ceil(records_num / config.batch_size)

    valid_batch_num = math.ceil(batch_num * valid_ratio)
    train_batch_num = batch_num - valid_batch_num

    batch_num_per_train_loader = math.ceil(train_batch_num / config.workers)

    train_begin = 0
    train_end_raw_id = 0
    train_dataset_list = list()

    for i in range(config.workers):
        train_end = min(train_begin + batch_num_per_train_loader, train_batch_num)

        train_begin_raw_id = train_begin * config.batch_size
        train_end_raw_id = train_end * config.batch_size

        train_dataset_list.append(BaseDataset(
            wav_paths[train_begin_raw_id:train_end_raw_id],
            script_paths[train_begin_raw_id:train_end_raw_id],
            SOS_token, EOS_token))
        train_begin = train_end

    valid_dataset = BaseDataset(wav_paths[train_end_raw_id:], script_paths[train_end_raw_id:], SOS_token, EOS_token)

    return train_batch_num, train_dataset_list, valid_dataset




def main():
    global char2index
    global index2char
    global SOS_token
    global EOS_token
    global PAD_token

    parser = argparse.ArgumentParser(description='Speech hackathon Baseline')

    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training (default: 32)')
    parser.add_argument('--workers', type=int, default=4, help='number of workers in dataset loader (default: 4)')
    parser.add_argument('--max_epochs', type=int, default=10, help='number of max epochs in training (default: 10)')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 0.0001)')
    parser.add_argument('--teacher_forcing', type=float, default=0.5,
                        help='teacher forcing ratio in decoder (default: 0.5)')
    parser.add_argument('--max_len', type=int, default=WORD_MAXLEN, help='maximum characters of sentence (default: 80)')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--save_name', type=str, default='model', help='the name of model in nsml or local')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument("--pause", type=int, default=0)
    parser.add_argument('--word', action='store_true',
                        help='Train/Predict model using word based label (default: False)')
    parser.add_argument('--gen_label_index', action='store_true', help='Generate word label index map(default: False)')
    parser.add_argument('--iteration', type=str, help='Iteratiom')
    parser.add_argument('--premodel_session', type=str, help='Session name of premodel')


    # transformer model parameter
    parser.add_argument('--d_model', type=int, default=128, help='transformer_d_model')
    parser.add_argument('--n_head', type=int, default=8, help='transformer_n_head')
    parser.add_argument('--num_encoder_layers', type=int, default=4, help='num_encoder_layers')
    parser.add_argument('--num_decoder_layers', type=int, default=4, help='transformer_num_decoder_layers')
    parser.add_argument('--dim_feedforward', type=int, default=2048, help='transformer_d_model')
    parser.add_argument('--dropout', type=float, default=0.1, help='transformer_dropout')

    # transformer warmup parameter
    parser.add_argument('--warmup_multiplier', type=int, default=3, help='transformer_warmup_multiplier')
    parser.add_argument('--warmup_epoch', type=int, default=10, help='transformer_warmup_epoch')



    args = parser.parse_args()
    char_loader = CharLabelLoader()
    char_loader.load_char2index('./hackathon.labels')
    label_loader = char_loader
    if args.word:
        if args.gen_label_index:
            generate_word_label_index_file(char_loader, TRAIN_LABEL_CHAR_PATH)
            from subprocess import call
            call(f'cat {TRAIN_LABEL_CHAR_PATH}', shell=True)
        # ??? ??? ??? ??
        word_loader = CharLabelLoader()
        word_loader.load_char2index('./hackathon.pos.labels')
        label_loader = word_loader
        if os.path.exists(TRAIN_LABEL_CHAR_PATH):
            generate_word_label_file(char_loader, word_loader, TRAIN_LABEL_POS_PATH, TRAIN_LABEL_CHAR_PATH)
    char2index = label_loader.char2index
    index2char = label_loader.index2char
    SOS_token = char2index['<s>']
    EOS_token = char2index['</s>']
    PAD_token = char2index['_']
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if args.cuda else 'cpu')

    ############ model
    print("model: transformer")
    # model = Transformer(d_model= args.d_model, n_head= args.n_head, num_encoder_layers= args.num_encoder_layers, num_decoder_layers= args.num_decoder_layers,
    #                     dim_feedforward= args.dim_feedforward, dropout= args.dropout, vocab_size= len(char2index), sound_maxlen= SOUND_MAXLEN, word_maxlen= WORD_MAXLEN)

    encoder = Encoder(d_input=128, n_layers=6, n_head=4,
                      d_k=128, d_v=128, d_model=128, d_inner=2048,
                      dropout=0.1, pe_maxlen=SOUND_MAXLEN)
    decoder = Decoder(sos_id= SOS_token, eos_id=EOS_token, n_tgt_vocab= len(char2index),
                      d_word_vec=128, n_layers=6, n_head=4,
                      d_k=128, d_v=128, d_model=128, d_inner=2048,
                      dropout=0.1,
                      tgt_emb_prj_weight_sharing=True,
                      pe_maxlen=SOUND_MAXLEN)
    model = Transformer(encoder, decoder)

    optimizer = TransformerOptimizer(
        torch.optim.Adam(model.parameters(), lr=0.0004, betas=(0.9, 0.98), eps=1e-09))

    ############/



    for param in model.parameters():
        param.data.uniform_(-0.08, 0.08)

    model = nn.DataParallel(model).to(device)

    """
    optimizer = optim.Adam(model.module.parameters(), lr=args.lr)

    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epochs)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=args.warmup_multiplier, total_epoch=args.warmup_epoch, after_scheduler=scheduler_cosine)
    
    
    criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=PAD_token).to(device)
    """

    bind_model(model, optimizer)

    if args.pause == 1:
        nsml.paused(scope=locals())

    if args.mode != "train":
        return

    data_list = os.path.join(DATASET_PATH, 'train_data', 'data_list.csv')
    wav_paths = list()
    script_paths = list()

    with open(data_list, 'r') as f:
        for line in f:
            # line: "aaa.wav,aaa.label"

            wav_path, script_path = line.strip().split(',')
            wav_paths.append(os.path.join(DATASET_PATH, 'train_data', wav_path))
            script_paths.append(os.path.join(DATASET_PATH, 'train_data', script_path))

    best_loss = 1e10
    begin_epoch = 0

    # load all target scripts for reducing disk i/o
    # target_path = os.path.join(DATASET_PATH, 'train_label')
    target_path = TRAIN_LABEL_CHAR_PATH
    if args.word:
        target_path = TRAIN_LABEL_POS_PATH
    load_targets(target_path)

    train_batch_num, train_dataset_list, valid_dataset = split_dataset(args, wav_paths, script_paths, valid_ratio=0.05)


    if args.iteration:
        if args.premodel_session:
            nsml.load(args.iteration, session=args.premodel_session)
            logger.info(f'Load {args.premodel_session} {args.iteration}')
        else:
            nsml.load(args.iteration)
            logger.info(f'Load {args.iteration}')
    logger.info('start')

    train_begin = time.time()

    for epoch in range(begin_epoch, args.max_epochs):
        # learning rate scheduler



        train_queue = queue.Queue(args.workers * 2)

        train_loader = MultiLoader(train_dataset_list, train_queue, args.batch_size, args.workers)
        train_loader.start()
        
        train_loss, train_cer = train(model, train_batch_num, train_queue, optimizer, device, train_begin,
                                      args.workers, 10, args.teacher_forcing)
        logger.info('Epoch %d (Training) Loss %0.4f CER %0.4f' % (epoch, train_loss, train_cer))
        
        train_loader.join()


        print("~~~~~~~~~~~~")

        if epoch ==10 or (epoch >48 and epoch%10==9):
            valid_queue = queue.Queue(args.workers * 2)
            valid_loader = BaseDataLoader(valid_dataset, valid_queue, args.batch_size, 0)
            valid_loader.start()

            eval_loss, eval_cer = evaluate(model, valid_loader, valid_queue, device, args.max_len,
                                           args.batch_size)
            logger.info('Epoch %d (Evaluate) Loss %0.4f CER %0.4f' % (epoch, eval_loss, eval_cer))

            valid_loader.join()

            nsml.report(False,
                        step=epoch, train_epoch__loss=train_loss, train_epoch__cer=train_cer,
                        eval__loss=eval_loss, eval__cer=eval_cer)

            best_model = (eval_loss < best_loss)
            nsml.save(args.save_name)

            if best_model:
                nsml.save('best')
                best_loss = eval_loss


if __name__ == "__main__":
    main()
