from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
from pprint import pprint

import torch
from torch import nn

# from data_utils.utils import Config
# from data_utils.vocab_tokenizer import Vocabulary
from transformer_model.embedding.embeddings import Embeddings

PAD_token = 0
SOUND_PAD_token = -100

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class Transformer(nn.Module):
    def __init__(self, d_model, n_head, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, vocab_size, sound_maxlen, word_maxlen) -> None:
        """Instantiating Transformer class
        Args:
            config (Config): model config, the instance of data_utils.utils.Config
            vocab (Vocabulary): the instance of data_utils.vocab_tokenizer.Vocabulary
        """
        super(Transformer, self).__init__()
        d_model = d_model #256
        n_head = n_head #8
        num_encoder_layers = num_encoder_layers #6
        num_decoder_layers = num_decoder_layers #6
        dim_feedforward = dim_feedforward #2048
        dropout = dropout #0.1
        vocab_size= vocab_size
        sound_maxlen= sound_maxlen
        word_maxlen= word_maxlen



        self.input_embedding_sound = Embeddings(d_model, sound_maxlen, vocab_size, SOUND_PAD_token, False)
        self.input_embedding_word = Embeddings(d_model, word_maxlen, vocab_size, PAD_token, True)


        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(5, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3)),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2)),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(5, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True)
        )


        self.transfomrer = torch.nn.Transformer(d_model=d_model,
                             nhead=n_head,
                             num_encoder_layers=num_encoder_layers,
                             num_decoder_layers=num_decoder_layers,
                             dim_feedforward=dim_feedforward,
                             dropout=dropout)

        self.proj_vocab_layer = nn.Linear(in_features=d_model, out_features=vocab_size)

        self.apply(self._initailze) # https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module.apply

    def forward(self, enc_input: torch.Tensor, dec_input: torch.Tensor) -> torch.Tensor:
        # [4, 3002, 128]
        input_var = enc_input.unsqueeze(1)
        # [4, 1, 3002, 128]
        #print(f'input_var.size: {input_var.size()}')
        x = self.conv(input_var)
        # [4, 32, 751, 6e]
        #print(f'x: {x.size()}')
        # BxCxTxD => BxCxDxT
        x = x.transpose(1, 2)
        # [4, 751, 32, 64]
        #print(f'x: {x.size()}')
        x = x.contiguous()
        # [4, 751, 32, 64]
        #print(f'x: {x.size()}')
        sizes = x.size()
        x = x.view(sizes[0], sizes[1], sizes[2] * sizes[3])
        # [4, 751, 2048]
        #print(f'x: {x.size()}')

        x_dec_embed = self.input_embedding_word(dec_input.long())
        

        # Masking
        src_key_padding_mask = enc_input == SOUND_PAD_token # tensor([[False, False, False,  True,  ...,  True]])
        src_key_padding_mask=src_key_padding_mask[:,:,0]
        tgt_key_padding_mask = dec_input == PAD_token
        tgt_key_padding_mask = tgt_key_padding_mask
        memory_key_padding_mask = src_key_padding_mask
        tgt_mask = self.transfomrer.generate_square_subsequent_mask(dec_input.size(1))
        
        

        # einsum ref: https://pytorch.org/docs/stable/torch.html#torch.einsum
        # https://obilaniu6266h16.wordpress.com/2016/02/04/einstein-summation-in-numpy/
        #x_enc_embed = torch.einsum('ijk->jik', x_enc_embed)
        x = torch.einsum('ijk->jik', x)
        x_dec_embed = torch.einsum('ijk->jik', x_dec_embed)
        #print(f'x: {x.size()}')
        #print(f'x_dec_embed.size: {x_dec_embed.size()}')

        # transformer ref: https://pytorch.org/docs/stable/nn.html#torch.nn.Transformer
        # src: (S, N, E) -> S: source sequence length N: batch size, E: feature number
        # src: (S, N, E) -> S: target sequence length N: batch size, E: feature number
        feature = self.transfomrer(src = x,
                                   tgt = x_dec_embed,
                                   #src_key_padding_mask = src_key_padding_mask,
                                   tgt_key_padding_mask = tgt_key_padding_mask,
                                   #memory_key_padding_mask=memory_key_padding_mask,
                                   tgt_mask = tgt_mask.to(device)) # src: (S,N,E) tgt: (T,N,E)

        logits = self.proj_vocab_layer(feature)

        logits = torch.einsum('ijk->jik', logits)

        return logits

        """
        # [4, 3002, 128]
        print(f'enc_input.size: {enc_input.size()}')
        x_enc_embed = self.input_embedding_sound(enc_input.float())
        x_dec_embed = self.input_embedding_word(dec_input.long())
        
        # [4, 3002, 128]
        print(f'x_enc_embed.size: {x_enc_embed.size()}')

        # Masking
        src_key_padding_mask = enc_input == SOUND_PAD_token # tensor([[False, False, False,  True,  ...,  True]])
        src_key_padding_mask=src_key_padding_mask[:,:,0]
        tgt_key_padding_mask = dec_input == PAD_token
        tgt_key_padding_mask = tgt_key_padding_mask
        memory_key_padding_mask = src_key_padding_mask
        tgt_mask = self.transfomrer.generate_square_subsequent_mask(dec_input.size(1))
        
        

        # einsum ref: https://pytorch.org/docs/stable/torch.html#torch.einsum
        # https://obilaniu6266h16.wordpress.com/2016/02/04/einstein-summation-in-numpy/
        x_enc_embed = torch.einsum('ijk->jik', x_enc_embed)
        x_dec_embed = torch.einsum('ijk->jik', x_dec_embed)

        # [3002, 4, 128]
        print(f'x_enc_embed.size: {x_enc_embed.size()}')

        # transformer ref: https://pytorch.org/docs/stable/nn.html#torch.nn.Transformer
        feature = self.transfomrer(src = x_enc_embed,
                                   tgt = x_dec_embed,
                                   src_key_padding_mask = src_key_padding_mask,
                                   tgt_key_padding_mask = tgt_key_padding_mask,
                                   memory_key_padding_mask=memory_key_padding_mask,
                                   tgt_mask = tgt_mask.to(device)) # src: (S,N,E) tgt: (T,N,E)

        logits = self.proj_vocab_layer(feature)

        logits = torch.einsum('ijk->jik', logits)

        return logits
        """

    def _initailze(self, layer):
        if isinstance(layer, (nn.Linear)):
            nn.init.kaiming_uniform_(layer.weight)
