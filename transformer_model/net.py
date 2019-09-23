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



        self.input_embedding_sound = Embeddings(d_model, sound_maxlen, vocab_size, PAD_token, False)
        self.input_embedding_word = Embeddings(d_model, word_maxlen, vocab_size, PAD_token, True)




        self.transfomrer = torch.nn.Transformer(d_model=d_model,
                             nhead=n_head,
                             num_encoder_layers=num_encoder_layers,
                             num_decoder_layers=num_decoder_layers,
                             dim_feedforward=dim_feedforward,
                             dropout=dropout)

        self.proj_vocab_layer = nn.Linear(in_features=d_model, out_features=vocab_size)

        self.apply(self._initailze) # https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module.apply

    def forward(self, enc_input: torch.Tensor, dec_input: torch.Tensor) -> torch.Tensor:

        x_enc_embed = self.input_embedding_sound(enc_input.float())
        x_dec_embed = self.input_embedding_word(dec_input.long())


        # Masking
        src_key_padding_mask = enc_input == 0 # tensor([[False, False, False,  True,  ...,  True]])
        src_key_padding_mask=src_key_padding_mask[:,:,0]
        tgt_key_padding_mask = dec_input == 0
        tgt_key_padding_mask = tgt_key_padding_mask
        memory_key_padding_mask = src_key_padding_mask
        tgt_mask = self.transfomrer.generate_square_subsequent_mask(dec_input.size(1))


        # einsum ref: https://pytorch.org/docs/stable/torch.html#torch.einsum
        # https://obilaniu6266h16.wordpress.com/2016/02/04/einstein-summation-in-numpy/
        x_enc_embed = torch.einsum('ijk->jik', x_enc_embed)
        x_dec_embed = torch.einsum('ijk->jik', x_dec_embed)


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

    def _initailze(self, layer):
        if isinstance(layer, (nn.Linear)):
            nn.init.kaiming_uniform_(layer.weight)